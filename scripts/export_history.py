import asyncio
import csv
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice


# BLE-Identifier aus der Geraetesuche einsetzen.
# macOS: CoreBluetooth-UUID; Linux/Windows: normalerweise Bluetooth-Adresse.
SENSOR_IDENTIFIER = "15B130E0-B0EC-9593-5B93-90A5B3DF411E"

# LYWSD03MMC Characteristics
TIME_UUID = "EBE0CCB7-7A0A-4B0C-8A1A-6FF2997DA3A6"
COUNT_UUID = "EBE0CCB9-7A0A-4B0C-8A1A-6FF2997DA3A6"
INDEX_UUID = "EBE0CCBA-7A0A-4B0C-8A1A-6FF2997DA3A6"
HISTORY_UUID = "EBE0CCBC-7A0A-4B0C-8A1A-6FF2997DA3A6"

OUTPUT_FILE = Path("../mi_history.csv")

# Kleine Batches reduzieren Probleme bei langen BLE-Notification-Streams.
BATCH_SIZE = 24
SCAN_TIMEOUT = 20
CONNECT_TIMEOUT = 60
BATCH_TIMEOUT = 45
CONNECT_RETRIES = 3

CSV_FIELDS = [
    "idx",
    "timestamp_device",
    "timestamp_corrected",
    "datetime_local",
    "temperature_min_c",
    "temperature_max_c",
    "humidity_min_percent",
    "humidity_max_percent",
]


@dataclass(frozen=True)
class HistoryRow:
    idx: int
    timestamp_device: int
    temperature_max: float
    humidity_max: int
    temperature_min: float
    humidity_min: int


def corrected_timestamp(timestamp_device: int, time_offset_seconds: int) -> int:
    return timestamp_device + time_offset_seconds


def datetime_local(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def row_to_csv(row: HistoryRow, time_offset_seconds: int) -> dict[str, object]:
    timestamp = corrected_timestamp(row.timestamp_device, time_offset_seconds)
    return {
        "idx": row.idx,
        "timestamp_device": row.timestamp_device,
        "timestamp_corrected": timestamp,
        "datetime_local": datetime_local(timestamp),
        "temperature_min_c": row.temperature_min,
        "temperature_max_c": row.temperature_max,
        "humidity_min_percent": row.humidity_min,
        "humidity_max_percent": row.humidity_max,
    }


def decode_history_row(data: bytearray) -> HistoryRow:
    if len(data) < struct.calcsize("<IIhBhB"):
        raise ValueError(f"Unerwartete Paketlaenge: {len(data)} Bytes")

    idx, timestamp, temp_max_raw, humidity_max, temp_min_raw, humidity_min = (
        struct.unpack_from("<IIhBhB", data)
    )

    return HistoryRow(
        idx=idx,
        timestamp_device=timestamp,
        temperature_max=temp_max_raw / 10,
        humidity_max=humidity_max,
        temperature_min=temp_min_raw / 10,
        humidity_min=humidity_min,
    )


async def find_sensor() -> BLEDevice:
    print(f"Suche Sensor {SENSOR_IDENTIFIER} ...")
    device = await BleakScanner.find_device_by_address(
        SENSOR_IDENTIFIER, timeout=SCAN_TIMEOUT
    )

    if device is None:
        raise RuntimeError(
            "Sensor nicht gefunden. Fuehre die Geraetesuche erneut aus "
            "und aktualisiere SENSOR_IDENTIFIER."
        )

    print(f"Gefunden: {device.name or 'unbekannt'} / {device.address}")
    return device


async def connect(device: BLEDevice) -> BleakClient:
    last_error: Exception | None = None

    for attempt in range(1, CONNECT_RETRIES + 1):
        client = BleakClient(device, timeout=CONNECT_TIMEOUT)

        try:
            await client.connect()
            return client
        except Exception as error:
            last_error = error
            if client.is_connected:
                await client.disconnect()

            if attempt < CONNECT_RETRIES:
                print(f"Verbindungsversuch {attempt} fehlgeschlagen; versuche erneut ...")
                await asyncio.sleep(3)

    raise RuntimeError("Bluetooth-Verbindung zum Sensor fehlgeschlagen.") from last_error


async def read_metadata(device: BLEDevice) -> tuple[int, int, int, int]:
    client = await connect(device)

    try:
        bounds = await client.read_gatt_char(COUNT_UUID)
        latest_idx, status_second_value = struct.unpack_from("<II", bounds)

        host_time_before = time.time()
        raw_device_time = await client.read_gatt_char(TIME_UUID)
        host_time_after = time.time()

        timestamp_device_now = struct.unpack_from("<I", raw_device_time)[0]
        timestamp_mac_now = int((host_time_before + host_time_after) / 2)
        time_offset_seconds = timestamp_mac_now - timestamp_device_now

        return latest_idx, status_second_value, timestamp_device_now, time_offset_seconds
    finally:
        await client.disconnect()


async def read_first_available(device: BLEDevice) -> HistoryRow:
    result: HistoryRow | None = None
    received = asyncio.Event()
    client = await connect(device)

    try:
        await client.write_gatt_char(INDEX_UUID, struct.pack("<I", 0), response=True)

        def callback(_, data: bytearray) -> None:
            nonlocal result
            if result is None:
                result = decode_history_row(data)
                received.set()

        await client.start_notify(HISTORY_UUID, callback)

        try:
            await asyncio.wait_for(received.wait(), timeout=BATCH_TIMEOUT)
        finally:
            await client.stop_notify(HISTORY_UUID)
    finally:
        await client.disconnect()

    if result is None:
        raise RuntimeError("Kein History-Datensatz empfangen.")

    return result


async def read_batch(
    device: BLEDevice, start_idx: int, end_idx: int
) -> list[HistoryRow]:
    rows: dict[int, HistoryRow] = {}
    finished = asyncio.Event()
    client = await connect(device)

    try:
        await client.write_gatt_char(
            INDEX_UUID, struct.pack("<I", start_idx), response=True
        )

        def callback(_, data: bytearray) -> None:
            row = decode_history_row(data)

            if start_idx <= row.idx <= end_idx:
                rows[row.idx] = row

            if row.idx >= end_idx:
                finished.set()

        await client.start_notify(HISTORY_UUID, callback)

        try:
            await asyncio.wait_for(finished.wait(), timeout=BATCH_TIMEOUT)
        except TimeoutError:
            print(
                f"  Timeout fuer Batch {start_idx}-{end_idx}; "
                f"{len(rows)} Records empfangen."
            )
        finally:
            await client.stop_notify(HISTORY_UUID)
    finally:
        await client.disconnect()

    return sorted(rows.values(), key=lambda row: row.idx)


def read_existing_rows_and_fix_timestamps(
    time_offset_seconds: int,
) -> dict[int, dict[str, object]]:
    if not OUTPUT_FILE.exists():
        return {}

    existing_rows: dict[int, dict[str, object]] = {}

    with OUTPUT_FILE.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            idx = int(row["idx"])
            timestamp_device_raw = row.get("timestamp_device") or row.get("timestamp")

            if timestamp_device_raw is None:
                raise RuntimeError(
                    "Bestehende CSV enthaelt keinen Device-Timestamp. "
                    "Benenne mi_history.csv um und starte neu."
                )

            timestamp_device = int(timestamp_device_raw)
            timestamp = corrected_timestamp(timestamp_device, time_offset_seconds)

            existing_rows[idx] = {
                "idx": idx,
                "timestamp_device": timestamp_device,
                "timestamp_corrected": timestamp,
                "datetime_local": datetime_local(timestamp),
                "temperature_min_c": row["temperature_min_c"],
                "temperature_max_c": row["temperature_max_c"],
                "humidity_min_percent": row["humidity_min_percent"],
                "humidity_max_percent": row["humidity_max_percent"],
            }

    write_all_rows(existing_rows)
    return existing_rows


def write_all_rows(rows: dict[int, dict[str, object]]) -> None:
    temp_file = OUTPUT_FILE.with_suffix(".csv.tmp")

    with temp_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for idx in sorted(rows):
            writer.writerow(rows[idx])

    temp_file.replace(OUTPUT_FILE)


def store_rows(
    new_rows: list[HistoryRow],
    existing_rows: dict[int, dict[str, object]],
    time_offset_seconds: int,
) -> int:
    written = 0

    for row in new_rows:
        if row.idx not in existing_rows:
            written += 1
        existing_rows[row.idx] = row_to_csv(row, time_offset_seconds)

    if new_rows:
        write_all_rows(existing_rows)

    return written


def find_clock_jumps(rows: dict[int, dict[str, object]]) -> list[tuple[int, int]]:
    jumps = []
    sorted_indices = sorted(rows)

    for previous_idx, current_idx in zip(sorted_indices, sorted_indices[1:]):
        if current_idx != previous_idx + 1:
            continue

        previous_timestamp = int(rows[previous_idx]["timestamp_device"])
        current_timestamp = int(rows[current_idx]["timestamp_device"])
        if current_timestamp - previous_timestamp != 3600:
            jumps.append((previous_idx, current_idx))

    return jumps


async def main() -> None:
    if SENSOR_IDENTIFIER == "DEIN-BLE-IDENTIFIER-HIER":
        raise SystemExit("Bitte zuerst SENSOR_IDENTIFIER im Script einsetzen.")

    device = await find_sensor()
    latest_idx, status_second_value, device_now, offset = await read_metadata(device)

    print(f"Neuster abgeschlossener Stunden-Record: {latest_idx}")
    print(f"Zweites Statusfeld des Sensors:          {status_second_value}")
    print(f"Interne Sensorzeit:                      {datetime_local(device_now)}")
    print(f"Zeitkorrektur zur Mac-Zeit:              {timedelta(seconds=offset)}")

    first_row = await read_first_available(device)
    first_timestamp = corrected_timestamp(first_row.timestamp_device, offset)

    print(f"Aeltester lieferbarer Record:             {first_row.idx}")
    print(f"Aeltester lieferbarer Zeitpunkt:          {datetime_local(first_timestamp)}")

    existing_rows = read_existing_rows_and_fix_timestamps(offset)
    if existing_rows:
        print(
            f"{len(existing_rows)} bestehende CSV-Records mit korrigierten "
            "Zeitstempeln uebernommen."
        )

    new_records = 0

    for start_idx in range(first_row.idx, latest_idx + 1, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE - 1, latest_idx)

        if all(idx in existing_rows for idx in range(start_idx, end_idx + 1)):
            print(f"Batch {start_idx}-{end_idx} bereits vorhanden.")
            continue

        print(f"Lese Batch {start_idx}-{end_idx} ...")
        rows = await read_batch(device, start_idx, end_idx)
        written = store_rows(rows, existing_rows, offset)
        new_records += written
        print(f"  {written} neue Records gespeichert.")
        await asyncio.sleep(0.5)

    print()
    print(f"Export abgeschlossen: {OUTPUT_FILE.resolve()}")
    print(f"Neue Records in diesem Lauf: {new_records}")
    print(f"Total Records in CSV:        {len(existing_rows)}")

    if existing_rows:
        first = existing_rows[min(existing_rows)]["datetime_local"]
        last = existing_rows[max(existing_rows)]["datetime_local"]
        print(f"Zeitraum:                    {first} bis {last}")

    jumps = find_clock_jumps(existing_rows)
    if jumps:
        print()
        print(
            "WARNUNG: In der internen Sensorzeit wurden Spruenge gefunden. "
            "Falls die Sensoruhr waehrend des Zeitraums gestellt oder "
            "zurueckgesetzt wurde, muessen einzelne Abschnitte separat "
            "korrigiert werden."
        )
        print(f"Betroffene Uebergaenge: {jumps[:10]}")


if __name__ == "__main__":
    asyncio.run(main())

