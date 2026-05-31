from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_CSV = Path(__file__).resolve().parent.parent / "mi_history.csv"
TIMEZONE = "Europe/Zurich"


st.set_page_config(
    page_title="Mi Temperature History",
    page_icon="🌡️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_local_csv(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time  # Cache-Key: Datei wird nach einer Aenderung neu geladen.
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    required_values = [
        "temperature_min_c",
        "temperature_max_c",
        "humidity_min_percent",
        "humidity_max_percent",
    ]
    missing = [column for column in required_values if column not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in der CSV: {', '.join(missing)}")

    if "datetime_local" in df.columns:
        timestamps = pd.to_datetime(df["datetime_local"], errors="coerce", utc=True)
    elif "timestamp_corrected" in df.columns:
        timestamps = pd.to_datetime(df["timestamp_corrected"], unit="s", errors="coerce", utc=True)
    elif "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=True)
    else:
        raise ValueError("Keine Zeitspalte gefunden: erwartet datetime_local oder timestamp_corrected.")

    df["datetime"] = timestamps.dt.tz_convert(TIMEZONE)

    numeric_columns = required_values + [column for column in ["idx", "timestamp_device", "timestamp_corrected"] if column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["datetime", *required_values]).sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)

    if df.empty:
        raise ValueError("Die CSV enthaelt keine gueltigen Messwerte.")

    df["temperature_mean_c"] = (df["temperature_min_c"] + df["temperature_max_c"]) / 2
    df["humidity_mean_percent"] = (df["humidity_min_percent"] + df["humidity_max_percent"]) / 2
    df["temperature_range_c"] = df["temperature_max_c"] - df["temperature_min_c"]
    df["humidity_range_percent"] = df["humidity_max_percent"] - df["humidity_min_percent"]
    return df


def aggregate_data(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    indexed = df.set_index("datetime")
    if resolution == "Stuendlich":
        return indexed

    frequency = {"Taeglich": "D", "Woechentlich": "W-MON"}[resolution]
    aggregated = indexed.resample(frequency).agg(
        temperature_min_c=("temperature_min_c", "min"),
        temperature_max_c=("temperature_max_c", "max"),
        temperature_mean_c=("temperature_mean_c", "mean"),
        humidity_min_percent=("humidity_min_percent", "min"),
        humidity_max_percent=("humidity_max_percent", "max"),
        humidity_mean_percent=("humidity_mean_percent", "mean"),
        records=("temperature_mean_c", "count"),
    )
    return aggregated.dropna(subset=["temperature_mean_c", "humidity_mean_percent"])


def make_band_chart(
        df: pd.DataFrame,
        minimum: str,
        maximum: str,
        mean: str,
        title: str,
        unit: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df[minimum],
            name="Minimum",
            mode="lines",
            line={"width": 0},
            hovertemplate=f"Minimum: %{{y:.1f}} {unit}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df[maximum],
            name="Min.–Max.-Bereich",
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            hovertemplate=f"Maximum: %{{y:.1f}} {unit}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df[mean],
            name="Mittelwert aus Min./Max.",
            mode="lines",
            hovertemplate=f"Mittel: %{{y:.1f}} {unit}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        yaxis_title=unit,
        xaxis_title=None,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return figure


def find_gaps(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("datetime").reset_index(drop=True)
    gap_hours = ordered["datetime"].diff().dt.total_seconds().div(3600)
    rows: list[dict[str, object]] = []

    for row_number, hours in gap_hours.items():
        if pd.notna(hours) and hours > 1.5:
            before = ordered.loc[row_number - 1, "datetime"]
            after = ordered.loc[row_number, "datetime"]
            rows.append(
                {
                    "Von": before,
                    "Bis": after,
                    "Fehlende Stunden": max(0, round(hours) - 1),
                }
            )

    return pd.DataFrame(rows)


def display_metric(label: str, value: float, unit: str) -> None:
    st.metric(label, f"{value:.1f} {unit}")


st.title("Mi Temperature & Humidity Monitor 2")
st.caption("Auswertung der exportierten Stunden-History. Pro Stunde liegen Minimum und Maximum vor, nicht jede einzelne Messung.")

uploaded_file = st.sidebar.file_uploader("Andere CSV oeffnen", type=["csv"])

try:
    if uploaded_file is not None:
        source_name = uploaded_file.name
        data = prepare_data(load_uploaded_csv(uploaded_file.getvalue()))
    else:
        if not DEFAULT_CSV.exists():
            st.info(f"Lege deine exportierte Datei neben dieses Script: `{DEFAULT_CSV.name}`")
            st.stop()
        source_name = DEFAULT_CSV.name
        data = prepare_data(load_local_csv(str(DEFAULT_CSV), DEFAULT_CSV.stat().st_mtime))
except Exception as error:
    st.error(f"CSV konnte nicht gelesen werden: {error}")
    st.stop()

if data["datetime"].dt.year.min() < 2000:
    st.error(
        "Die CSV enthaelt offenbar noch unkorrigierte Sensor-Zeitstempel aus 1970. "
        "Fuehre zuerst das korrigierte Export-Script erneut aus."
    )
    st.stop()

st.sidebar.subheader("Ansicht")
minimum_date = data["datetime"].dt.date.min()
maximum_date = data["datetime"].dt.date.max()
selected_dates = st.sidebar.date_input(
    "Zeitraum",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)
resolution = st.sidebar.selectbox("Aufloesung", ["Stuendlich", "Taeglich", "Woechentlich"], index=1)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates[0] if isinstance(selected_dates, tuple) else selected_dates

start = pd.Timestamp(start_date, tz=TIMEZONE)
end = pd.Timestamp(end_date, tz=TIMEZONE) + pd.Timedelta(days=1)
filtered = data[(data["datetime"] >= start) & (data["datetime"] < end)].copy()

if filtered.empty:
    st.warning("Fuer den ausgewaehlten Zeitraum sind keine Daten vorhanden.")
    st.stop()

chart_data = aggregate_data(filtered, resolution)
gaps = find_gaps(filtered)

st.caption(
    f"Quelle: `{source_name}` · Zeitraum: {filtered['datetime'].min():%d.%m.%Y %H:%M} bis "
    f"{filtered['datetime'].max():%d.%m.%Y %H:%M} · {len(filtered):,} Stunden-Records".replace(",", "'")
)

metric_columns = st.columns(6)
with metric_columns[0]:
    display_metric("Temperatur min.", filtered["temperature_min_c"].min(), "°C")
with metric_columns[1]:
    display_metric("Temperatur mittel", filtered["temperature_mean_c"].mean(), "°C")
with metric_columns[2]:
    display_metric("Temperatur max.", filtered["temperature_max_c"].max(), "°C")
with metric_columns[3]:
    display_metric("Feuchte min.", filtered["humidity_min_percent"].min(), "%")
with metric_columns[4]:
    display_metric("Feuchte mittel", filtered["humidity_mean_percent"].mean(), "%")
with metric_columns[5]:
    display_metric("Feuchte max.", filtered["humidity_max_percent"].max(), "%")

st.plotly_chart(
    make_band_chart(
        chart_data,
        "temperature_min_c",
        "temperature_max_c",
        "temperature_mean_c",
        f"Temperaturverlauf ({resolution.lower()})",
        "°C",
    ),
    use_container_width=True,
)

st.plotly_chart(
    make_band_chart(
        chart_data,
        "humidity_min_percent",
        "humidity_max_percent",
        "humidity_mean_percent",
        f"Luftfeuchtigkeit ({resolution.lower()})",
        "%",
    ),
    use_container_width=True,
)

left, right = st.columns(2)

with left:
    temperature_histogram = px.histogram(
        filtered,
        x="temperature_mean_c",
        nbins=32,
        labels={"temperature_mean_c": "Temperaturmittel pro Stunde (°C)"},
        title="Verteilung der Temperatur",
    )
    temperature_histogram.update_layout(yaxis_title="Stunden", showlegend=False)
    st.plotly_chart(temperature_histogram, use_container_width=True)

with right:
    relationship = px.scatter(
        filtered,
        x="temperature_mean_c",
        y="humidity_mean_percent",
        labels={
            "temperature_mean_c": "Temperaturmittel (°C)",
            "humidity_mean_percent": "Feuchtemittel (%)",
        },
        title="Temperatur und Luftfeuchtigkeit",
        hover_data={"datetime": True},
        opacity=0.5,
    )
    relationship.update_traces(hovertemplate="%{x:.1f} °C · %{y:.1f} %<extra></extra>")
    st.plotly_chart(relationship, use_container_width=True)

st.subheader("Datenqualitaet")
quality_columns = st.columns(3)
with quality_columns[0]:
    st.metric("Records im Zeitraum", f"{len(filtered):,}".replace(",", "'"))
with quality_columns[1]:
    st.metric("Luecken", len(gaps))
with quality_columns[2]:
    missing_hours = int(gaps["Fehlende Stunden"].sum()) if not gaps.empty else 0
    st.metric("Fehlende Stunden", missing_hours)

if gaps.empty:
    st.success("Keine zeitlichen Luecken innerhalb des ausgewaehlten Zeitraums gefunden.")
else:
    st.dataframe(gaps, hide_index=True, use_container_width=True)

with st.expander("Rohdaten anzeigen"):
    visible_columns = [
        "datetime",
        "temperature_min_c",
        "temperature_max_c",
        "humidity_min_percent",
        "humidity_max_percent",
        *(["idx"] if "idx" in filtered.columns else []),
    ]
    st.dataframe(
        filtered[visible_columns].sort_values("datetime", ascending=False),
        hide_index=True,
        use_container_width=True,
    )

st.download_button(
    "Gefilterte Daten als CSV herunterladen",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="mi_history_filtered.csv",
    mime="text/csv",
)
