"""Flower Dashboard: monitor federated learning rounds in real time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Flower Dashboard",
    page_icon="🌸",
    layout="wide",
    menu_items={
        "About": "Interactive dashboard for monitoring Flower federated learning runs.",
    },
)


@st.cache_data(show_spinner=False)
def load_round_data(path: Path) -> Dict:
    """Load round metrics from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_rounds(raw: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-round and per-client DataFrames from raw payload."""
    rounds = raw.get("rounds", [])
    if not rounds:
        return pd.DataFrame(), pd.DataFrame()

    rounds_df = pd.DataFrame(rounds)
    rounds_df["round"] = rounds_df["round"].astype(int)
    rounds_df = rounds_df.sort_values("round")

    client_rows: List[Dict] = []
    for round_info in rounds:
        round_number = round_info.get("round")
        for client in round_info.get("clients", []):
            client_rows.append({**client, "round": round_number})

    clients_df = pd.DataFrame(client_rows)
    if not clients_df.empty:
        clients_df["round"] = clients_df["round"].astype(int)
        clients_df = clients_df.sort_values(["round", "client_id"])
    return rounds_df, clients_df


def compute_stragglers(clients_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Flag clients whose duration is longer than the configured threshold."""
    if clients_df.empty or "duration" not in clients_df:
        return pd.DataFrame(columns=clients_df.columns if not clients_df.empty else [])

    durations = clients_df["duration"].replace({None: np.nan}).astype(float)
    cutoff = durations.quantile(threshold)
    mask = durations >= cutoff
    stragglers = clients_df.loc[mask].copy()
    stragglers["duration_cutoff"] = round(cutoff, 2)
    return stragglers


def compute_anomalies(clients_df: pd.DataFrame, metric: str, z_thresh: float) -> pd.DataFrame:
    """Detect anomalous client updates using a z-score threshold."""
    if clients_df.empty or metric not in clients_df:
        return pd.DataFrame(columns=clients_df.columns if not clients_df.empty else [])

    metric_series = clients_df[metric].replace({None: np.nan}).astype(float).dropna()
    if metric_series.std(ddof=0) == 0 or metric_series.empty:
        return pd.DataFrame(columns=clients_df.columns if not clients_df.empty else [])

    zscores = (metric_series - metric_series.mean()) / metric_series.std(ddof=0)
    indices = zscores.index[np.abs(zscores) >= z_thresh]
    anomalies = clients_df.loc[indices].copy()
    anomalies["z_score"] = zscores.loc[indices].round(2)
    anomalies["metric"] = metric
    return anomalies


def build_loss_accuracy_chart(rounds_df: pd.DataFrame) -> alt.Chart:
    melted = rounds_df.melt(id_vars="round", value_vars=["loss", "accuracy"], var_name="metric")
    return (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("value:Q", title="Metric value"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["round", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=300)
    )


def build_participation_chart(clients_df: pd.DataFrame) -> alt.Chart:
    counts = clients_df.groupby(["round", "status"], dropna=False)["client_id"].count().reset_index()
    counts.rename(columns={"client_id": "count"}, inplace=True)
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("count:Q", title="Client count"),
            color=alt.Color("status:N", title="Status"),
            tooltip=["round", "status", "count"],
        )
        .properties(height=300)
    )


def build_contribution_chart(clients_df: pd.DataFrame, contribution_field: str) -> alt.Chart:
    contributions = (
        clients_df.groupby(["round", "client_id"], dropna=False)[contribution_field]
        .sum()
        .reset_index()
    )
    contributions.rename(columns={contribution_field: "contribution"}, inplace=True)
    return (
        alt.Chart(contributions)
        .mark_bar()
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("contribution:Q", title=contribution_field.capitalize()),
            color=alt.Color("client_id:N", title="Client"),
            tooltip=["round", "client_id", alt.Tooltip("contribution:Q", format=".2f")],
        )
        .properties(height=300)
    )


st.title("🌸 Flower Dashboard")
st.caption(
    "Plug-and-play analytics for Flower federated learning runs. "
    "Drop a metrics export or connect to a live pipeline to inspect participation, metrics, and anomalies."
)

with st.sidebar:
    st.header("Data source")
    default_path = Path(__file__).parent / "assets" / "sample_metrics.json"
    use_sample = st.toggle("Use bundled sample data", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload metrics JSON", type=["json"])

    refresh_button = st.button("Refresh data")

    st.divider()
    st.header("Detection settings")
    duration_quantile = st.slider(
        "Straggler quantile threshold", min_value=0.5, max_value=0.99, value=0.9, step=0.01
    )
    anomaly_metric = st.selectbox(
        "Metric for anomaly detection", options=["loss", "accuracy", "duration"], index=0
    )
    z_score_threshold = st.slider("Anomaly z-score", min_value=1.0, max_value=3.5, value=2.5, step=0.1)

if refresh_button:
    st.experimental_rerun()

if uploaded is not None:
    raw_data = json.load(uploaded)
elif default_path.exists():
    raw_data = load_round_data(default_path)
else:
    st.error("No data available. Please upload a metrics JSON file.")
    st.stop()

rounds_df, clients_df = parse_rounds(raw_data)

if rounds_df.empty:
    st.warning("No round-level metrics found in the provided file.")
    st.stop()

st.subheader("Training overview")
metrics_container = st.container()
col1, col2, col3, col4 = metrics_container.columns(4)
col1.metric("Rounds", int(rounds_df["round"].max()))
col2.metric("Best accuracy", f"{rounds_df['accuracy'].max():.3f}")
col3.metric("Final loss", f"{rounds_df['loss'].iloc[-1]:.3f}")
if "server_time" in rounds_df:
    col4.metric("Avg server time", f"{rounds_df['server_time'].mean():.2f}s")
else:
    col4.metric("Data source", "static")

loss_accuracy_chart = build_loss_accuracy_chart(rounds_df)
st.altair_chart(loss_accuracy_chart, use_container_width=True)

if clients_df.empty:
    st.warning("Client-level metrics missing. Participation and anomaly sections skipped.")
    st.stop()

st.subheader("Client participation")
st.altair_chart(build_participation_chart(clients_df), use_container_width=True)

st.subheader("Round contributions")
contribution_field = "examples" if "examples" in clients_df else "weight"
st.altair_chart(
    build_contribution_chart(clients_df, contribution_field), use_container_width=True
)

st.subheader("Alerts")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("#### Stragglers")
    stragglers = compute_stragglers(clients_df, duration_quantile)
    if stragglers.empty:
        st.success("No stragglers detected with the current threshold.")
    else:
        st.dataframe(
            stragglers[["round", "client_id", "duration", "status"]],
            use_container_width=True,
        )

with col_b:
    st.markdown("#### Dropped clients")
    dropped = clients_df[clients_df["status"].str.lower().isin(["dropped", "failed", "timeout"])]
    if dropped.empty:
        st.success("No dropped clients.")
    else:
        st.dataframe(dropped[["round", "client_id", "status", "duration"]], use_container_width=True)

with col_c:
    st.markdown("#### Anomalous updates")
    anomalies = compute_anomalies(clients_df, anomaly_metric, z_score_threshold)
    if anomalies.empty:
        st.success("No anomalies detected with current settings.")
    else:
        st.dataframe(
            anomalies[["round", "client_id", anomaly_metric, "z_score", "status"]],
            use_container_width=True,
        )

st.divider()

st.subheader("Raw data preview")
st.dataframe(clients_df, use_container_width=True, hide_index=True)
