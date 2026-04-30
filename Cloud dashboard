"""
cloud_dashboard.py
Deploy this to Streamlit Cloud.
Reads live prediction history from Google Sheets and displays it.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials
import datetime
import json

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Fault Diagnosis AI — Live Monitor",
    page_icon="⚙️",
    layout="wide"
)

SHEET_NAME = "FaultDiagnosisLog"

CLASS_COLORS = {
    "Healthy":      "#3fb950",
    "Misalignment": "#d29922",
    "Unbalance":    "#f78166",
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# =========================================================
# GOOGLE SHEETS CONNECTION
# =========================================================
@st.cache_resource
def get_worksheet():
    """Connect using credentials stored in Streamlit secrets."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds      = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client     = gspread.authorize(creds)
    return client.open(SHEET_NAME).sheet1


@st.cache_data(ttl=5)   # refresh every 5 seconds
def fetch_data():
    """Pull all rows from the sheet and return as DataFrame."""
    try:
        ws      = get_worksheet()
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()


# =========================================================
# HEADER
# =========================================================
st.title("⚙️ Fault Diagnosis AI — Live Cloud Monitor")
st.caption(
    "British University in Egypt  |  MTRN_RP20  |  "
    "Smart Predictive Maintenance System  |  Auto-refreshes every 5 s"
)

st.markdown("---")

# Auto-refresh button + manual refresh
col_r1, col_r2, col_r3 = st.columns([1, 1, 4])
with col_r1:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_r2:
    auto_refresh = st.toggle("Auto-refresh", value=True)

df = fetch_data()

if df.empty:
    st.info("No predictions logged yet. Start the local dashboard and run a live diagnosis.")
    st.stop()

# =========================================================
# LATEST PREDICTION — BIG CARDS
# =========================================================
latest = df.iloc[-1]
pred   = str(latest["Prediction"])
color  = CLASS_COLORS.get(pred, "#388bfd")
conf   = float(latest["Confidence (%)"])
rpm    = float(latest["Speed (RPM)"])
ts     = str(latest["Timestamp"])

st.markdown("### 🔴 Latest Reading")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center;"
        f"border:2px solid {color}'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>PREDICTION</p>"
        f"<p style='margin:6px 0 0;color:{color};font-size:32px;font-weight:700'>{pred}</p>"
        f"</div>", unsafe_allow_html=True)
with c2:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>CONFIDENCE</p>"
        f"<p style='margin:6px 0 0;color:#e6edf3;font-size:32px;font-weight:700'>{conf:.1f}%</p>"
        f"</div>", unsafe_allow_html=True)
with c3:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>SPEED</p>"
        f"<p style='margin:6px 0 0;color:#e6edf3;font-size:32px;font-weight:700'>{rpm:.0f} RPM</p>"
        f"</div>", unsafe_allow_html=True)
with c4:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>LAST UPDATED</p>"
        f"<p style='margin:6px 0 0;color:#e6edf3;font-size:14px;font-weight:500'>{ts}</p>"
        f"</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# CHARTS
# =========================================================
col_left, col_right = st.columns(2)

with col_left:
    # Prediction history timeline
    fig_hist = go.Figure()
    for cls, clr in CLASS_COLORS.items():
        mask = df["Prediction"] == cls
        fig_hist.add_trace(go.Scatter(
            x=df.loc[mask, "Timestamp"],
            y=df.loc[mask, "Prediction"],
            mode="markers",
            name=cls,
            marker=dict(color=clr, size=12, symbol="circle"),
        ))
    fig_hist.update_layout(
        title="Prediction History",
        xaxis_title="Time",
        yaxis=dict(
            categoryorder="array",
            categoryarray=["Healthy", "Misalignment", "Unbalance"]
        ),
        height=300,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", y=-0.3)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    # Confidence over time
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Confidence (%)"],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(
            color=[CLASS_COLORS.get(p, "#388bfd") for p in df["Prediction"]],
            size=8
        ),
        name="Confidence"
    ))
    fig_conf.add_hline(y=80, line_dash="dot", line_color="#d29922",
                       annotation_text="80% threshold")
    fig_conf.update_layout(
        title="Confidence Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 105],
        height=300,
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_conf, use_container_width=True)

# Speed over time
fig_rpm = go.Figure()
fig_rpm.add_trace(go.Scatter(
    x=df["Timestamp"],
    y=df["Speed (RPM)"],
    mode="lines+markers",
    line=dict(color="#a371f7", width=2),
    marker=dict(size=6),
    name="Speed"
))
fig_rpm.update_layout(
    title="Motor Speed Over Time",
    xaxis_title="Time",
    yaxis_title="Speed (RPM)",
    height=250,
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig_rpm, use_container_width=True)

# =========================================================
# FAULT DISTRIBUTION PIE
# =========================================================
col_pie, col_table = st.columns([1, 2])

with col_pie:
    counts = df["Prediction"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=[CLASS_COLORS.get(c, "#388bfd") for c in counts.index],
        hole=0.4,
        textinfo="label+percent"
    ))
    fig_pie.update_layout(
        title="Fault Distribution",
        height=300,
        margin=dict(t=40, b=10)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    st.markdown("### 📋 Recent Predictions")
    display_df = df[["Timestamp", "Prediction", "Confidence (%)", "Speed (RPM)", "RMS (g)"]].tail(20)
    display_df = display_df.sort_values("Timestamp", ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# =========================================================
# SUMMARY STATS
# =========================================================
st.markdown("---")
st.markdown("### 📊 Session Summary")
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Total Readings", len(df))
with s2:
    healthy_pct = (df["Prediction"] == "Healthy").mean() * 100
    st.metric("Healthy %", f"{healthy_pct:.1f}%")
with s3:
    st.metric("Avg Confidence", f"{df['Confidence (%)'].mean():.1f}%")
with s4:
    st.metric("Avg Speed", f"{df['Speed (RPM)'].mean():.0f} RPM")

# =========================================================
# AUTO REFRESH
# =========================================================
if auto_refresh:
    import time
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
