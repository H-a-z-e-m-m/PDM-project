"""
cloud_dashboard.py
Deploy this to Streamlit Cloud.
Reads live prediction history from Google Sheets and displays it.
"""

import time

import gspread
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Fault Diagnosis AI - Live Monitor",
    page_icon="gear",
    layout="wide",
)

SHEET_NAME = "FaultDiagnosisLog"

CLASS_COLORS = {
    "Healthy": "#3fb950",
    "Misalignment": "#d29922",
    "Unbalance": "#f78166",
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

EXPECTED_COLUMNS = [
    "Timestamp",
    "Prediction",
    "Confidence (%)",
    "Speed (RPM)",
    "RMS (g)",
    "Ratio 2x",
    "Ratio 3x",
    "Ratio 4x",
    "Prob Healthy",
    "Prob Misalignment",
    "Prob Unbalance",
    "Session ID",
    "Event Type",
    "Mode",
    "Source",
    "Model",
    "Model SHA",
]


# =========================================================
# GOOGLE SHEETS CONNECTION
# =========================================================
@st.cache_resource
def get_worksheet():
    """Connect using credentials stored in Streamlit secrets."""
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)
    try:
        client.set_timeout(12)
    except Exception:
        pass
    return client.open(SHEET_NAME).sheet1


def rows_to_records(values):
    if not values:
        return []

    headers = list(values[0])
    for col in EXPECTED_COLUMNS:
        if col not in headers:
            headers.append(col)

    records = []
    for row in values[1:]:
        padded = list(row) + [""] * max(0, len(headers) - len(row))
        record = {header: padded[idx] for idx, header in enumerate(headers)}
        records.append(record)
    return records


def has_values(df, column):
    if column not in df.columns:
        return False
    return df[column].astype(str).str.strip().ne("").any()


def latest_session_id(df):
    if df.empty or "Session ID" not in df.columns:
        return ""
    with_sessions = df[df["Session ID"].astype(str).str.strip().ne("")]
    if with_sessions.empty:
        return ""
    return str(with_sessions.iloc[-1]["Session ID"])


@st.cache_data(ttl=5)
def fetch_data():
    """Pull all rows from the sheet and return as DataFrame."""
    try:
        ws = get_worksheet()

        # Avoid gspread get_all_records(), which can fail on Streamlit Cloud with:
        # APIError [400]: Unable to parse range: 'Sheet1'
        values = ws.get("A1:Q10000")
        records = rows_to_records(values)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        event_type = df["Event Type"].astype(str).str.strip()
        df.loc[event_type.eq(""), "Event Type"] = "prediction"
        legacy_test = df["Prediction"].astype(str).str.strip().str.lower().eq("cloud logging test")
        df.loc[legacy_test, "Event Type"] = "test"

        df = df[df["Event Type"].astype(str).str.lower().eq("prediction")].copy()
        df = df[df["Prediction"].isin(CLASS_COLORS.keys())].copy()
        if df.empty:
            return pd.DataFrame()

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Confidence (%)"] = pd.to_numeric(df["Confidence (%)"], errors="coerce")
        df["Speed (RPM)"] = pd.to_numeric(df["Speed (RPM)"], errors="coerce")
        df["RMS (g)"] = pd.to_numeric(df["RMS (g)"], errors="coerce")

        df = df.dropna(subset=["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()


# =========================================================
# HEADER
# =========================================================
st.title("Fault Diagnosis AI - Live Cloud Monitor")
st.caption(
    "British University in Egypt | MTRN_RP20 | "
    "Smart Predictive Maintenance System | Auto-refreshes every 5 s"
)

st.markdown("---")

col_r1, col_r2, col_r3 = st.columns([1, 1, 4])
with col_r1:
    if st.button("Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_r2:
    auto_refresh = st.toggle("Auto-refresh", value=True)

df = fetch_data()

if df.empty:
    st.info("No predictions logged yet. Start the local dashboard and run a live diagnosis.")
    st.stop()


# =========================================================
# SESSION / FILTER CONTROLS
# =========================================================
latest_session = latest_session_id(df)
session_values = sorted([
    str(x) for x in df["Session ID"].dropna().unique()
    if str(x).strip()
])
mode_values = sorted([
    str(x) for x in df["Mode"].dropna().unique()
    if str(x).strip()
])

st.markdown("### Cloud View")
f1, f2, f3, f4 = st.columns([2.2, 1.6, 1.6, 1.2])
with f1:
    if session_values:
        session_options = ["All sessions", "Latest session"] + session_values
        session_choice = st.selectbox("Session", session_options, index=0)
    else:
        session_choice = "All sessions"
        st.caption("Older rows have no Session ID. New local-dashboard rows will appear with one.")
with f2:
    pred_filter = st.multiselect(
        "Prediction",
        list(CLASS_COLORS.keys()),
        default=list(CLASS_COLORS.keys()),
    )
with f3:
    if mode_values:
        mode_filter = st.multiselect("Mode", mode_values, default=mode_values)
    else:
        mode_filter = []
        st.caption("Mode metadata not found yet.")
with f4:
    download_slot = st.empty()

view_df = df.copy()
if session_choice == "Latest session" and latest_session:
    view_df = view_df[view_df["Session ID"].astype(str) == latest_session]
elif session_choice not in ("All sessions", "Latest session"):
    view_df = view_df[view_df["Session ID"].astype(str) == session_choice]
if pred_filter:
    view_df = view_df[view_df["Prediction"].isin(pred_filter)]
if mode_filter:
    view_df = view_df[view_df["Mode"].isin(mode_filter)]

if view_df.empty:
    st.warning("No cloud predictions match the selected filters.")
    if auto_refresh:
        time.sleep(5)
        st.cache_data.clear()
        st.rerun()
    st.stop()

with f4:
    download_slot.download_button(
        "Download CSV",
        view_df.to_csv(index=False).encode("utf-8"),
        file_name="cloud_predictions_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

meta_cols = st.columns(4)
with meta_cols[0]:
    st.metric("Visible Rows", len(view_df))
with meta_cols[1]:
    st.metric("Sessions", view_df["Session ID"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
with meta_cols[2]:
    st.metric("Latest Session", latest_session if latest_session else "Legacy")
with meta_cols[3]:
    st.metric("Last Mode", str(view_df.iloc[-1]["Mode"] or "Legacy"))

st.markdown("---")


# =========================================================
# LATEST PREDICTION - BIG CARDS
# =========================================================
latest = view_df.iloc[-1]
pred = str(latest["Prediction"])
color = CLASS_COLORS.get(pred, "#388bfd")
conf = float(latest["Confidence (%)"]) if pd.notna(latest["Confidence (%)"]) else 0.0
rpm = float(latest["Speed (RPM)"]) if pd.notna(latest["Speed (RPM)"]) else 0.0
ts = str(latest["Timestamp"])

st.markdown("### Latest Reading")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center;"
        f"border:2px solid {color}'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>PREDICTION</p>"
        f"<p style='margin:6px 0 0;color:{color};font-size:32px;font-weight:700'>{pred}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>CONFIDENCE</p>"
        f"<p style='margin:6px 0 0;color:#e6edf3;font-size:32px;font-weight:700'>{conf:.1f}%</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>SPEED</p>"
        f"<p style='margin:6px 0 0;color:#e6edf3;font-size:32px;font-weight:700'>{rpm:.0f} RPM</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background:#161b22;text-align:center'>"
        f"<p style='margin:0;color:#8b949e;font-size:13px'>LAST UPDATED</p>"
        f"<p style='margin:6px 0 0;color:#e6edf3;font-size:14px;font-weight:500'>{ts}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

if str(latest.get("Session ID", "")).strip():
    st.caption(
        f"Session: {latest['Session ID']} | "
        f"Mode: {latest.get('Mode', '') or 'n/a'} | "
        f"Source: {latest.get('Source', '') or 'n/a'}"
    )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# CHARTS
# =========================================================
col_left, col_right = st.columns(2)

with col_left:
    fig_hist = go.Figure()
    for cls, clr in CLASS_COLORS.items():
        mask = view_df["Prediction"] == cls
        fig_hist.add_trace(go.Scatter(
            x=view_df.loc[mask, "Timestamp"],
            y=view_df.loc[mask, "Prediction"],
            mode="markers",
            name=cls,
            marker=dict(color=clr, size=12, symbol="circle"),
        ))
    fig_hist.update_layout(
        title="Prediction History",
        xaxis_title="Time",
        yaxis=dict(
            categoryorder="array",
            categoryarray=["Healthy", "Misalignment", "Unbalance"],
        ),
        height=300,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", y=-0.3),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Scatter(
        x=view_df["Timestamp"],
        y=view_df["Confidence (%)"],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(
            color=[CLASS_COLORS.get(p, "#388bfd") for p in view_df["Prediction"]],
            size=8,
        ),
        name="Confidence",
    ))
    fig_conf.add_hline(
        y=80,
        line_dash="dot",
        line_color="#d29922",
        annotation_text="80% threshold",
    )
    fig_conf.update_layout(
        title="Confidence Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 105],
        height=300,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_conf, use_container_width=True)

fig_rpm = go.Figure()
fig_rpm.add_trace(go.Scatter(
    x=view_df["Timestamp"],
    y=view_df["Speed (RPM)"],
    mode="lines+markers",
    line=dict(color="#a371f7", width=2),
    marker=dict(size=6),
    name="Speed",
))
fig_rpm.update_layout(
    title="Motor Speed Over Time",
    xaxis_title="Time",
    yaxis_title="Speed (RPM)",
    height=250,
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig_rpm, use_container_width=True)


# =========================================================
# FAULT DISTRIBUTION PIE
# =========================================================
col_pie, col_table = st.columns([1, 2])

with col_pie:
    counts = view_df["Prediction"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=[CLASS_COLORS.get(c, "#388bfd") for c in counts.index],
        hole=0.4,
        textinfo="label+percent",
    ))
    fig_pie.update_layout(
        title="Fault Distribution",
        height=300,
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    st.markdown("### Recent Predictions")
    display_cols = ["Timestamp", "Prediction", "Confidence (%)", "Speed (RPM)", "RMS (g)"]
    for optional_col in ["Session ID", "Mode", "Source", "Model"]:
        if has_values(view_df, optional_col):
            display_cols.append(optional_col)
    display_df = view_df[display_cols].tail(20)
    display_df = display_df.sort_values("Timestamp", ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =========================================================
# SUMMARY STATS
# =========================================================
st.markdown("---")
st.markdown("### Session Summary")
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Visible Readings", len(view_df))
with s2:
    healthy_pct = (view_df["Prediction"] == "Healthy").mean() * 100
    st.metric("Healthy %", f"{healthy_pct:.1f}%")
with s3:
    st.metric("Avg Confidence", f"{view_df['Confidence (%)'].mean():.1f}%")
with s4:
    st.metric("Avg Speed", f"{view_df['Speed (RPM)'].mean():.0f} RPM")


# =========================================================
# AUTO REFRESH
# =========================================================
if auto_refresh:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
