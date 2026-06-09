"""
cloud_dashboard.py
Deploy this to Streamlit Cloud.
Reads live prediction history from Google Sheets and displays it.
"""

import time
from html import escape

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

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1480px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1 {
            font-size: 2.35rem !important;
            letter-spacing: 0 !important;
            margin-bottom: 0.25rem !important;
        }
        h3 {
            margin-top: 0.35rem !important;
            margin-bottom: 0.65rem !important;
        }
        .status-card {
            padding: 16px 18px;
            border-radius: 8px;
            background: #161b22;
            border: 1px solid #30363d;
            min-height: 104px;
        }
        .status-card .label {
            margin: 0;
            color: #8b949e;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .status-card .value {
            margin: 8px 0 0;
            color: #e6edf3;
            font-size: 28px;
            font-weight: 800;
            line-height: 1.1;
            overflow-wrap: anywhere;
        }
        .status-card .detail {
            margin: 8px 0 0;
            color: #8b949e;
            font-size: 13px;
        }
        .info-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 18px;
        }
        .info-item {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 14px;
            min-width: 0;
        }
        .info-item .label {
            margin: 0;
            color: #8b949e;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .info-item .value {
            margin: 6px 0 0;
            color: #e6edf3;
            font-size: 18px;
            font-weight: 750;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .section-note {
            color: #8b949e;
            font-size: 14px;
            margin-top: -0.2rem;
        }
        hr {
            margin: 1.15rem 0 !important;
        }
        @media (max-width: 900px) {
            .block-container {
                padding: 1rem 0.85rem 1.5rem;
            }
            h1 {
                font-size: 1.75rem !important;
                line-height: 1.15 !important;
            }
            .info-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }
            .status-card {
                min-height: auto;
                padding: 14px;
            }
            .status-card .value {
                font-size: 24px;
            }
        }
        @media (max-width: 560px) {
            .info-strip {
                grid-template-columns: 1fr;
            }
            h1 {
                font-size: 1.45rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
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


def compact_text(value, head=18, tail=5):
    if pd.isna(value):
        return "Legacy"
    text = str(value or "").strip()
    if not text:
        return "Legacy"
    limit = head + tail + 3
    if len(text) <= limit:
        return text
    return f"{text[:head]}...{text[-tail:]}"


def compact_source(value):
    if pd.isna(value):
        return "Live log"
    text = str(value or "").strip()
    if not text:
        return "Live log"
    normalized = text.replace("\\", "/").rstrip("/")
    return compact_text(normalized.split("/")[-1] or normalized, 22, 5)


def info_item(label, value):
    return (
        "<div class='info-item'>"
        f"<p class='label'>{escape(str(label))}</p>"
        f"<p class='value'>{escape(str(value))}</p>"
        "</div>"
    )


@st.cache_data(ttl=5)
def fetch_data():
    """Pull all rows from the sheet and return as DataFrame."""
    try:
        ws = get_worksheet()

        # Avoid gspread get_all_records(), which can fail on Streamlit Cloud with:
        # APIError [400]: Unable to parse range: 'Sheet1'.
        # Use whole columns so the monitor does not miss new rows after 10,000.
        values = ws.get("A:Q")
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

col_r1, col_r2, col_r3 = st.columns([1.05, 1.05, 5])
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

st.markdown("### Cloud Controls")
st.markdown(
    "<p class='section-note'>The default view follows the latest live session. "
    "Switch to all sessions when reviewing the full history.</p>",
    unsafe_allow_html=True,
)
f1, f2, f3 = st.columns([2.2, 2.0, 1.1])
with f1:
    if session_values:
        session_options = ["All sessions", "Latest session"] + session_values
        session_choice = st.selectbox("Session", session_options, index=1)
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
    download_slot = st.empty()

view_df = df.copy()
if session_choice == "Latest session" and latest_session:
    view_df = view_df[view_df["Session ID"].astype(str) == latest_session]
elif session_choice not in ("All sessions", "Latest session"):
    view_df = view_df[view_df["Session ID"].astype(str) == session_choice]
if pred_filter:
    view_df = view_df[view_df["Prediction"].isin(pred_filter)]

if view_df.empty:
    st.warning("No cloud predictions match the selected filters.")
    if auto_refresh:
        time.sleep(5)
        st.cache_data.clear()
        st.rerun()
    st.stop()

with f3:
    download_slot.download_button(
        "Download CSV",
        view_df.to_csv(index=False).encode("utf-8"),
        file_name="cloud_predictions_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

session_count = view_df["Session ID"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()
status_html = (
    "<div class='info-strip'>"
    + info_item("Visible rows", len(view_df))
    + info_item("Sessions", session_count)
    + info_item("Latest session", compact_text(latest_session, 20, 5))
    + info_item("Last source", compact_source(view_df.iloc[-1]["Source"]))
    + "</div>"
)
st.markdown(status_html, unsafe_allow_html=True)


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
        f"<div class='status-card' style='text-align:center;"
        f"border:2px solid {color}'>"
        f"<p class='label'>Prediction</p>"
        f"<p class='value' style='color:{color}'>{pred}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='status-card' style='text-align:center'>"
        f"<p class='label'>Confidence</p>"
        f"<p class='value'>{conf:.1f}%</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='status-card' style='text-align:center'>"
        f"<p class='label'>Speed</p>"
        f"<p class='value'>{rpm:.0f} RPM</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"<div class='status-card' style='text-align:center'>"
        f"<p class='label'>Last Updated</p>"
        f"<p class='detail' style='font-size:14px;color:#e6edf3;overflow-wrap:anywhere'>{escape(ts)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

if str(latest.get("Session ID", "")).strip():
    st.caption(
        f"Session: {latest['Session ID']} | "
        f"Source: {compact_source(latest.get('Source', ''))}"
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    st.markdown("### Recent Predictions")
    display_cols = ["Timestamp", "Prediction", "Confidence (%)", "Speed (RPM)", "RMS (g)"]
    for optional_col in ["Session ID", "Source", "Model"]:
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
