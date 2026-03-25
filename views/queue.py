from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


def _to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _density_label(queue_value: int) -> str:
    if queue_value < 6:
        return "Low"
    if queue_value <= 15:
        return "Medium"
    if queue_value <= 25:
        return "High"
    return "Very High"


def _trend_label(values: pd.Series):
    if len(values) < 2:
        return "Stable", "#64748b", "-"

    first = _to_float(values.iloc[0], 0.0)
    last = _to_float(values.iloc[-1], 0.0)
    diff = last - first

    if diff > 1.0:
        return "Increasing", "#dc2626", "↑"
    if diff < -1.0:
        return "Decreasing", "#16a34a", "↓"
    return "Stable", "#64748b", "→"


def _prepare_df():
    df = st.session_state.get("df", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    work = df.copy()

    if "frame" not in work.columns:
        work["frame"] = range(1, len(work) + 1)

    if "queue_count" not in work.columns:
        work["queue_count"] = 0

    if "camera_id" not in work.columns:
        work["camera_id"] = "CAM-01"

    if "location" not in work.columns:
        work["location"] = "Unknown Location"

    if "timestamp" in work.columns:
        work["ts"] = pd.to_datetime(work["timestamp"], errors="coerce")
    else:
        fps = _to_float(st.session_state.get("live_fps", 24.0), 24.0)
        start = pd.Timestamp.now() - pd.to_timedelta((len(work) / max(fps, 1.0)), unit="s")
        work["ts"] = start + pd.to_timedelta(work["frame"] / max(fps, 1.0), unit="s")

    work["ts"] = work["ts"].ffill().fillna(pd.Timestamp.now())
    work["queue_count"] = pd.to_numeric(work["queue_count"], errors="coerce").fillna(0)
    if "queue_density" not in work.columns:
        if "queue_area" in work.columns:
            area = pd.to_numeric(work["queue_area"], errors="coerce").replace(0, pd.NA)
            work["queue_density"] = work["queue_count"] / area
        else:
            work["queue_density"] = 0.0
    else:
        work["queue_density"] = pd.to_numeric(work["queue_density"], errors="coerce").fillna(0)

    return work.sort_values("ts")


def _kpi_card(title: str, value: str, subtitle: str, icon_svg: str, bg: str, fg: str):
    st.markdown(
        f"""
        <div class="qa-kpi">
            <div class="qa-kpi-top">
                <div class="qa-kpi-icon" style="background:{bg};color:{fg};">{icon_svg}</div>
            </div>
            <div class="qa-kpi-value">{value}</div>
            <div class="qa-kpi-title">{title}</div>
            <div class="qa-kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _icon(name: str) -> str:
    icons = {
        "queue": (
            '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M4 5h16M4 12h16M4 19h16"/><circle cx="8" cy="5" r="1.5"/><circle cx="14" cy="12" r="1.5"/><circle cx="18" cy="19" r="1.5"/>'
            "</svg>"
        ),
        "clock": (
            '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<circle cx="12" cy="12" r="9"/><path d="M12 7v6l4 2"/>'
            "</svg>"
        ),
        "peak": (
            '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M3 20h18"/><path d="M6 16l4-4 3 3 5-6"/><path d="M17 9h3v3"/>'
            "</svg>"
        ),
        "alert": (
            '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="m10.29 3.86-7.82 13.54A2 2 0 0 0 4.2 20h15.6a2 2 0 0 0 1.73-2.6L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'
            "</svg>"
        ),
    }
    return icons.get(name, "")


def show():
    st.markdown(
        """
        <style>
        .qa-head { background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%); border: 1px solid #e7edf4; border-radius: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); padding: 16px 18px; margin-bottom: 14px; }
        .qa-title { margin: 0; font-size: 36px; color: #0f172a; line-height: 1.1; }
        .qa-sub { margin: 6px 0 0 0; color: #64748b; font-size: 14px; }
        .qa-card { background: #ffffff; border: 1px solid #e7edf4; border-radius: 16px; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05); padding: 14px; }
        .qa-kpi { background: #ffffff; border: 1px solid #e6edf4; border-radius: 14px; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04); padding: 12px; min-height: 132px; }
        .qa-kpi-top { display: flex; justify-content: flex-start; margin-bottom: 6px; }
        .qa-kpi-icon { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
        .qa-kpi-icon svg { width: 20px; height: 20px; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
        .qa-kpi-value { font-size: 34px; line-height: 1; font-weight: 800; color: #0f172a; margin-bottom: 6px; }
        .qa-kpi-title { font-size: 13px; color: #64748b; font-weight: 600; }
        .qa-kpi-sub { font-size: 12px; color: #94a3b8; margin-top: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="qa-head">
            <h1 class="qa-title">Queue Analytics</h1>
            <p class="qa-sub">Real-time queue length and density monitoring</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_df = _prepare_df()
    if raw_df is None:
        st.info("No queue data — upload and process a video first.")
        st.caption("Process a video to see queue analytics.")
        return

    ctrl_left, _, ctrl_right = st.columns([2, 1, 1])
    with ctrl_left:
        period = st.selectbox("Time Period", ["Last 1 Hour", "Last 24 Hours", "All Time"], index=1)

    max_ts = raw_df["ts"].max()
    if period == "Last 1 Hour":
        filtered = raw_df[raw_df["ts"] >= max_ts - timedelta(hours=1)].copy()
    elif period == "Last 24 Hours":
        filtered = raw_df[raw_df["ts"] >= max_ts - timedelta(hours=24)].copy()
    else:
        filtered = raw_df.copy()

    if filtered.empty:
        st.info("Process a video to see queue analytics.")
        return

    with ctrl_right:
        export_csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Data",
            data=export_csv,
            file_name="queue_analytics.csv",
            mime="text/csv",
            width="stretch",
        )

    fps = _to_float(st.session_state.get("live_fps", 24.0), 24.0)
    frame_sec = 1.0 / max(1.0, fps)

    avg_queue = _to_float(filtered["queue_count"].mean(), 0.0)
    avg_density = _to_float(filtered["queue_density"].mean(), 0.0)
    peak_queue = _to_int(filtered["queue_count"].max(), 0)
    latest_signal = "N/A"
    if "signal_state" in filtered.columns:
        latest_signal = str(filtered["signal_state"].iloc[-1])

    latest_by_cam = filtered.sort_values("ts").groupby("camera_id", as_index=False).tail(1)
    high_density_zones = int((latest_by_cam["queue_count"] >= 16).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Avg Queue Length", f"{avg_queue:.1f}", "Across selected period", _icon("queue"), "#dffaf3", "#0f9f96")
    with c2:
        _kpi_card("Avg Queue Density", f"{avg_density:.4f}", "Vehicles per unit area", _icon("clock"), "#fff4dd", "#ea580c")
    with c3:
        _kpi_card("Peak Queue Length", f"{peak_queue}", "Maximum observed queue", _icon("peak"), "#e8ecff", "#4f46e5")
    with c4:
        _kpi_card("High Density Zones", f"{high_density_zones}", "Queue >= 16", _icon("alert"), "#ffe4e6", "#dc2626")

    st.markdown('<div class="qa-card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown("### Current Signal State")
    st.markdown(f"<div style='font-weight:800;font-size:18px;color:#0f172a;'>{latest_signal}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="qa-card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown("### Queue Length Trend")

    trend_df = filtered[["ts", "queue_count"]].copy()
    trend_df.rename(columns={"ts": "Timestamp", "queue_count": "Queue Count"}, inplace=True)

    fig = px.bar(trend_df, x="Timestamp", y="Queue Count")
    fig.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis_title="",
        yaxis_title="",
        bargap=0.2,
    )
    fig.update_traces(marker_color="#14b8a6")
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    grouped = filtered.sort_values("ts").groupby(["camera_id", "location"], as_index=False)

    rows = []
    for (camera, location), g in grouped:
        current_queue = _to_int(g["queue_count"].iloc[-1], 0)
        avg_q = _to_float(g["queue_count"].mean(), 0.0)
        max_q = _to_int(g["queue_count"].max(), 0)
        density = _density_label(current_queue)
        trend_label, trend_color, trend_icon = _trend_label(g["queue_count"])

        rows.append(
            {
                "Location": str(location),
                "Camera": f"🎥 {camera}",
                "Current Queue": current_queue,
                "Avg Queue": round(avg_q, 1),
                "Max Queue": max_q,
                "Density": density,
                "Trend": f"{trend_icon} {trend_label}",
                "_trend_color": trend_color,
            }
        )

    status_df = pd.DataFrame(rows).sort_values(by=["Current Queue", "Max Queue"], ascending=False)

    def _style_df(df_in: pd.DataFrame):
        def density_style(val):
            if val == "Low":
                return "background-color:#dcfce7;color:#166534;font-weight:700;border-radius:10px;"
            if val == "Medium":
                return "background-color:#fef9c3;color:#854d0e;font-weight:700;border-radius:10px;"
            if val == "High":
                return "background-color:#ffedd5;color:#9a3412;font-weight:700;border-radius:10px;"
            return "background-color:#fee2e2;color:#991b1b;font-weight:700;border-radius:10px;"

        trend_colors = df_in["_trend_color"].tolist()

        def trend_style(_):
            return [f"color:{c};font-weight:700;" for c in trend_colors]

        return (
            df_in.drop(columns=["_trend_color"]) 
            .style
            .map(density_style, subset=["Density"])
            .apply(trend_style, subset=["Trend"], axis=0)
        )

    st.markdown('<div class="qa-card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown("### Queue Status by Location")
    st.dataframe(_style_df(status_df), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
