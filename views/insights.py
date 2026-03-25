import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from backend import analytics


def _safe_df():
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _avg_speed_change(df, time_range):
    if df is None:
        return 0.0
    ts = None
    for col in ["timestamp", "time", "datetime", "date"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if not ts.isna().all():
                break
    if ts is None or ts.isna().all():
        return 0.0

    if time_range == "Today":
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + pd.Timedelta(days=1)
        prev_start = start - pd.Timedelta(days=1)
        prev_end = start
    elif time_range == "Last 7 Days":
        end = datetime.now()
        start = end - pd.Timedelta(days=7)
        prev_end = start
        prev_start = start - pd.Timedelta(days=7)
    else:
        end = datetime.now()
        start = end - pd.Timedelta(days=30)
        prev_end = start
        prev_start = start - pd.Timedelta(days=30)

    curr = df[(ts >= start) & (ts < end)]
    prev = df[(ts >= prev_start) & (ts < prev_end)]

    def _avg_speed(d):
        for col in ["speed", "avg_speed", "average_speed"]:
            if col in d.columns:
                series = pd.to_numeric(d[col], errors="coerce").dropna()
                if not series.empty:
                    return float(series.mean())
        return None

    curr_avg = _avg_speed(curr) or 0.0
    prev_avg = _avg_speed(prev) or 0.0
    if prev_avg == 0:
        return 0.0 if curr_avg == 0 else 100.0
    return (curr_avg - prev_avg) / prev_avg * 100.0


def show():
    df = _safe_df()

    st.markdown(
        """
        <style>
        .ins-root { padding: 8px 6px 18px 6px; }

        .panel {
            background: #ffffff;
            border: 1px solid #e7edf4;
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            padding: 14px;
        }

        .title {
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }

        .subtitle {
            color: #64748b;
            font-size: 12px;
            margin-top: 4px;
        }

        .kpi {
            background: #ffffff;
            border: 1px solid #eef2f7;
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
        }

        .kpi-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .kpi-icon {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .kpi-badge {
            font-size: 10px;
            font-weight: 700;
            border-radius: 999px;
            padding: 4px 8px;
        }

        .badge-up { background: #dcfce7; color: #166534; }
        .badge-down { background: #fee2e2; color: #991b1b; }

        .kpi-label {
            color: #64748b;
            font-size: 12px;
            margin-bottom: 4px;
        }

        .kpi-value {
            color: #0f172a;
            font-size: 22px;
            font-weight: 800;
            line-height: 1;
        }

        .kpi-sub {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 6px;
        }

        .section-title {
            font-size: 14px;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 10px 0;
        }

        .info-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            border-radius: 8px;
            padding: 12px;
            font-size: 13px;
            text-align: center;
        }

        .insight-card {
            border: 1px solid #eef2f7;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
        }

        .insight-title {
            font-size: 12px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 6px;
        }

        .insight-text {
            color: #475569;
            font-size: 11px;
            margin-bottom: 8px;
        }

        .insight-rec {
            background: #ffffff;
            border: 1px solid #e7edf4;
            border-radius: 8px;
            padding: 6px 8px;
            font-size: 10px;
            color: #334155;
        }

        .sev-badge {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 999px;
        }

        .sev-high { background: #fee2e2; color: #991b1b; }
        .sev-medium { background: #ffedd5; color: #9a3412; }

        .hour-row {
            margin-bottom: 12px;
        }

        .hour-top {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #475569;
            margin-bottom: 6px;
        }

        .bar {
            height: 8px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2dd4bf 0%, #14b8a6 100%);
        }

        .pred-card {
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            border: 1px solid #eef2f7;
        }

        .pred-title {
            font-size: 12px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .pred-sub {
            font-size: 11px;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='ins-root'>", unsafe_allow_html=True)

    header_left, header_right = st.columns([3, 1.6])
    with header_left:
        st.markdown("<div class='title'>Trends & Insights</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>AI-powered analytics and predictive insights</div>", unsafe_allow_html=True)
    with header_right:
        h1, h2 = st.columns([2, 1])
        time_range = h1.selectbox("Range", ["Last 7 Days", "Today", "Last 30 Days"], index=0, label_visibility="collapsed")

        export_payload = io.BytesIO()
        export_data = {
            "time_range": time_range,
            "trend_metrics": analytics.get_trend_metrics(time_range),
            "weekly_data": analytics.get_weekly_data(),
            "peak_hours": analytics.get_peak_hours(),
            "predictive": analytics.get_predictive_insights(),
        }
        export_df = pd.json_normalize(export_data)
        export_df.to_csv(export_payload, index=False)
        export_payload.seek(0)
        h2.download_button(
            "Export Insights",
            data=export_payload,
            file_name="insights_export.csv",
            mime="text/csv",
            width="stretch",
        )

    metrics = analytics.get_trend_metrics(time_range)
    avg_speed_change = _avg_speed_change(df, time_range)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Traffic Growth", "\u2197", "#d1fae5", "#10b981", f"{metrics['traffic_growth']:+.0f}%", "vs previous period", metrics["traffic_growth"]),
        ("Violation Rate", "\u2198", "#fee2e2", "#ef4444", f"{metrics['violation_rate_change']:+.0f}%", "vs previous period", metrics["violation_rate_change"]),
        ("Avg Speed", "\u23F1", "#e0e7ff", "#6366f1", f"{metrics['avg_speed']:.0f} km/h" if metrics["avg_speed"] else "N/A", "across all zones", avg_speed_change),
        ("Violation Rate", "\u26A0", "#fff7ed", "#f97316", f"{metrics['violation_percent']:.1f}%" if metrics["violation_percent"] else "N/A", "of total traffic", metrics["violation_rate_change"]),
    ]

    for col, (label, icon, bg, color, value, sub, change) in zip([k1, k2, k3, k4], kpis):
        badge_cls = "badge-up" if change >= 0 else "badge-down"
        with col:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-top">
                        <div class="kpi-icon" style="background:{bg};color:{color};">{icon}</div>
                        <span class="kpi-badge {badge_cls}">{change:+.0f}%</span>
                    </div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    weekly = analytics.get_weekly_data()
    weekly_left, weekly_right = st.columns(2)

    with weekly_left:
        st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Weekly Violation Trend</div>", unsafe_allow_html=True)
        viol_map = weekly.get("daily_violations", {})
        if not viol_map or sum(viol_map.values()) == 0:
            st.markdown("<div class='info-box'>No weekly violations data available.</div>", unsafe_allow_html=True)
        else:
            vdf = pd.DataFrame({"Day": list(viol_map.keys()), "Count": list(viol_map.values())})
            fig = px.bar(vdf, x="Day", y="Count", color_discrete_sequence=["#ef4444"])
            fig.update_layout(height=260, margin=dict(l=8, r=8, t=6, b=6), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with weekly_right:
        st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Weekly Traffic Volume</div>", unsafe_allow_html=True)
        traf_map = weekly.get("daily_traffic", {})
        if not traf_map or sum(traf_map.values()) == 0:
            st.markdown("<div class='info-box'>No weekly traffic data available.</div>", unsafe_allow_html=True)
        else:
            tdf = pd.DataFrame({"Day": list(traf_map.keys()), "Count": list(traf_map.values())})
            fig = px.bar(tdf, x="Day", y="Count", color_discrete_sequence=["#14b8a6"])
            fig.update_layout(height=260, margin=dict(l=8, r=8, t=6, b=6), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Key Insights & Recommendations</div>", unsafe_allow_html=True)

    peak_hours = analytics.get_peak_hours()
    hotspot = "N/A"
    hotspot_count = 0
    queue_trend = "stable"
    if df is not None and "location" in df.columns:
        loc_series = df["location"].astype(str).str.strip()
        loc_series = loc_series[loc_series != ""]
        if not loc_series.empty:
            hotspot = loc_series.value_counts().idxmax()
            hotspot_count = int(loc_series.value_counts().max())

    weekend_drop = None
    if df is not None:
        ts = None
        for col in ["timestamp", "time", "datetime", "date"]:
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce")
                if not ts.isna().all():
                    break
        if ts is not None and not ts.isna().all():
            weekdays = df[ts.dt.dayofweek < 5]
            weekends = df[ts.dt.dayofweek >= 5]
            weekday_traffic = float(analytics._frame_traffic(weekdays).sum()) if not weekdays.empty else 0.0
            weekend_traffic = float(analytics._frame_traffic(weekends).sum()) if not weekends.empty else 0.0
            if weekday_traffic > 0:
                weekend_drop = (1 - (weekend_traffic / weekday_traffic)) * 100.0

    if df is not None and "queue_count" in df.columns:
        q_series = pd.to_numeric(df["queue_count"], errors="coerce").dropna()
        if len(q_series) > 4:
            recent = q_series.tail(len(q_series) // 3).mean()
            prev = q_series.head(len(q_series) // 3).mean()
            if prev > 0:
                change = ((recent - prev) / prev) * 100.0
                queue_trend = "increasing" if change > 5 else "decreasing" if change < -5 else "stable"

    i1, i2 = st.columns(2)
    with i1:
        peak_text = "No peak hour data available."
        rec_text = "Process additional video to identify peak periods."
        if peak_hours:
            top = peak_hours[0]
            peak_text = f"Highest traffic volume observed between {top['time_range']}."
            rec_text = "Recommendation: Consider deploying additional monitoring during these hours."
        st.markdown(
            f"""
            <div class="insight-card" style="background:#fef2f2;">
                <div class="insight-title">Peak Traffic Hours <span class="sev-badge sev-high">HIGH</span></div>
                <div class="insight-text">{peak_text}</div>
                <div class="insight-rec">{rec_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with i2:
        hot_text = "No hotspot data available."
        hot_rec = "Recommendation: Add more cameras to identify hotspots."
        if hotspot != "N/A":
            hot_text = f"{hotspot} shows highest traffic concentration with {hotspot_count} events."
            hot_rec = "Recommendation: Increase enforcement and install warning signage."
        st.markdown(
            f"""
            <div class="insight-card" style="background:#fff7ed;">
                <div class="insight-title">Violation Hotspot <span class="sev-badge sev-high">HIGH</span></div>
                <div class="insight-text">{hot_text}</div>
                <div class="insight-rec">{hot_rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    i3, i4 = st.columns(2)
    with i3:
        weekend_text = "Weekend pattern data not available."
        weekend_rec = "Recommendation: Capture weekend sessions for comparison."
        if weekend_drop is not None:
            weekend_text = f"Traffic volume changes by {weekend_drop:.0f}% on weekends compared to weekdays."
            weekend_rec = "Recommendation: Optimize resource allocation for weekend shifts."
        st.markdown(
            f"""
            <div class="insight-card" style="background:#ecfeff;">
                <div class="insight-title">Weekend Traffic Pattern <span class="sev-badge sev-medium">MEDIUM</span></div>
                <div class="insight-text">{weekend_text}</div>
                <div class="insight-rec">{weekend_rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with i4:
        queue_text = f"Queue length trend is {queue_trend}."
        queue_rec = "Recommendation: Review signal timing optimization opportunities."
        st.markdown(
            f"""
            <div class="insight-card" style="background:#eef2ff;">
                <div class="insight-title">Queue Length Trend <span class="sev-badge sev-medium">MEDIUM</span></div>
                <div class="insight-text">{queue_text}</div>
                <div class="insight-rec">{queue_rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    peak_col, pred_col = st.columns([2, 1])
    with peak_col:
        st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Peak Traffic Hours</div>", unsafe_allow_html=True)
        if not peak_hours:
            st.markdown("<div class='info-box'>No peak hour analytics available.</div>", unsafe_allow_html=True)
        else:
            top_hours = peak_hours[:5]
            max_traffic = max([h["traffic"] for h in top_hours]) if top_hours else 0
            for item in top_hours:
                percent = (item["traffic"] / max_traffic * 100) if max_traffic else 0
                st.markdown(
                    f"""
                    <div class="hour-row">
                        <div class="hour-top">
                            <span>{item['time_range']}</span>
                            <span>{item['violations']} violations</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill" style="width:{percent}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with pred_col:
        st.markdown("<div class='panel' style='margin-top:14px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Predictive Analytics</div>", unsafe_allow_html=True)
        preds = analytics.get_predictive_insights()
        if preds["tomorrow_forecast"] == 0 and preds["violation_prediction"] == [0, 0]:
            st.markdown("<div class='info-box'>No predictive insights available yet.</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="pred-card" style="background:#ecfeff;">
                    <div class="pred-title">Tomorrow's Forecast</div>
                    <div class="pred-sub">Expected traffic volume: {preds['tomorrow_forecast']} vehicles</div>
                </div>
                <div class="pred-card" style="background:#fff7ed;">
                    <div class="pred-title">Violation Prediction</div>
                    <div class="pred-sub">Estimated violations: {preds['violation_prediction'][0]}-{preds['violation_prediction'][1]}</div>
                </div>
                <div class="pred-card" style="background:#eef2ff;">
                    <div class="pred-title">Queue Length Forecast</div>
                    <div class="pred-sub">Peak queue: {preds['queue_forecast'][0]}-{preds['queue_forecast'][1]} vehicles</div>
                </div>
                <div class="pred-card" style="background:#fee2e2;">
                    <div class="pred-title">Congestion Alert</div>
                    <div class="pred-sub">{'High risk detected' if preds['congestion_alert'] else 'Normal traffic expected'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
