from datetime import datetime, timedelta
import os

import pandas as pd
import streamlit as st


def _safe_df():
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()
    if os.path.exists("traffic_log.csv"):
        try:
            log_df = pd.read_csv("traffic_log.csv")
            if not log_df.empty:
                return log_df
        except Exception:
            return None
    return None


def _find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _parse_timestamp(df, column):
    if column is None:
        return None
    ts = pd.to_datetime(df[column], errors="coerce")
    if ts.isna().all():
        return None
    return ts


def _period_bounds(period):
    now = datetime.now()
    if period == "Today":
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
        prev_start = start - timedelta(days=1)
        prev_end = start
        return start, end, prev_start, prev_end
    if period == "Week":
        start = now - timedelta(days=7)
        end = now
        prev_start = start - timedelta(days=7)
        prev_end = start
        return start, end, prev_start, prev_end
    return None, None, None, None


def _class_counts(df, class_col, id_col):
    if df is None or df.empty or class_col is None:
        return {"car": 0, "bike": 0, "bus": 0, "truck": 0, "auto": 0}, 0
    work = df.copy()
    work[class_col] = work[class_col].astype(str).str.lower().str.strip()
    if id_col and id_col in work.columns:
        total = int(work[id_col].nunique())
    else:
        total = int(len(work))

    def _count(cls):
        sub = work[work[class_col].isin(cls)]
        if id_col and id_col in sub.columns:
            return int(sub[id_col].nunique())
        return int(len(sub))

    return {
        "car": _count(["car", "sedan", "suv", "hatchback"]),
        "bike": _count(["bike", "bicycle", "motorbike", "motorcycle"]),
        "bus": _count(["bus"]),
        "truck": _count(["truck", "lorry"]),
        "auto": _count(["auto", "auto-rickshaw", "autorickshaw", "rickshaw", "three wheeler", "three-wheeler", "tuk tuk"]),
    }, total


def _percent_change(curr, prev):
    if prev == 0:
        return 0.0 if curr == 0 else 100.0
    return (curr - prev) / prev * 100.0


def _hourly_counts(ts):
    if ts is None:
        return None
    hours = ts.dropna().dt.hour
    if hours.empty:
        return None
    buckets = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24)]
    labels = [f"{start:02d}:00" for start, _ in buckets]
    counts = []
    for start, end in buckets:
        counts.append(int(((hours >= start) & (hours < end)).sum()))
    return pd.DataFrame({"Hour": labels, "Count": counts})


def _peak_hour(hour_df):
    if hour_df is None or hour_df.empty:
        return "N/A", 0
    idx = hour_df["Count"].idxmax()
    return hour_df.loc[idx, "Hour"], int(hour_df.loc[idx, "Count"])


def _avg_speed(df):
    if df is None or df.empty:
        return None
    for col in ["speed", "avg_speed", "average_speed"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.mean())
    metrics = st.session_state.get("metrics", {}) or {}
    for key in ["avg_speed", "average_speed"]:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return None


def _busiest_zone(df):
    if df is None or df.empty:
        return "N/A", 0
    for col in ["location", "zone", "region", "camera_id"]:
        if col in df.columns:
            series = df[col].astype(str).str.strip()
            series = series[series != ""]
            if not series.empty:
                top = series.value_counts().idxmax()
                count = int(series.value_counts().max())
                return top, count
    return "N/A", 0


def show():
    df = _safe_df()

    st.markdown(
        """
        <style>
        .vs-root { padding: 8px 6px 18px 6px; }

        .panel {
            background: #ffffff;
            border: 1px solid #e7edf4;
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            padding: 14px;
        }

        .header-title {
            margin: 0;
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
        }

        .header-sub {
            margin: 4px 0 0 0;
            color: #64748b;
            font-size: 12px;
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
            margin: 0 0 12px 0;
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

        .bar-row {
            margin-bottom: 12px;
        }

        .bar-top {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #475569;
            margin-bottom: 6px;
        }

        .bar {
            height: 10px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2dd4bf 0%, #14b8a6 100%);
        }

        .qs-card {
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            border: 1px solid #eef2f7;
        }

        .qs-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #475569;
        }

        .qs-value {
            margin-top: 6px;
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
        }

        .qs-sub {
            margin-top: 2px;
            font-size: 11px;
            color: #94a3b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='vs-root'>", unsafe_allow_html=True)

    head_left, head_right = st.columns([3, 2])
    with head_left:
        st.markdown("<h2 class='header-title'>Vehicle Statistics</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='header-sub'>Comprehensive vehicle detection and classification analytics</p>",
            unsafe_allow_html=True,
        )
    with head_right:
        h1, h2 = st.columns([2, 1])
        period = h1.selectbox("Period", ["Today", "Week", "All Time"], index=0, label_visibility="collapsed")
        if df is None or df.empty:
            h2.button("Export Data", width="stretch")
        else:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            h2.download_button("Export Data", data=csv_bytes, file_name="vehicle_statistics.csv", mime="text/csv", width="stretch")

    if df is None or df.empty:
        st.markdown("<div class='info-box'>No vehicle statistics available. Process a video first.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    class_col = _find_column(df, ["class", "vehicle_class", "vehicle_type", "label"])
    id_col = _find_column(df, ["vehicle_id", "track_id", "id"])
    ts_col = _find_column(df, ["timestamp", "time", "datetime", "date"])
    ts = _parse_timestamp(df, ts_col)

    start, end, prev_start, prev_end = _period_bounds(period)
    if ts is not None and start is not None:
        curr_mask = (ts >= start) & (ts < end)
        prev_mask = (ts >= prev_start) & (ts < prev_end)
        curr_df = df[curr_mask].copy()
        prev_df = df[prev_mask].copy()
    else:
        curr_df = df.copy()
        prev_df = df.iloc[0:0].copy()

    curr_counts, curr_total = _class_counts(curr_df, class_col, id_col)
    prev_counts, _ = _class_counts(prev_df, class_col, id_col)

    kpi_defs = [
        ("Cars", "🚗", "#dcfce7", "#10b981", curr_counts["car"], prev_counts["car"]),
        ("Bikes", "🏍️", "#ffedd5", "#f97316", curr_counts["bike"], prev_counts["bike"]),
        ("Buses", "🚌", "#e0e7ff", "#6366f1", curr_counts["bus"], prev_counts["bus"]),
        ("Trucks", "🚚", "#fee2e2", "#ef4444", curr_counts["truck"], prev_counts["truck"]),
        ("Autos", "🛺", "#fdf2f8", "#db2777", curr_counts["auto"], prev_counts["auto"]),
    ]

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, icon, bg, color, curr, prev) in zip([c1, c2, c3, c4, c5], kpi_defs):
        change = _percent_change(curr, prev)
        badge_cls = "badge-up" if change >= 0 else "badge-down"
        share = 0.0 if curr_total == 0 else (curr / curr_total) * 100
        with col:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-top">
                        <div class="kpi-icon" style="background:{bg};color:{color};">{icon}</div>
                        <span class="kpi-badge {badge_cls}">{change:+.0f}%</span>
                    </div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{curr}</div>
                    <div class="kpi-sub">{share:.0f}% of total traffic</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    chart_col, stats_col = st.columns([2.2, 1])
    with chart_col:
        st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Vehicle Flow by Hour</div>", unsafe_allow_html=True)

        hour_df = _hourly_counts(_parse_timestamp(curr_df, ts_col) if ts_col else None)
        if hour_df is None:
            st.markdown(
                "<div class='info-box'>Process a video to generate vehicle statistics.</div>",
                unsafe_allow_html=True,
            )
        else:
            max_val = int(hour_df["Count"].max()) if not hour_df.empty else 0
            for _, row in hour_df.iterrows():
                percent = (row["Count"] / max_val * 100) if max_val else 0
                st.markdown(
                    f"""
                    <div class="bar-row">
                        <div class="bar-top">
                            <span>{row['Hour']}</span>
                            <span>{int(row['Count'])}</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill" style="width:{percent}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with stats_col:
        st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Quick Stats</div>", unsafe_allow_html=True)

        total_display = curr_total
        peak_hour, peak_count = _peak_hour(_hourly_counts(_parse_timestamp(curr_df, ts_col) if ts_col else None))
        avg_speed = _avg_speed(curr_df)
        zone_name, zone_count = _busiest_zone(curr_df)

        st.markdown(
            f"""
            <div class="qs-card" style="background:#ecfeff;">
                <div class="qs-title"><span>Total Vehicles</span><span>🚗</span></div>
                <div class="qs-value">{total_display}</div>
                <div class="qs-sub">Detected {period.lower()}</div>
            </div>
            <div class="qs-card" style="background:#fff7ed;">
                <div class="qs-title"><span>Peak Hour</span><span>🕒</span></div>
                <div class="qs-value">{peak_hour}</div>
                <div class="qs-sub">{peak_count} vehicles/hour</div>
            </div>
            <div class="qs-card" style="background:#eef2ff;">
                <div class="qs-title"><span>Avg Speed</span><span>🏁</span></div>
                <div class="qs-value">{f"{avg_speed:.0f} km/h" if avg_speed is not None else "N/A"}</div>
                <div class="qs-sub">Across all zones</div>
            </div>
            <div class="qs-card" style="background:#ffe4e6;">
                <div class="qs-title"><span>Busiest Zone</span><span>📍</span></div>
                <div class="qs-value">{zone_name}</div>
                <div class="qs-sub">{zone_count} vehicles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
