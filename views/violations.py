from datetime import datetime

import pandas as pd
import streamlit as st


def _lucide_svg(icon_name: str) -> str:
    icons = {
        "alert": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="m10.29 3.86-7.82 13.54A2 2 0 0 0 4.2 20h15.6a2 2 0 0 0 1.73-2.6L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
            '<path d="M12 9v4"/><path d="M12 17h.01"/>'
            "</svg>"
        ),
        "light": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<rect x="7" y="2.5" width="10" height="15" rx="5"/>'
            '<circle cx="12" cy="7" r="1.8"/>'
            '<circle cx="12" cy="12" r="1.8"/>'
            '<path d="M12 17.8v3"/>'
            "</svg>"
        ),
        "speed": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M20 13a8 8 0 1 0-16 0"/>'
            '<path d="m12 13 3.2-3.2"/>'
            '<path d="M5 13h.01"/><path d="M19 13h.01"/>'
            "</svg>"
        ),
        "layers": (
            '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="m12 2 9 5-9 5-9-5z"/>'
            '<path d="m3 12 9 5 9-5"/>'
            '<path d="m3 17 9 5 9-5"/>'
            "</svg>"
        ),
    }
    return icons.get(icon_name, "")


def _to_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_violations():
    data = st.session_state.get("violations", [])
    if isinstance(data, list):
        return data
    return []


def _build_dataframe(violations):
    if not violations:
        return pd.DataFrame()
    df = pd.DataFrame(violations)
    if df.empty:
        return df
    for col in ["id", "type", "location", "vehicle", "severity", "status", "time"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "speed" in df.columns:
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    return df


def _counts_by_type(df: pd.DataFrame):
    if df is None or df.empty or "type" not in df.columns:
        return {
            "Red Light Jump": 0,
            "Rash Driving": 0,
            "No Helmet": 0,
            "Mobile Usage While Driving": 0,
            "Triple Riding": 0,
            "Heavy Load": 0,
            "Wrong Lane": 0,
            "Other": 0,
        }
    series = df["type"].fillna("").astype(str).str.strip()
    red = int((series == "Red Light Jump").sum())
    rash = int((series == "Rash Driving").sum())
    helmet = int((series == "No Helmet").sum())
    mobile = int((series == "Mobile Usage While Driving").sum())
    triple = int((series == "Triple Riding").sum())
    heavy = int((series == "Heavy Load").sum())
    wrong = int((series == "Wrong Lane").sum())
    known = {
        "Red Light Jump",
        "Rash Driving",
        "No Helmet",
        "Mobile Usage While Driving",
        "Triple Riding",
        "Heavy Load",
        "Wrong Lane",
    }
    other = int((~series.isin(known) & (series != "")).sum())
    return {
        "Red Light Jump": red,
        "Rash Driving": rash,
        "No Helmet": helmet,
        "Mobile Usage While Driving": mobile,
        "Triple Riding": triple,
        "Heavy Load": heavy,
        "Wrong Lane": wrong,
        "Other": other,
    }


def _parse_hour(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.hour
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).hour
        except Exception:
            continue
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.hour)


def _split_vehicle(text):
    if text is None:
        return "", ""
    raw = str(text).strip()
    if not raw:
        return "", ""
    if "(" in raw and raw.endswith(")"):
        left, right = raw.rsplit("(", 1)
        return left.strip(), right[:-1].strip()
    return raw, ""


def _extract_date(row):
    for key in ["date", "day", "timestamp", "datetime"]:
        if key in row and row.get(key):
            value = row.get(key)
            try:
                parsed = pd.to_datetime(value, errors="coerce")
                if pd.isna(parsed):
                    return str(value)
                return parsed.strftime("%Y-%m-%d")
            except Exception:
                return str(value)
    return ""


def _bucket_label(hour):
    buckets = [
        (6, 9, "06:00 - 09:00"),
        (9, 12, "09:00 - 12:00"),
        (12, 15, "12:00 - 15:00"),
        (15, 18, "15:00 - 18:00"),
        (18, 21, "18:00 - 21:00"),
    ]
    for start, end, label in buckets:
        if start <= hour < end:
            return label
    return None


def show():
    violations = _safe_violations()
    df = _build_dataframe(violations)

    counts = _counts_by_type(df)
    total_violations = int(sum(counts.values())) if not df.empty else 0
    red_count = counts.get("Red Light Jump", 0)
    rash_count = counts.get("Rash Driving", 0)
    other_count = counts.get("Other", 0) + counts.get("Wrong Lane", 0)

    st.markdown(
        """
        <style>
        .vio-root { padding: 8px 6px 18px 6px; }

        .panel {
            background: #ffffff;
            border: 1px solid #e7edf4;
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            padding: 14px;
        }

        .panel-title {
            font-size: 20px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }

        .panel-sub {
            color: #64748b;
            font-size: 12px;
            margin: 4px 0 0 0;
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
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
        }

        .kpi-icon {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0f172a;
        }

        .kpi-icon svg {
            width: 18px;
            height: 18px;
            stroke: currentColor;
            stroke-width: 2.2;
            stroke-linecap: round;
            stroke-linejoin: round;
            fill: none;
        }

        .kpi-label {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 4px;
        }

        .kpi-value {
            color: #0f172a;
            font-size: 24px;
            font-weight: 800;
            line-height: 1;
        }

        .info-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 13px;
        }

        .table-wrap {
            border: 1px solid #e7edf4;
            border-radius: 12px;
            overflow: hidden;
        }

        .vio-table {
            width: 100%;
            border-collapse: collapse;
        }

        .vio-table th {
            text-align: left;
            padding: 9px 12px;
            background: #f8fafc;
            border-bottom: 1px solid #e7edf4;
            color: #475569;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .vio-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #eef2f7;
            color: #1f2937;
            font-size: 13px;
            vertical-align: top;
        }

        .cell-main {
            font-weight: 700;
            color: #0f172a;
            font-size: 12px;
        }

        .cell-sub {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 2px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 10px;
            font-weight: 700;
        }

        .sev-high { background: #fee2e2; color: #991b1b; }
        .sev-medium { background: #ffedd5; color: #9a3412; }
        .sev-low { background: #fef9c3; color: #854d0e; }

        .st-verified { background: #dcfce7; color: #166534; }
        .st-review { background: #dbeafe; color: #1d4ed8; }
        .st-pending { background: #e5e7eb; color: #4b5563; }

        .section-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .section-head h3 {
            margin: 0;
            color: #0f172a;
            font-size: 16px;
            font-weight: 800;
        }

        .select-wrap .stSelectbox > div {
            min-width: 140px;
        }

        .progress-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
            font-size: 12px;
            color: #475569;
        }

        .progress-bar {
            flex: 1;
            height: 8px;
            border-radius: 999px;
            background: #e5e7eb;
            overflow: hidden;
            margin-left: 8px;
        }

        .progress-fill {
            height: 100%;
            border-radius: 999px;
        }

        .progress-right {
            min-width: 36px;
            text-align: right;
            color: #0f172a;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='vio-root'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel" style="margin-bottom:14px;">
            <p class="panel-title">Violation Detection</p>
            <p class="panel-sub">Automated violation analytics across monitored intersections</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (_lucide_svg("alert"), "#ffe4e6", "#dc2626", "Total Violations", total_violations),
        (_lucide_svg("light"), "#fff4dd", "#ea580c", "Red Light Jumps", red_count),
        (_lucide_svg("speed"), "#fff2cc", "#f59e0b", "Rash Driving", rash_count),
        (_lucide_svg("layers"), "#e0e7ff", "#6366f1", "Other Violations", other_count),
    ]

    for col, (icon_svg, bg, icon_color, label, value) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-top">
                        <div class="kpi-icon" style="background:{bg};color:{icon_color};">{icon_svg}</div>
                    </div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown("<div class='section-head'><h3>Violation Records</h3></div>", unsafe_allow_html=True)
    with head_right:
        filter_options = [
            "All Types",
            "Red Light Jump",
            "Rash Driving",
            "No Helmet",
            "Mobile Usage While Driving",
            "Triple Riding",
            "Heavy Load",
            "Wrong Lane",
            "Other",
        ]
        st.markdown("<div class='select-wrap'>", unsafe_allow_html=True)
        selected_filter = st.selectbox("All Types", filter_options, index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    if df.empty:
        st.markdown(
            "<div class='info-box'>No violations detected yet. Process a video to generate violation analytics.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    filtered_df = df.copy()
    if selected_filter != "All Types" and "type" in filtered_df.columns:
        if selected_filter == "Other":
            filtered_df = filtered_df[
                ~filtered_df["type"].isin(
                    [
                        "Red Light Jump",
                        "Rash Driving",
                        "No Helmet",
                        "Mobile Usage While Driving",
                        "Triple Riding",
                        "Heavy Load",
                        "Wrong Lane",
                    ]
                )
            ]
        else:
            filtered_df = filtered_df[filtered_df["type"] == selected_filter]

    def _badge(cls, text):
        return f"<span class='badge {cls}'>{text}</span>"

    rows_html = []
    for _, row in filtered_df.iterrows():
        sev = str(row.get("severity", "")).strip().title()
        status = str(row.get("status", "")).strip().title()
        sev_cls = "sev-low"
        if sev == "High":
            sev_cls = "sev-high"
        elif sev == "Medium":
            sev_cls = "sev-medium"

        status_cls = "st-pending"
        if status == "Verified":
            status_cls = "st-verified"
        elif status == "Under Review":
            status_cls = "st-review"

        speed_val = row.get("speed", "")
        speed_display = ""
        if pd.notna(speed_val) and speed_val != "":
            speed_display = f"{_to_int(speed_val, speed_val)} km/h"

        vid = str(row.get("id", "")).strip()
        cam_id = str(row.get("camera_id", "")).strip() or str(row.get("camera", "")).strip()
        vehicle_main, vehicle_sub = _split_vehicle(row.get("vehicle", ""))
        date_text = _extract_date(row)

        rows_html.append(
            "<tr>"
            f"<td><div class='cell-main'>{vid}</div><div class='cell-sub'>{cam_id}</div></td>"
            f"<td>{row.get('type', '')}</td>"
            f"<td>{row.get('location', '')}</td>"
            f"<td><div class='cell-main'>{vehicle_main}</div><div class='cell-sub'>{vehicle_sub}</div></td>"
            f"<td>{speed_display}</td>"
            f"<td><div class='cell-main'>{row.get('time', '')}</div><div class='cell-sub'>{date_text}</div></td>"
            f"<td>{_badge(sev_cls, sev if sev else 'N/A')}</td>"
            f"<td>{_badge(status_cls, status if status else 'N/A')}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="table-wrap">
            <table class="vio-table">
                <thead>
                    <tr>
                        <th>Violation ID</th>
                        <th>Type</th>
                        <th>Location</th>
                        <th>Vehicle</th>
                        <th>Speed</th>
                        <th>Time</th>
                        <th>Severity</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html) if rows_html else '<tr><td colspan="8">No matching records.</td></tr>'}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    dist_col, trend_col = st.columns(2)

    with dist_col:
        st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-head'><h3>Violation Distribution</h3></div>", unsafe_allow_html=True)
        if total_violations == 0:
            st.markdown(
                "<div class='info-box'>Distribution will appear after violations are detected.</div>",
                unsafe_allow_html=True,
            )
        else:
            dist_items = [
                ("Red Light Jump", counts.get("Red Light Jump", 0), "#ef4444"),
                ("Rash Driving", counts.get("Rash Driving", 0), "#f97316"),
                ("No Helmet", counts.get("No Helmet", 0), "#f59e0b"),
                ("Mobile Usage", counts.get("Mobile Usage While Driving", 0), "#ec4899"),
                ("Triple Riding", counts.get("Triple Riding", 0), "#06b6d4"),
                ("Heavy Load", counts.get("Heavy Load", 0), "#8b5cf6"),
                ("Wrong Lane", counts.get("Wrong Lane", 0), "#22c55e"),
                ("Other Violations", counts.get("Other", 0), "#6366f1"),
            ]
            for label, count, color in dist_items:
                percent = round((count / total_violations) * 100, 1) if total_violations else 0
                st.markdown(
                    f"""
                    <div class="progress-row">
                        <div>{label}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width:{percent}%;background:{color};"></div>
                        </div>
                        <div class="progress-right">{percent:.0f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with trend_col:
        st.markdown('<div class="panel" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("<div class='section-head'><h3>Hourly Violation Trend</h3></div>", unsafe_allow_html=True)
        if df.empty or "time" not in df.columns:
            st.markdown(
                "<div class='info-box'>Hourly trend will appear after violations are detected.</div>",
                unsafe_allow_html=True,
            )
        else:
            hours = df["time"].apply(_parse_hour).dropna().astype(int)
            if hours.empty:
                st.markdown(
                    "<div class='info-box'>Hourly trend will appear after violations are detected.</div>",
                    unsafe_allow_html=True,
                )
            else:
                bucket_counts = {
                    "06:00 - 09:00": 0,
                    "09:00 - 12:00": 0,
                    "12:00 - 15:00": 0,
                    "15:00 - 18:00": 0,
                    "18:00 - 21:00": 0,
                }
                for hour in hours:
                    label = _bucket_label(int(hour))
                    if label in bucket_counts:
                        bucket_counts[label] += 1
                max_count = max(bucket_counts.values()) if bucket_counts else 0
                for label, count in bucket_counts.items():
                    percent = (count / max_count * 100) if max_count else 0
                    st.markdown(
                        f"""
                        <div class="progress-row">
                            <div>{label}</div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width:{percent}%;background:#14b8a6;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
