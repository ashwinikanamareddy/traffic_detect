import json
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from backend import analytics


REPORT_DIR = "reports"
INDEX_PATH = os.path.join(REPORT_DIR, "reports_index.json")


def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def _load_index():
    _ensure_report_dir()
    if not os.path.exists(INDEX_PATH):
        return []
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_index(items):
    _ensure_report_dir()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def _human_size(num_bytes):
    if num_bytes is None:
        return "0 B"
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _get_df():
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()
    return None


def _filter_by_date(df, date_range):
    if df is None or df.empty:
        return df
    if not date_range:
        return df

    if isinstance(date_range, dict):
        mode = date_range.get("type")
        start = date_range.get("start")
        end = date_range.get("end")
    else:
        mode = date_range
        start = None
        end = None

    ts = None
    for col in ["timestamp", "time", "datetime", "date"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if not ts.isna().all():
                break
    if ts is None or ts.isna().all():
        return df

    now = datetime.now()
    if mode == "Today":
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
    elif mode == "Last 7 Days":
        end = now
        start = end - timedelta(days=7)
    elif mode == "Last 30 Days":
        end = now
        start = end - timedelta(days=30)

    if start is None or end is None:
        return df

    return df[(ts >= start) & (ts < end)]


def _collect_report_data(report_type, df, sections):
    metrics = st.session_state.get("metrics", {}) or {}
    violations = st.session_state.get("violations", []) or []
    violations_df = st.session_state.get("violations_df", None)

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": report_type,
        "metrics": metrics,
    }

    if "Executive Summary" in sections:
        data["executive_summary"] = analytics.get_trend_metrics("Last 7 Days")

    if "Detailed Statistics" in sections:
        data["detailed_statistics"] = {
            "total_vehicles": int(metrics.get("total_vehicles", 0)),
            "queue_count": int(metrics.get("queue_count", 0)),
            "total_violations": int(metrics.get("total_violations", 0)),
            "red_light_violations": int(metrics.get("red_light_violations", 0)),
            "rash_driving": int(metrics.get("rash_driving", 0)),
            "no_helmet_violations": int(metrics.get("no_helmet_violations", 0)),
            "mobile_usage_violations": int(metrics.get("mobile_usage_violations", 0)),
            "triple_riding_violations": int(metrics.get("triple_riding_violations", 0)),
            "heavy_load_violations": int(metrics.get("heavy_load_violations", 0)),
        }

    if "Charts and Graphs" in sections:
        data["weekly"] = analytics.get_weekly_data()

    if "Raw Data Tables" in sections and df is not None:
        data["raw_rows"] = int(len(df))

    if report_type == "Violation Report" or "Violation Evidence Images" in sections:
        if violations_df is not None and isinstance(violations_df, pd.DataFrame) and not violations_df.empty:
            data["violations"] = violations_df.to_dict(orient="records")
        elif violations:
            data["violations"] = violations
        else:
            data["violations"] = []

    if report_type == "Queue Analysis Report":
        if df is not None and "queue_count" in df.columns:
            q = pd.to_numeric(df["queue_count"], errors="coerce").dropna()
            data["queue"] = {
                "avg_queue": float(q.mean()) if not q.empty else 0.0,
                "peak_queue": float(q.max()) if not q.empty else 0.0,
            }

    if report_type == "Hourly Trends Report":
        data["peak_hours"] = analytics.get_peak_hours()

    if report_type == "Camera Performance Report":
        if df is not None and "camera_id" in df.columns:
            cam_counts = df["camera_id"].astype(str).str.strip()
            cam_counts = cam_counts[cam_counts != ""]
            data["camera_performance"] = cam_counts.value_counts().to_dict()
        else:
            data["camera_performance"] = {}

    return data


def _write_pdf(path, title, data):
    lines = [title, f"Generated: {data.get('generated_at', '')}"]
    for key, value in data.items():
        if key in {"generated_at", "report_type"}:
            continue
        if isinstance(value, (dict, list)):
            lines.append(f"{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"  - items: {len(value)}")
        else:
            lines.append(f"{key}: {value}")

    lines = [line.encode("ascii", "ignore").decode("ascii") for line in lines]

    def _escape(text):
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    y = 760
    content_lines = []
    for line in lines[:60]:
        content_lines.append(f"50 {y} Td ({_escape(line)}) Tj")
        y -= 14
    content = "BT /F1 12 Tf " + " ".join(content_lines) + " ET"
    content_bytes = content.encode("latin-1", errors="ignore")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj")
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    objects.append(b"5 0 obj << /Length %d >> stream\n%s\nendstream endobj" % (len(content_bytes), content_bytes))

    xref_offsets = []
    pdf = b"%PDF-1.4\n"
    for obj in objects:
        xref_offsets.append(len(pdf))
        pdf += obj + b"\n"

    xref_pos = len(pdf)
    pdf += b"xref\n0 %d\n" % (len(objects) + 1)
    pdf += b"0000000000 65535 f \n"
    for off in xref_offsets:
        pdf += f"{off:010d} 00000 n \n".encode("ascii")
    pdf += b"trailer << /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    pdf += b"startxref\n%d\n%%EOF" % xref_pos

    with open(path, "wb") as f:
        f.write(pdf)


def generate_report(report_type, date_range, format, sections):
    _ensure_report_dir()
    df = _get_df()
    df = _filter_by_date(df, date_range)
    data = _collect_report_data(report_type, df, sections)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_type = report_type.replace(" ", "_").lower()
    fmt = format.lower()
    filename = f"{safe_type}_{timestamp}.{fmt}"
    path = os.path.join(REPORT_DIR, filename)

    if fmt == "csv":
        if "Raw Data Tables" in sections and df is not None:
            df.to_csv(path, index=False)
        else:
            pd.json_normalize(data).to_csv(path, index=False)
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif fmt == "excel":
        try:
            with pd.ExcelWriter(path) as writer:
                pd.json_normalize(data).to_excel(writer, sheet_name="Summary", index=False)
                if "Raw Data Tables" in sections and df is not None:
                    df.to_excel(writer, sheet_name="RawData", index=False)
                viols = data.get("violations", [])
                if isinstance(viols, list) and viols:
                    pd.DataFrame(viols).to_excel(writer, sheet_name="Violations", index=False)
        except Exception:
            return None
    elif fmt == "pdf":
        _write_pdf(path, report_type, data)
    else:
        return None

    size = os.path.getsize(path) if os.path.exists(path) else 0
    entry = {
        "name": report_type,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "format": format.upper(),
        "size": _human_size(size),
        "path": path,
    }

    index = _load_index()
    index.insert(0, entry)
    _save_index(index[:10])
    return path


def get_recent_reports():
    index = _load_index()
    cleaned = []
    for item in index:
        path = item.get("path")
        if path and os.path.exists(path):
            size = _human_size(os.path.getsize(path))
            cleaned.append(
                {
                    "name": item.get("name", "Report"),
                    "date": item.get("date", ""),
                    "format": item.get("format", ""),
                    "size": size,
                    "path": path,
                }
            )
    return cleaned
