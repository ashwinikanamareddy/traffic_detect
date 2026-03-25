from datetime import datetime, timedelta

import pandas as pd
import streamlit as st


def _get_df():
    df = st.session_state.get("df", None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()
    return None


def _timestamp_series(df):
    if df is None or df.empty:
        return None
    for col in ["timestamp", "time", "datetime", "date"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if not ts.isna().all():
                return ts
    return None


def _frame_traffic(df):
    if df is None or df.empty:
        return pd.Series(dtype="float")
    if "frame" in df.columns and "total_vehicles" in df.columns:
        return df.drop_duplicates(subset=["frame"])["total_vehicles"].fillna(0)
    if "frame_id" in df.columns and "total_vehicles" in df.columns:
        return df.drop_duplicates(subset=["frame_id"])["total_vehicles"].fillna(0)
    if "track_id" in df.columns:
        return pd.Series([df["track_id"].nunique()])
    return pd.Series([len(df)])


def _violation_events(df):
    if df is None or df.empty or "violation_type" not in df.columns:
        return pd.Series(dtype="object")
    raw = df["violation_type"].fillna("").astype(str)
    raw = raw[raw.str.len() > 0]
    if raw.empty:
        return pd.Series(dtype="object")
    exploded = raw.str.split("|").explode().str.strip()
    exploded = exploded[exploded.str.len() > 0]
    return exploded


def _period_bounds(time_range):
    now = datetime.now()
    if time_range == "Today":
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
        prev_start = start - timedelta(days=1)
        prev_end = start
        return start, end, prev_start, prev_end
    if time_range == "Last 7 Days":
        start = now - timedelta(days=7)
        end = now
        prev_start = start - timedelta(days=7)
        prev_end = start
        return start, end, prev_start, prev_end
    if time_range == "Last 30 Days":
        start = now - timedelta(days=30)
        end = now
        prev_start = start - timedelta(days=30)
        prev_end = start
        return start, end, prev_start, prev_end
    return None, None, None, None


def _avg_speed_from_df(df):
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


def get_trend_metrics(time_range):
    df = _get_df()
    if df is None:
        return {
            "traffic_growth": 0.0,
            "violation_rate_change": 0.0,
            "avg_speed": 0.0,
            "violation_percent": 0.0,
        }

    ts = _timestamp_series(df)
    start, end, prev_start, prev_end = _period_bounds(time_range)
    if ts is None or start is None:
        curr_df = df
        prev_df = df.iloc[0:0]
    else:
        curr_df = df[(ts >= start) & (ts < end)].copy()
        prev_df = df[(ts >= prev_start) & (ts < prev_end)].copy()

    curr_traffic = float(_frame_traffic(curr_df).sum())
    prev_traffic = float(_frame_traffic(prev_df).sum())

    curr_violations = float(_violation_events(curr_df).shape[0])
    prev_violations = float(_violation_events(prev_df).shape[0])

    traffic_growth = 0.0 if prev_traffic == 0 else ((curr_traffic - prev_traffic) / prev_traffic) * 100.0
    curr_rate = 0.0 if curr_traffic == 0 else (curr_violations / curr_traffic) * 100.0
    prev_rate = 0.0 if prev_traffic == 0 else (prev_violations / prev_traffic) * 100.0
    violation_rate_change = 0.0 if prev_rate == 0 else ((curr_rate - prev_rate) / prev_rate) * 100.0

    avg_speed = _avg_speed_from_df(curr_df) or 0.0
    violation_percent = curr_rate

    return {
        "traffic_growth": float(traffic_growth),
        "violation_rate_change": float(violation_rate_change),
        "avg_speed": float(avg_speed),
        "violation_percent": float(violation_percent),
    }


def get_weekly_data():
    df = _get_df()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if df is None:
        return {
            "daily_traffic": {d: 0 for d in days},
            "daily_violations": {d: 0 for d in days},
        }

    ts = _timestamp_series(df)
    if ts is None:
        return {
            "daily_traffic": {d: 0 for d in days},
            "daily_violations": {d: 0 for d in days},
        }

    recent_start = datetime.now() - timedelta(days=7)
    recent_mask = ts >= recent_start
    recent_df = df[recent_mask].copy()
    recent_ts = ts[recent_mask]

    traffic_series = _frame_traffic(recent_df)
    if "frame" in recent_df.columns:
        traffic_by_day = recent_df.drop_duplicates(subset=["frame"]).assign(day=recent_ts.dt.day_name().str[:3])
        traffic_map = traffic_by_day.groupby("day")["total_vehicles"].sum().to_dict() if "total_vehicles" in traffic_by_day.columns else {}
    else:
        traffic_map = {}

    violations = _violation_events(recent_df)
    viol_df = recent_df[recent_df["violation_type"].fillna("").astype(str).str.len() > 0] if "violation_type" in recent_df.columns else recent_df.iloc[0:0]
    viol_day_map = {}
    if not viol_df.empty:
        viol_day = recent_ts.loc[viol_df.index].dt.day_name().str[:3]
        counts = viol_day.value_counts().to_dict()
        viol_day_map = counts

    daily_traffic = {d: int(traffic_map.get(d, 0)) for d in days}
    if sum(daily_traffic.values()) == 0 and not traffic_series.empty:
        daily_traffic = {d: 0 for d in days}

    daily_violations = {d: int(viol_day_map.get(d, 0)) for d in days}

    return {
        "daily_traffic": daily_traffic,
        "daily_violations": daily_violations,
    }


def get_peak_hours():
    df = _get_df()
    if df is None:
        return []
    ts = _timestamp_series(df)
    if ts is None:
        return []

    hours = ts.dt.hour
    traffic = _frame_traffic(df)
    if "frame" in df.columns and "total_vehicles" in df.columns:
        hour_df = df.drop_duplicates(subset=["frame"]).assign(hour=hours)
        traffic_by_hour = hour_df.groupby("hour")["total_vehicles"].sum().to_dict()
    else:
        traffic_by_hour = hours.value_counts().to_dict()

    viol_df = df[df["violation_type"].fillna("").astype(str).str.len() > 0] if "violation_type" in df.columns else df.iloc[0:0]
    viol_hour_map = {}
    if not viol_df.empty:
        viol_hours = hours.loc[viol_df.index]
        viol_hour_map = viol_hours.value_counts().to_dict()

    items = []
    for hour, traffic_val in traffic_by_hour.items():
        start = int(hour)
        end = (start + 1) % 24
        label = f"{start:02d}:00 - {end:02d}:00"
        items.append(
            {
                "time_range": label,
                "traffic": int(traffic_val),
                "violations": int(viol_hour_map.get(hour, 0)),
            }
        )

    items.sort(key=lambda x: x["traffic"], reverse=True)
    return items


def get_predictive_insights():
    df = _get_df()
    if df is None:
        return {
            "tomorrow_forecast": 0,
            "violation_prediction": [0, 0],
            "queue_forecast": [0, 0],
            "congestion_alert": False,
        }

    ts = _timestamp_series(df)
    if ts is None:
        return {
            "tomorrow_forecast": 0,
            "violation_prediction": [0, 0],
            "queue_forecast": [0, 0],
            "congestion_alert": False,
        }

    daily_frames = df.drop_duplicates(subset=["frame"]) if "frame" in df.columns else df
    daily_frames["date"] = ts.dt.date
    daily_traffic = daily_frames.groupby("date")["total_vehicles"].sum() if "total_vehicles" in daily_frames.columns else daily_frames.groupby("date").size()
    traffic_avg = float(daily_traffic.tail(7).mean()) if not daily_traffic.empty else 0.0

    viol_series = _violation_events(df)
    if not viol_series.empty:
        viol_df = df[df["violation_type"].fillna("").astype(str).str.len() > 0]
        viol_df = viol_df.assign(date=ts.loc[viol_df.index].dt.date)
        daily_viol = viol_df.groupby("date").size()
        viol_avg = float(daily_viol.tail(7).mean()) if not daily_viol.empty else 0.0
    else:
        viol_avg = 0.0

    queue_avg = 0.0
    if "queue_count" in df.columns:
        queue_avg = float(pd.to_numeric(df["queue_count"], errors="coerce").dropna().mean() or 0.0)

    viol_min = int(max(0, viol_avg * 0.9))
    viol_max = int(max(0, viol_avg * 1.1))
    queue_min = int(max(0, queue_avg * 0.8))
    queue_max = int(max(0, queue_avg * 1.2))

    congestion_alert = bool(queue_avg >= 12 or (traffic_avg > 0 and (viol_avg / max(1.0, traffic_avg)) > 0.05))

    return {
        "tomorrow_forecast": int(round(traffic_avg)),
        "violation_prediction": [viol_min, viol_max],
        "queue_forecast": [queue_min, queue_max],
        "congestion_alert": congestion_alert,
    }
