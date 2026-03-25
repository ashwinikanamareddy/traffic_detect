import os
import time
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from backend.process_video import process_frame


def _to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def _init_state():
    defaults = {
        "live_running": False,
        "live_paused": False,
        "live_video_path": None,
        "live_frame_index": 0,
        "live_fps": 24.0,
        "live_rows": [],
        "live_event_log": [],
        "live_queue_threshold": 8,
        "live_violation_threshold": 20,
        "run_dir": None,
        "processed_video_path": None,
        "violations_df": None,
        "vehicle_counts": {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "autos": 0},
        "processed": False,
        "df": None,
        "metrics": {
            "total_vehicles": 0,
            "queue_count": 0,
            "total_violations": 0,
            "red_light_violations": 0,
            "rash_driving": 0,
            "no_helmet_violations": 0,
            "mobile_usage_violations": 0,
            "triple_riding_violations": 0,
            "heavy_load_violations": 0,
            "cars": 0,
            "bikes": 0,
            "buses": 0,
            "trucks": 0,
            "autos": 0,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _icon_svg(name: str) -> str:
    icons = {
        "car": (
            '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M3 12h18l-1.4-4.2A2 2 0 0 0 17.7 6H6.3a2 2 0 0 0-1.9 1.8z"/>'
            '<path d="M3 12v5h2"/><path d="M21 12v5h-2"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/>'
            "</svg>"
        ),
        "bike": (
            '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<circle cx="5.5" cy="17.5" r="3.5"/><circle cx="18.5" cy="17.5" r="3.5"/>'
            '<path d="m5.5 17.5 5-8h3l2 3h3"/><path d="M12 9.5 10 5H7"/>'
            "</svg>"
        ),
        "bus": (
            '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<rect x="4" y="4" width="16" height="13" rx="2"/><path d="M4 11h16"/><circle cx="8" cy="18" r="1.5"/><circle cx="16" cy="18" r="1.5"/>'
            "</svg>"
        ),
        "truck": (
            '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M3 7h11v9H3z"/><path d="M14 10h4l3 3v3h-7z"/><circle cx="7" cy="18" r="2"/><circle cx="18" cy="18" r="2"/>'
            "</svg>"
        ),
        "auto": (
            '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M4 13h16l-1.2-4a2 2 0 0 0-1.9-1.4H7.1A2 2 0 0 0 5.2 9z"/>'
            '<path d="M4 13v4h2m14-4v4h-2"/><path d="M8 8V6h8v2"/>'
            '<circle cx="8" cy="17" r="2"/><circle cx="16" cy="17" r="2"/>'
            "</svg>"
        ),
    }
    return icons.get(name, "")


def _save_uploaded_file(uploaded):
    os.makedirs("uploads", exist_ok=True)
    file_name = os.path.basename(uploaded.name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("uploads", f"{ts}_{file_name}")
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def _camera_id_from_file(path: str) -> str:
    base = os.path.basename(path)
    return f"CAM-{abs(hash(base)) % 90 + 10:02d}"


def _append_live_row(camera_id: str, counts: dict, queue_count: int, events: list, frame_idx: int):
    red = _to_int(st.session_state.metrics.get("red_light_violations", 0), 0)
    rash = _to_int(st.session_state.metrics.get("rash_driving", 0), 0)

    for event in events:
        if event.get("type") == "violation":
            red += 1
        if event.get("type") == "high_queue":
            rash += 1

    total = int(sum(counts.values()))
    queue_density = float(queue_count) / float(max(1, total))

    row = {
        "frame": int(frame_idx),
        "camera_id": camera_id,
        "location": "Uploaded Feed",
        "cars": int(counts.get("cars", 0)),
        "bikes": int(counts.get("bikes", 0)),
        "buses": int(counts.get("buses", 0)),
        "trucks": int(counts.get("trucks", 0)),
        "autos": int(counts.get("autos", 0)),
        "total_vehicles": total,
        "queue_count": int(queue_count),
        "queue_density": queue_density,
        "red_light_violations": red,
        "rash_driving": rash,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    st.session_state.live_rows.append(row)
    st.session_state.df = pd.DataFrame(st.session_state.live_rows)
    st.session_state.metrics = {
        "cars": row["cars"],
        "bikes": row["bikes"],
        "buses": row["buses"],
        "trucks": row["trucks"],
        "autos": row["autos"],
        "total_vehicles": row["total_vehicles"],
        "queue_count": row["queue_count"],
        "queue_density_avg": row["queue_density"],
        "total_violations": _to_int(row["red_light_violations"], 0) + _to_int(row["rash_driving"], 0),
        "red_light_violations": row["red_light_violations"],
        "rash_driving": row["rash_driving"],
        "no_helmet_violations": 0,
        "mobile_usage_violations": 0,
        "triple_riding_violations": 0,
        "heavy_load_violations": 0,
    }


def _render_kpis(holder, counts):
    with holder.container():
        k1, k2, k3, k4, k5 = st.columns(5)
        cards = [
            ("Cars", counts.get("cars", 0), "car", "#dffaf3", "#0f9f96"),
            ("Bikes", counts.get("bikes", 0), "bike", "#fff4dd", "#ea580c"),
            ("Buses", counts.get("buses", 0), "bus", "#e8ecff", "#4f46e5"),
            ("Trucks", counts.get("trucks", 0), "truck", "#ffe4e6", "#dc2626"),
            ("Autos", counts.get("autos", 0), "auto", "#fdf2f8", "#db2777"),
        ]
        for col, (title, value, icon_key, bg, fg) in zip([k1, k2, k3, k4, k5], cards):
            with col:
                st.markdown(
                    f"""
                    <div class="live-kpi">
                        <div class="live-kpi-top">
                            <div class="live-kpi-icon" style="background:{bg};color:{fg};">{_icon_svg(icon_key)}</div>
                        </div>
                        <div class="live-kpi-label">{title}</div>
                        <div class="live-kpi-value">{int(value)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_event_panel(event_holder):
    with event_holder.container():
        st.markdown("### Recent Events")
        events = st.session_state.live_event_log[-8:]
        if not events:
            st.info("No events yet.")
            return

        for ev in reversed(events):
            ev_type = str(ev.get("type", "event")).replace("_", " ").title()
            st.markdown(
                f"""
                <div class="event-item">
                    <p class="event-type">{ev_type}</p>
                    <p class="event-meta">{ev.get('message', '')}</p>
                    <p class="event-meta">{ev.get('camera_id', 'CAM-00')} | frame {ev.get('frame', 0)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _read_pause_frame(path: str, frame_idx: int, settings: dict, camera_id: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False, None

    processed, _, _ = process_frame(
        frame,
        settings=settings,
        frame_number=frame_idx,
        camera_id=camera_id,
        draw_overlays=True,
    )
    return True, processed


def show():
    _init_state()

    st.markdown(
        """
        <style>
        .live-head {
            background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
            border: 1px solid #e6edf4;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
            padding: 16px 18px;
            margin-bottom: 14px;
        }
        .live-head h1 { margin: 0; font-size: 36px; color: #0f172a; line-height: 1.1; }
        .live-head p { margin: 6px 0 0 0; color: #64748b; font-size: 14px; }
        .live-card { background: #ffffff; border: 1px solid #e7edf4; border-radius: 16px; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05); padding: 14px; }
        .live-kpi { background: #ffffff; border: 1px solid #e6edf4; border-radius: 14px; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05); padding: 12px; }
        .live-kpi-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .live-kpi-icon { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
        .live-kpi-icon svg { width: 18px; height: 18px; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
        .live-kpi-label { color: #64748b; font-size: 13px; }
        .live-kpi-value { color: #0f172a; font-size: 34px; font-weight: 800; line-height: 1; }
        .event-item { background: #f8fafc; border: 1px solid #e7edf4; border-radius: 12px; padding: 10px; margin-bottom: 8px; }
        .event-type { font-size: 13px; font-weight: 700; color: #0f172a; margin: 0; }
        .event-meta { font-size: 12px; color: #64748b; margin: 2px 0 0 0; }
        @media (max-width: 900px) { .live-head h1 { font-size: 30px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="live-head">
            <h1>Live Video Feed</h1>
            <p>Real-time camera monitoring with AI detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov", "mkv"])
    last_uploaded = st.session_state.get("last_uploaded_video_path")
    if last_uploaded and os.path.exists(last_uploaded):
        st.caption(f"Using last uploaded video from dashboard: {os.path.basename(last_uploaded)}")

    b1, b2, b3 = st.columns(3)
    start_clicked = b1.button("Start Live Feed", width="stretch")
    play_clicked = b2.button("Play", width="stretch")
    pause_clicked = b3.button("Pause", width="stretch")
    stop_clicked = st.button("Stop", width="stretch")

    if start_clicked:
        if not uploaded and not last_uploaded:
            st.warning("Upload a video or process one from dashboard to start live feed.")
            st.session_state.live_running = False
        else:
            if uploaded:
                path = _save_uploaded_file(uploaded)
            else:
                path = last_uploaded
            st.session_state.live_video_path = path
            st.session_state.live_frame_index = 0
            st.session_state.live_rows = []
            st.session_state.live_event_log = []
            st.session_state.vehicle_counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "autos": 0}
            st.session_state.metrics = {
                "cars": 0,
                "bikes": 0,
                "buses": 0,
                "trucks": 0,
                "autos": 0,
                "total_vehicles": 0,
                "queue_count": 0,
                "red_light_violations": 0,
                "rash_driving": 0,
            }
            st.session_state.df = None
            st.session_state.processed = True
            st.session_state.live_running = True
            st.session_state.live_paused = False

    if play_clicked and st.session_state.live_video_path:
        st.session_state.live_running = True
        st.session_state.live_paused = False

    if pause_clicked:
        st.session_state.live_paused = True

    if stop_clicked:
        st.session_state.live_running = False
        st.session_state.live_paused = False
        st.session_state.live_frame_index = 0

    left, right = st.columns([3, 1])

    with left:
        st.markdown('<div class="live-card">', unsafe_allow_html=True)
        settings = {
            "bounding_boxes": st.checkbox("Bounding Boxes", value=True, key="live_bbox"),
            "vehicle_ids": st.checkbox("Vehicle IDs", value=True, key="live_ids"),
            "queue_zones": st.checkbox("Queue Zones", value=True, key="live_queue_zones"),
            "violation_alerts": st.checkbox("Violation Alerts", value=True, key="live_violations"),
            "highlight_violation_red": st.checkbox("Red Violation Highlight", value=True, key="live_violation_red"),
            "queue_threshold": st.session_state.live_queue_threshold,
            "violation_vehicle_threshold": st.session_state.live_violation_threshold,
            "conf_threshold": 0.35,
            "detect_imgsz": 416,
        }
        frame_placeholder = st.empty()
        kpi_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="live-card">', unsafe_allow_html=True)
        event_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    _render_kpis(kpi_placeholder, st.session_state.vehicle_counts)
    _render_event_panel(event_placeholder)

    current_path = st.session_state.live_video_path
    if not current_path:
        frame_placeholder.info("Upload and start live feed")
        return

    cap = cv2.VideoCapture(current_path)
    if not cap.isOpened():
        frame_placeholder.error("Unable to open selected video.")
        st.session_state.live_running = False
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.live_frame_index)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 24.0
    st.session_state.live_fps = float(fps)

    camera_id = _camera_id_from_file(current_path)

    rendered_any_frame = False
    while st.session_state.live_running and not st.session_state.live_paused:
        ok, frame = cap.read()
        if not ok:
            if not rendered_any_frame:
                frame_placeholder.error(
                    "Unable to decode frames from this video. Try a standard H.264 MP4 file."
                )
            st.session_state.live_running = False
            break

        frame_idx = st.session_state.live_frame_index
        processed_frame, counts, events = process_frame(
            frame,
            settings=settings,
            frame_number=frame_idx,
            camera_id=camera_id,
            draw_overlays=True,
        )

        queue_count = max(0, int(sum(counts.values())) - int(counts.get("bikes", 0)))
        st.session_state.vehicle_counts = counts

        _append_live_row(
            camera_id=camera_id,
            counts=counts,
            queue_count=queue_count,
            events=events,
            frame_idx=frame_idx,
        )

        for event in events:
            st.session_state.live_event_log.append(
                {
                    "type": event.get("type", "event"),
                    "message": event.get("message", ""),
                    "camera_id": event.get("camera_id", camera_id),
                    "frame": event.get("frame", frame_idx),
                }
            )
        st.session_state.live_event_log = st.session_state.live_event_log[-100:]

        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", width="stretch")
        rendered_any_frame = True
        _render_kpis(kpi_placeholder, st.session_state.vehicle_counts)
        _render_event_panel(event_placeholder)

        st.session_state.live_frame_index += 1
        time.sleep(1.0 / max(1.0, st.session_state.live_fps))

    cap.release()

    if st.session_state.live_paused:
        ok, frame = _read_pause_frame(current_path, st.session_state.live_frame_index, settings, camera_id)
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", width="stretch")
        else:
            frame_placeholder.info("Paused")
