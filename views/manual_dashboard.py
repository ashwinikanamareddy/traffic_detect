import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from backend.detection import detect_vehicles
from backend.queue_analyzer import lane_wise_queue_stats, point_in_polygon
from backend.tracking import MultiObjectTracker
from backend.video_loader import get_first_frame, save_uploaded_video
from backend.violation_detector import RedLightViolationDetector

# Compatibility shim for streamlit-drawable-canvas with newer Streamlit versions.
try:
    import streamlit.elements.image as _st_image_mod

    if not hasattr(_st_image_mod, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

        def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            return _new_image_to_url(
                image=image,
                layout_config=_LayoutConfig(width=int(width) if isinstance(width, int) else width),
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )

        _st_image_mod.image_to_url = _image_to_url_compat
except Exception:
    pass

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None


Point = Tuple[int, int]
Polygon = List[Point]


def _init_state():
    defaults = {
        "manual_video_path": None,
        "manual_first_frame": None,
        "manual_display_frame": None,
        "manual_scale_x": 1.0,
        "manual_scale_y": 1.0,
        "manual_uploaded_token": None,
        "manual_config_path": os.path.join("history", "manual_config.json"),
        "manual_lanes": [],
        "manual_lane_locked": False,
        "manual_lane_stage": "idle",
        "manual_stop_line": None,
        "manual_stop_locked": False,
        "manual_lane_canvas_seed": 0,
        "manual_stop_canvas_seed": 0,
        "manual_config_mode": "",
        "manual_signal_state": "RED",
        "manual_running": False,
        "manual_analysis_started": False,
        "manual_frame_idx": 0,
        "manual_tracker": None,
        "manual_violation_detector": None,
        "manual_violations": [],
        "manual_class_dist": {"car": 0, "bike": 0, "bus": 0, "truck": 0, "auto": 0},
        "manual_total_vehicle_count": 0,
        "manual_seen_track_ids": set(),
        "manual_lane_stats": [],
        "manual_latest_frame": None,
        "manual_export_path": None,
        "manual_writer": None,
        "manual_export_ready": False,
        "manual_cap": None,
        "manual_det_conf": 0.35,
        "manual_trajectory_history": {},
        "manual_frame_shape": None,
        "manual_frames_per_step": 2,
        "manual_rash_ids": set(),
        "manual_rash_streaks": {},
        "manual_rash_speed_threshold": 38.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Strict manual mode: do not auto-populate lanes/stop-line from previous config.


def _load_config(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_config(path: str, updates: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _load_config(path)
    data.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _parse_lanes_from_config(cfg: Dict) -> List[Polygon]:
    lanes_out: List[Polygon] = []
    raw_lanes = cfg.get("lanes")
    if not isinstance(raw_lanes, list):
        return lanes_out

    for lane in raw_lanes:
        points_raw = lane.get("points") if isinstance(lane, dict) else lane
        if not isinstance(points_raw, list):
            continue
        pts: Polygon = []
        for p in points_raw:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pts.append((int(p[0]), int(p[1])))
        if len(pts) == 4:
            lanes_out.append(pts)
    return lanes_out


def _parse_stopline_from_config(cfg: Dict) -> Optional[Tuple[Point, Point]]:
    raw = cfg.get("stop_line")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    p1, p2 = raw
    if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple))):
        return None
    if len(p1) != 2 or len(p2) != 2:
        return None
    return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def _prepare_display_frame(frame, max_width: int = 1100):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame.copy(), 1.0, 1.0
    scale = float(max_width) / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    disp = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return disp, float(w) / float(new_w), float(h) / float(new_h)


def _bgr_to_pil(frame) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _release_writer():
    writer = st.session_state.get("manual_writer")
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    st.session_state.manual_writer = None


def _release_capture():
    cap = st.session_state.get("manual_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.manual_cap = None


def _extract_points_from_canvas(canvas_data: Optional[Dict]) -> List[Point]:
    points: List[Point] = []
    objects = (canvas_data or {}).get("objects", [])
    for obj in objects:
        if str(obj.get("type", "")).lower() != "circle":
            continue
        x = int(round(float(obj.get("left", 0.0)) + float(obj.get("radius", 0.0))))
        y = int(round(float(obj.get("top", 0.0)) + float(obj.get("radius", 0.0))))
        points.append((x, y))
    return points


def _to_original_point(display_point: Point, first_frame) -> Point:
    sx = float(st.session_state.get("manual_scale_x", 1.0))
    sy = float(st.session_state.get("manual_scale_y", 1.0))
    h, w = first_frame.shape[:2]
    ox = max(0, min(w - 1, int(round(display_point[0] * sx))))
    oy = max(0, min(h - 1, int(round(display_point[1] * sy))))
    return ox, oy


def _to_display_point(original_point: Point) -> Point:
    sx = float(st.session_state.get("manual_scale_x", 1.0))
    sy = float(st.session_state.get("manual_scale_y", 1.0))
    dx = int(round(original_point[0] / max(sx, 1e-9)))
    dy = int(round(original_point[1] / max(sy, 1e-9)))
    return dx, dy


def _draw_lane_preview(display_frame, saved_lanes: List[Polygon], current_points: List[Point]):
    out = display_frame.copy()

    for idx, lane in enumerate(saved_lanes):
        lane_disp = [_to_display_point(p) for p in lane]
        pts = np.array(lane_disp, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        center = np.mean(np.array(lane_disp, dtype=np.float32), axis=0)
        cv2.putText(out, f"Lane {idx + 1}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    for p in current_points[:4]:
        cv2.circle(out, p, 5, (0, 255, 0), -1)

    if len(current_points) >= 2:
        pts = np.array(current_points[:4], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False if len(current_points) < 4 else True, (0, 255, 0), 2)
    if len(current_points) == 4:
        center = np.mean(np.array(current_points, dtype=np.float32), axis=0)
        cv2.putText(
            out,
            f"Lane {len(saved_lanes) + 1}",
            (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def _draw_stop_preview(display_frame, stop_line: Optional[Tuple[Point, Point]], current_points: List[Point]):
    out = display_frame.copy()
    if stop_line is not None:
        p1 = _to_display_point(stop_line[0])
        p2 = _to_display_point(stop_line[1])
        cv2.line(out, p1, p2, (0, 0, 255), 3)
    for p in current_points[:2]:
        cv2.circle(out, p, 5, (0, 0, 255), -1)
    if len(current_points) == 2:
        cv2.line(out, current_points[0], current_points[1], (0, 0, 255), 3)
    return out


def _lane_id_for_point(point: Tuple[float, float], lanes: List[Polygon]) -> Optional[int]:
    for idx, lane in enumerate(lanes):
        if point_in_polygon(point, lane):
            return idx + 1
    return None


def _annotate_frame(frame, tracks, lanes: List[Polygon], stop_line, signal_state: str, violations: List[Dict], rash_ids=None):
    canvas = frame.copy()
    violation_types_by_id: Dict[int, List[str]] = {}
    for v in violations:
        tid = int(v.get("track_id", -1))
        if tid < 0:
            continue
        vtype = str(v.get("type", v.get("violation_type", "Violation")))
        violation_types_by_id.setdefault(tid, []).append(vtype)
    violating_ids = set(violation_types_by_id.keys())
    rash_ids = set() if rash_ids is None else {int(x) for x in rash_ids}

    for idx, lane in enumerate(lanes):
        pts = np.array(lane, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
        center = np.mean(np.array(lane, dtype=np.float32), axis=0)
        cv2.putText(canvas, f"Lane {idx + 1}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if stop_line is not None:
        cv2.line(canvas, stop_line[0], stop_line[1], (0, 0, 255), 3)

    for tr in tracks:
        tid = int(tr.get("track_id", -1))
        x, y, w, h = tr.get("bbox", (0, 0, 1, 1))
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        vehicle_type = str(tr.get("vehicle_type", "vehicle"))
        is_rash = tid in rash_ids
        color = (0, 0, 255) if (tid in violating_ids or is_rash) else (56, 189, 248)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        label = f"ID {tid} | {vehicle_type.upper()}"
        if tid in violating_ids:
            label += f" | {violation_types_by_id.get(tid, ['Violation'])[0]}"
        if is_rash:
            label += " | RASH"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )

    sig_color = (0, 0, 255) if str(signal_state).upper() == "RED" else (0, 180, 0)
    cv2.rectangle(canvas, (10, 10), (360, 60), (15, 23, 42), -1)
    cv2.putText(canvas, f"Signal: {signal_state}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sig_color, 2, cv2.LINE_AA)
    return canvas


def _start_analysis():
    if st.session_state.manual_running:
        return

    video_path = st.session_state.get("manual_video_path")
    lanes = st.session_state.get("manual_lanes", [])
    stop_line = st.session_state.get("manual_stop_line")
    if not video_path or not os.path.exists(video_path):
        st.error("Upload a video first.")
        return
    if not st.session_state.get("manual_lane_locked", False) or not lanes:
        st.error("Configure and save all lanes first.")
        return
    if not st.session_state.get("manual_stop_locked", False) or stop_line is None:
        st.error("Configure and save stop line first.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps is None or fps <= 0:
        fps = 24.0
    if width <= 0 or height <= 0:
        cap.release()
        st.error("Unable to read video dimensions.")
        return

    os.makedirs("history", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join("history", f"manual_annotated_{ts}.mp4")
    writer = cv2.VideoWriter(export_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))

    stop = st.session_state.manual_stop_line
    st.session_state.manual_tracker = MultiObjectTracker(iou_threshold=0.28, max_age=18, min_hits=2, max_center_distance=95.0)
    st.session_state.manual_violation_detector = RedLightViolationDetector(
        stop_line=((float(stop[0][0]), float(stop[0][1])), (float(stop[1][0]), float(stop[1][1])))
    )
    st.session_state.manual_violations = []
    st.session_state.manual_class_dist = {"car": 0, "bike": 0, "bus": 0, "truck": 0, "auto": 0}
    st.session_state.manual_total_vehicle_count = 0
    st.session_state.manual_seen_track_ids = set()
    st.session_state.manual_lane_stats = []
    st.session_state.manual_frame_idx = 0
    st.session_state.manual_trajectory_history = {}
    st.session_state.manual_rash_ids = set()
    st.session_state.manual_rash_streaks = {}
    st.session_state.manual_frame_shape = None
    st.session_state.manual_export_path = export_path
    st.session_state.manual_writer = writer
    st.session_state.manual_cap = cap
    st.session_state.manual_export_ready = False
    st.session_state.manual_running = True
    st.session_state.manual_analysis_started = True


def _process_next_frame():
    if not st.session_state.manual_running:
        return

    frame_idx = int(st.session_state.get("manual_frame_idx", 0))
    lanes = st.session_state.get("manual_lanes", [])
    stop_line = st.session_state.get("manual_stop_line")
    tracker = st.session_state.get("manual_tracker")
    vdet = st.session_state.get("manual_violation_detector")
    signal_state = st.session_state.get("manual_signal_state", "RED")
    cap = st.session_state.get("manual_cap")

    if cap is None:
        cap = cv2.VideoCapture(st.session_state.get("manual_video_path"))
        st.session_state.manual_cap = cap

    ok, frame = cap.read()
    if not ok:
        st.session_state.manual_running = False
        _release_capture()
        _release_writer()
        st.session_state.manual_export_ready = True
        return

    try:
        if frame_idx == 0:
            print(f"[manual_dashboard] analysis frame.shape={frame.shape}")
            st.session_state.manual_frame_shape = tuple(frame.shape)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detections, counts, _ = detect_vehicles(frame, conf_threshold=float(st.session_state.get("manual_det_conf", 0.35)), imgsz=416)
        tracks, _ = tracker.update(detections=detections, frame_id=frame_idx, timestamp=ts)
    except Exception as exc:
        st.session_state.manual_running = False
        _release_capture()
        _release_writer()
        st.session_state.manual_export_ready = True
        st.error(f"Analysis stopped due to runtime error: {exc}")
        return

    st.session_state.manual_lane_stats = lane_wise_queue_stats(tracks, lanes)

    events = vdet.update(tracks, signal_state=signal_state, timestamp=ts)
    if events:
        track_by_id = {int(tr["track_id"]): tr for tr in tracks}
        existing = {(int(v["track_id"]), str(v["timestamp"])) for v in st.session_state.manual_violations}
        for ev in events:
            tid = int(ev["track_id"])
            key = (tid, str(ev["timestamp"]))
            if key in existing:
                continue
            tr = track_by_id.get(tid)
            lane_id = None
            if tr is not None:
                lane_id = _lane_id_for_point((float(tr.get("center_x", 0.0)), float(tr.get("center_y", 0.0))), lanes)
            ev["lane"] = int(lane_id) if lane_id is not None else -1
            st.session_state.manual_violations.append(ev)

    class_map = {"cars": "car", "bikes": "bike", "buses": "bus", "trucks": "truck", "autos": "auto"}
    for k, out_k in class_map.items():
        st.session_state.manual_class_dist[out_k] += int(counts.get(k, 0))
    # Persist trajectory history per vehicle ID.
    hist = st.session_state.get("manual_trajectory_history", {})
    rash_ids = set(st.session_state.get("manual_rash_ids", set()))
    rash_streaks = dict(st.session_state.get("manual_rash_streaks", {}))
    for tr in tracks:
        tid = int(tr.get("track_id", -1))
        cx = float(tr.get("center_x", 0.0))
        cy = float(tr.get("center_y", 0.0))
        path = hist.get(tid, [])
        path.append((cx, cy))
        if len(path) > 120:
            path = path[-120:]
        hist[tid] = path

        speed = 0.0
        if len(path) >= 4:
            p_now = path[-1]
            p_prev = path[-4]
            speed = float(((p_now[0] - p_prev[0]) ** 2 + (p_now[1] - p_prev[1]) ** 2) ** 0.5) / 3.0
        if speed >= float(st.session_state.get("manual_rash_speed_threshold", 38.0)):
            rash_streaks[tid] = int(rash_streaks.get(tid, 0)) + 1
        else:
            rash_streaks[tid] = 0

        if rash_streaks.get(tid, 0) >= 3:
            rash_ids.add(tid)
            existing_rash = {
                (int(v.get("track_id", -1)), str(v.get("type", "")))
                for v in st.session_state.manual_violations
            }
            if (tid, "Rash Driving") not in existing_rash:
                lane_id = _lane_id_for_point((cx, cy), lanes)
                st.session_state.manual_violations.append(
                    {
                        "track_id": tid,
                        "lane": int(lane_id) if lane_id is not None else -1,
                        "timestamp": ts,
                        "type": "Rash Driving",
                    }
                )

    st.session_state.manual_trajectory_history = hist
    st.session_state.manual_rash_ids = rash_ids
    st.session_state.manual_rash_streaks = rash_streaks

    seen_ids = set(st.session_state.get("manual_seen_track_ids", set()))
    for tr in tracks:
        seen_ids.add(int(tr.get("track_id", -1)))
    seen_ids.discard(-1)
    st.session_state.manual_seen_track_ids = seen_ids
    st.session_state.manual_total_vehicle_count = len(seen_ids)

    annotated = _annotate_frame(
        frame,
        tracks,
        lanes,
        stop_line,
        signal_state,
        st.session_state.manual_violations,
        rash_ids=rash_ids,
    )
    st.session_state.manual_latest_frame = annotated
    writer = st.session_state.get("manual_writer")
    if writer is not None:
        writer.write(annotated)
    st.session_state.manual_frame_idx = frame_idx + 1


def _render_lane_config(first_frame):
    st.markdown("### Lane Configuration (Strict: 4 clicks per lane)")
    if st_canvas is None:
        st.error("`streamlit-drawable-canvas` is missing. Install it to enable in-app drawing.")
        return

    locked = bool(st.session_state.get("manual_lane_locked", False))
    running = bool(st.session_state.get("manual_running", False))
    analysis_started = bool(st.session_state.get("manual_analysis_started", False))
    lane_stage = str(st.session_state.get("manual_lane_stage", "idle"))
    if locked:
        st.success("Lane configuration saved and locked.")
    if running or analysis_started:
        st.info("Lane editing is allowed only before analysis starts.")

    display_frame = st.session_state.get("manual_display_frame")
    saved_lanes: List[Polygon] = st.session_state.get("manual_lanes", [])
    preview_bg = _draw_lane_preview(display_frame, saved_lanes, [])

    if locked or running or analysis_started:
        st.image(cv2.cvtColor(preview_bg, cv2.COLOR_BGR2RGB), channels="RGB", caption="Lane Configuration", width="stretch")
        return

    if lane_stage != "drawing":
        st.info("Click 'Add Lane' to start drawing a lane.")
        current_points: List[Point] = []
    else:
        st.caption("Click exactly 4 points for the current lane.")
        bg = _bgr_to_pil(preview_bg)
        canvas = st_canvas(
            fill_color="rgba(0,255,0,0.0)",
            stroke_width=2,
            stroke_color="#00ff55",
            background_image=bg,
            update_streamlit=True,
            height=bg.height,
            width=bg.width,
            drawing_mode="point",
            point_display_radius=4,
            display_toolbar=False,
            key=f"manual_lane_point_canvas_{st.session_state.manual_lane_canvas_seed}_{st.session_state.get('manual_uploaded_token', 'none')}",
        )

        raw_points = _extract_points_from_canvas(canvas.json_data if canvas is not None else None)
        current_points = raw_points[:4]
        if len(raw_points) > 4:
            st.warning("Current lane supports exactly 4 points. Extra clicks are ignored.")

    st.caption(f"Current lane points: {len(current_points)}/4 | Saved lanes: {len(saved_lanes)}")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Add Lane", key="lane_add_lane_btn", width="stretch", disabled=(lane_stage == "drawing")):
        st.session_state.manual_lane_stage = "drawing"
        st.session_state.manual_lane_canvas_seed += 1
        st.rerun()

    if c2.button("Clear Current Lane", key="lane_clear_current_btn", width="stretch", disabled=(lane_stage != "drawing")):
        st.session_state.manual_lane_canvas_seed += 1
        st.rerun()

    if c3.button("Save Lane", key="lane_save_lane_btn", width="stretch", disabled=(lane_stage != "drawing" or len(current_points) != 4)):
        mapped = [_to_original_point(p, first_frame) for p in current_points]
        st.session_state.manual_lanes.append(mapped)
        st.session_state.manual_lane_stage = "idle"
        st.session_state.manual_lane_canvas_seed += 1
        st.success(f"Lane {len(st.session_state.manual_lanes)} saved.")
        st.rerun()

    if c4.button(
        "Finish Configuration",
        key="lane_finish_config_btn",
        width="stretch",
        disabled=(lane_stage == "drawing" or len(st.session_state.manual_lanes) == 0),
    ):
        lanes_payload = [{"id": i + 1, "points": [[int(x), int(y)] for x, y in lane]} for i, lane in enumerate(st.session_state.manual_lanes)]
        _save_config(st.session_state.manual_config_path, {"lanes": lanes_payload})
        st.session_state.manual_lane_locked = True
        st.success(f"Lane configuration finished with {len(lanes_payload)} lane(s).")


def _render_stop_config(first_frame):
    st.markdown("### Stop Line Configuration")
    if st_canvas is None:
        st.error("`streamlit-drawable-canvas` is missing. Install it to enable in-app drawing.")
        return

    locked = bool(st.session_state.get("manual_stop_locked", False))
    running = bool(st.session_state.get("manual_running", False))
    analysis_started = bool(st.session_state.get("manual_analysis_started", False))
    if locked:
        st.success("Stop line saved and locked.")
    if running or analysis_started:
        st.info("Stop line editing is allowed only before analysis starts.")

    display_frame = st.session_state.get("manual_display_frame")
    saved_line = st.session_state.get("manual_stop_line")
    preview_bg = _draw_stop_preview(display_frame, saved_line, [])

    if locked or running or analysis_started:
        st.image(cv2.cvtColor(preview_bg, cv2.COLOR_BGR2RGB), channels="RGB", caption="Stop Line Configuration", width="stretch")
        return

    st.caption("Click exactly 2 points for stop line.")
    bg = _bgr_to_pil(preview_bg)
    canvas = st_canvas(
        fill_color="rgba(255,0,0,0.0)",
        stroke_width=2,
        stroke_color="#ff2d2d",
        background_image=bg,
        update_streamlit=True,
        height=bg.height,
        width=bg.width,
        drawing_mode="point",
        point_display_radius=4,
        display_toolbar=False,
        key=f"manual_stop_point_canvas_{st.session_state.manual_stop_canvas_seed}_{st.session_state.get('manual_uploaded_token', 'none')}",
    )

    raw_points = _extract_points_from_canvas(canvas.json_data if canvas is not None else None)
    current_points = raw_points[:2]
    if len(raw_points) > 2:
        st.warning("Stop line supports exactly 2 points. Extra clicks are ignored.")

    st.caption(f"Current stop line points: {len(current_points)}/2")

    s1, s2, s3 = st.columns(3)
    if s1.button("Save Stop Line", key="stop_save_line_btn", width="stretch", disabled=len(current_points) != 2):
        p1 = _to_original_point(current_points[0], first_frame)
        p2 = _to_original_point(current_points[1], first_frame)
        st.session_state.manual_stop_line = (p1, p2)
        st.session_state.manual_stop_locked = True
        _save_config(st.session_state.manual_config_path, {"stop_line": [[p1[0], p1[1]], [p2[0], p2[1]]]})
        st.success("Stop line saved. Editing disabled.")

    if s2.button("Clear Current Stop Line", key="stop_clear_line_btn", width="stretch"):
        st.session_state.manual_stop_canvas_seed += 1
        st.rerun()

    if s3.button("Configure Stop Line", key="stop_config_mode_btn_inner", width="stretch"):
        st.session_state.manual_config_mode = "stop"


def _render_top_section():
    t_col, signal_col = st.columns([2, 1])
    with t_col:
        st.title("Traffic Intelligence System")
    with signal_col:
        st.markdown("**Signal Control**")
        r_col, g_col = st.columns(2)
        if r_col.button("RED", key="top_red_btn", width="stretch"):
            st.session_state.manual_signal_state = "RED"
        if g_col.button("GREEN", key="top_green_btn", width="stretch"):
            st.session_state.manual_signal_state = "GREEN"

    status = str(st.session_state.get("manual_signal_state", "RED")).upper()
    color = "#d62828" if status == "RED" else "#2a9d8f"
    st.markdown(
        f"""
        <div style="padding:12px 16px;border-radius:10px;background:{color};color:white;font-size:28px;font-weight:700;text-align:center;">
            Current Signal Status: {status}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show():
    _init_state()
    st.title("Manual Config + Analysis")

    uploaded = st.file_uploader("Upload pre-recorded traffic video", type=["mp4", "avi", "mov", "mkv"], key="manual_upload")
    if uploaded is not None:
        token = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("manual_uploaded_token") != token:
            path = save_uploaded_video(uploaded)
            first = get_first_frame(path)
            if first is not None:
                st.session_state.manual_video_path = path
                st.session_state.manual_first_frame = first
                st.session_state.manual_uploaded_token = token
                display, sx, sy = _prepare_display_frame(first)
                st.session_state.manual_display_frame = display
                st.session_state.manual_scale_x = sx
                st.session_state.manual_scale_y = sy
                st.session_state.manual_lanes = []
                st.session_state.manual_lane_locked = False
                st.session_state.manual_lane_stage = "idle"
                st.session_state.manual_stop_line = None
                st.session_state.manual_stop_locked = False
                st.session_state.manual_lane_canvas_seed += 1
                st.session_state.manual_stop_canvas_seed += 1
                st.session_state.manual_config_mode = "lane"
                st.session_state.manual_analysis_started = False
                st.success(f"Video loaded: {os.path.basename(path)}")

    first_frame = st.session_state.get("manual_first_frame")
    if first_frame is None:
        st.info("Upload a video to start.")
        return

    st.markdown("**Phase 4: Signal Control**")
    sig1, sig2, sig3 = st.columns([1, 1, 2])
    if sig1.button("RED", key="signal_red_btn", width="stretch"):
        st.session_state.manual_signal_state = "RED"
    if sig2.button("GREEN", key="signal_green_btn", width="stretch"):
        st.session_state.manual_signal_state = "GREEN"
    sig3.info(f"Current Signal State: {st.session_state.manual_signal_state}")

    st.markdown("**Phase 5: Run Analysis**")
    run_col, stop_col = st.columns(2)
    if run_col.button("Start Analysis", key="analysis_start_btn", width="stretch", disabled=st.session_state.manual_running):
        _start_analysis()
    if stop_col.button("Stop Analysis", key="analysis_stop_btn", width="stretch", disabled=not st.session_state.manual_running):
        st.session_state.manual_running = False
        _release_capture()
        _release_writer()
        st.session_state.manual_export_ready = True

    analysis_started = bool(st.session_state.get("manual_analysis_started", False))
    if not analysis_started:
        st.markdown("**Phase 1: Video Upload**")

        st.markdown("**Phase 2 / 3: Configuration**")
        cfg_col1, cfg_col2 = st.columns(2)
        if cfg_col1.button(
            "Configure Lanes",
            key="cfg_lanes_btn_main",
            width="stretch",
            disabled=st.session_state.manual_running or st.session_state.manual_analysis_started,
        ):
            st.session_state.manual_config_mode = "lane"
        if cfg_col2.button(
            "Configure Stop Line",
            key="cfg_stop_btn_main",
            width="stretch",
            disabled=st.session_state.manual_running or st.session_state.manual_analysis_started,
        ):
            st.session_state.manual_config_mode = "stop"

        mode = st.session_state.get("manual_config_mode", "lane")
        if mode == "stop":
            _render_stop_config(first_frame)
        else:
            _render_lane_config(first_frame)

    if st.session_state.manual_running:
        steps = int(st.session_state.get("manual_frames_per_step", 2))
        for _ in range(max(1, steps)):
            if not st.session_state.manual_running:
                break
            _process_next_frame()

    frame = st.session_state.get("manual_latest_frame")
    lane_stats = st.session_state.get("manual_lane_stats", [])
    avg_density = float(np.mean([float(r["density"]) for r in lane_stats])) if lane_stats else 0.0

    st.markdown("**Live Monitoring**")
    live_col, metric_col = st.columns([7, 3])
    with live_col:
        video_placeholder = st.empty()
        if frame is None:
            frame = first_frame
        video_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            caption="Live Annotated Video",
            width="stretch",
            output_format="JPEG",
        )

    with metric_col:
        st.metric("Total Vehicle Count", int(st.session_state.manual_total_vehicle_count))
        st.metric("Total Violations", int(len(st.session_state.manual_violations)))
        st.metric("Average Density", f"{avg_density:.3f}")
        st.metric("Current Frame Number", int(st.session_state.manual_frame_idx))
        if st.session_state.get("manual_frame_shape") is not None:
            st.caption(f"Frame shape: {st.session_state.manual_frame_shape}")

    st.markdown("**Lane-wise Queue / Density / Status**")
    if lane_stats:
        lane_rows = []
        for row in lane_stats:
            lane_rows.append(
                {
                    "Lane ID": int(row["lane_index"]) + 1,
                    "Vehicle Count": int(row["vehicle_count"]),
                    "Density": round(float(row["density"]), 3),
                    "Status": str(row["queue_status"]),
                }
            )
        st.dataframe(pd.DataFrame(lane_rows), width="stretch", hide_index=True)
    else:
        st.info("No lane stats yet.")

    st.markdown("**Vehicle Class Distribution**")
    class_dist = st.session_state.get("manual_class_dist", {})
    dist_df = pd.DataFrame([{"Class": k, "Count": int(v)} for k, v in class_dist.items() if int(v) > 0])
    if dist_df.empty:
        st.info("No class distribution yet.")
    else:
        st.bar_chart(dist_df.set_index("Class"))

    st.markdown("**Violations List (Vehicle ID + Timestamp)**")
    violations = st.session_state.get("manual_violations", [])
    if violations:
        vio_df = pd.DataFrame(
            [
                {
                    "Vehicle ID": int(v["track_id"]),
                    "Lane": int(v.get("lane", -1)),
                    "Timestamp": str(v["timestamp"]),
                    "Violation Type": str(v.get("type", v.get("violation_type", "Violation"))),
                }
                for v in violations
            ]
        )
        st.dataframe(vio_df, width="stretch", hide_index=True)
    else:
        st.info("No violations detected.")

    st.subheader("Video Export")
    export_path = st.session_state.get("manual_export_path")
    export_ready = bool(st.session_state.get("manual_export_ready"))
    if export_ready and export_path and os.path.exists(export_path):
        with open(export_path, "rb") as f:
            st.download_button(
                "Export Annotated Video",
                data=f.read(),
                file_name=os.path.basename(export_path),
                mime="video/mp4",
                width="stretch",
            )
    else:
        st.caption("Export Annotated Video will be enabled after analysis ends/stops.")

    if st.session_state.manual_running:
        time.sleep(0.005)
        st.rerun()
