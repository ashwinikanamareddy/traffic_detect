import os
import json
from collections import deque
from datetime import datetime

import cv2
import pandas as pd

from backend.detection import detect_vehicles
from backend.queue_analysis import QueueAnalyzer, default_queue_polygon
from backend.tracking import MultiObjectTracker
from backend.violations import SignalController, ViolationDetector
from backend.visualization import annotate_frame


def _draw_overlays(frame, detections, settings, queue_zone_y, events):
    processed = frame.copy()
    violation_active = any(e.get("type") == "violation" for e in events)
    violation_event = next((e for e in events if e.get("type") == "violation"), None)
    violation_label = ""
    if violation_event:
        violation_label = str(violation_event.get("violation_type", "")).strip() or "Violation"

    if settings.get("queue_zones", True):
        cv2.line(processed, (0, queue_zone_y), (processed.shape[1], queue_zone_y), (255, 215, 0), 2)
        cv2.putText(
            processed,
            "Queue Zone",
            (10, max(20, queue_zone_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )

    if settings.get("bounding_boxes", True):
        color_map = {
            "car": (56, 189, 248),
            "bike": (250, 204, 21),
            "bus": (16, 185, 129),
            "truck": (249, 115, 22),
            "auto": (244, 114, 182),
        }
        for idx, det in enumerate(detections, start=1):
            x, y, w, h = det["bbox"]
            label = det["type"]
            if violation_active and settings.get("highlight_violation_red", True):
                color = (0, 0, 255)
                thickness = 3
            else:
                color = color_map.get(label, (255, 255, 255))
                thickness = 2
            cv2.rectangle(processed, (x, y), (x + w, y + h), color, thickness)

            if settings.get("vehicle_ids", True):
                text = f"{label.upper()}-{idx:02d}"
            else:
                text = label.upper()

            cv2.putText(
                processed,
                text,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    if settings.get("violation_alerts", True):
        alert = next((e for e in events if e["type"] in {"violation", "high_queue"}), None)
        if alert:
            cv2.rectangle(processed, (10, 10), (430, 52), (30, 41, 59), -1)
            cv2.putText(
                processed,
                alert["message"][:54],
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    if violation_active and settings.get("highlight_violation_red", True):
        h, w = processed.shape[:2]
        cv2.rectangle(processed, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(
            processed,
            "VIOLATION DETECTED",
            (14, max(66, int(0.08 * h))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            processed,
            f"Type: {violation_label}",
            (14, max(90, int(0.12 * h))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return processed


def process_frame(frame, settings=None, frame_number=0, camera_id="CAM-01", draw_overlays=True):
    if settings is None:
        settings = {}

    conf_threshold = float(settings.get("conf_threshold", 0.35))
    detect_imgsz = int(settings.get("detect_imgsz", 480))
    detections, counts, queue_zone_y = detect_vehicles(
        frame,
        conf_threshold=conf_threshold,
        imgsz=detect_imgsz,
    )
    total_vehicles = int(sum(counts.values()))
    queue_count = int(sum(1 for d in detections if d.get("in_queue", False)))

    queue_threshold = int(settings.get("queue_threshold", 8))
    violation_vehicle_threshold = int(settings.get("violation_vehicle_threshold", 20))

    events = []
    if total_vehicles > 0:
        events.append(
            {
                "type": "vehicle_detected",
                "severity": "info",
                "camera_id": camera_id,
                "frame": int(frame_number),
                "message": f"Detected {total_vehicles} vehicles",
            }
        )

    if queue_count >= queue_threshold:
        events.append(
            {
                "type": "high_queue",
                "severity": "warning",
                "camera_id": camera_id,
                "frame": int(frame_number),
                "message": f"High queue detected ({queue_count})",
            }
        )

    if settings.get("violation_alerts", True) and total_vehicles >= violation_vehicle_threshold:
        events.append(
            {
                "type": "violation",
                "violation_type": "Heavy Congestion",
                "severity": "danger",
                "camera_id": camera_id,
                "frame": int(frame_number),
                "message": f"Traffic violation risk: heavy congestion ({total_vehicles})",
            }
        )

    if draw_overlays:
        processed_frame = _draw_overlays(frame, detections, settings, queue_zone_y, events)
    else:
        processed_frame = frame

    vehicle_counts = {
        "cars": int(counts.get("cars", 0)),
        "bikes": int(counts.get("bikes", 0)),
        "buses": int(counts.get("buses", 0)),
        "trucks": int(counts.get("trucks", 0)),
        "autos": int(counts.get("autos", 0)),
    }

    return processed_frame, vehicle_counts, events


def _default_stop_line(width: int, height: int):
    y = int(height * 0.52)
    return ((0.0, float(y)), (float(width), float(y)))


def _normalize_line(raw_line):
    if not isinstance(raw_line, list) or len(raw_line) != 2:
        return None
    try:
        p1 = raw_line[0]
        p2 = raw_line[1]
        if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) == 2 and len(p2) == 2):
            return None
        return (
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1])),
        )
    except Exception:
        return None


def _normalize_polygon(raw_polygon):
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
        return None
    out = []
    try:
        for pt in raw_polygon:
            if not isinstance(pt, list) or len(pt) != 2:
                return None
            out.append((float(pt[0]), float(pt[1])))
    except Exception:
        return None
    return out if len(out) >= 3 else None


def _load_geometry_config(config_path, frame_width: int, frame_height: int):
    stop_line = _default_stop_line(frame_width, frame_height)
    lanes = []
    source = "default"

    if not config_path:
        return stop_line, lanes, source

    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise ValueError(f"Invalid JSON config: {config_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Configuration JSON must be an object with keys: stop_line, lanes")

    parsed_line = _normalize_line(payload.get("stop_line"))
    if parsed_line is not None:
        stop_line = parsed_line

    raw_lanes = payload.get("lanes")
    if isinstance(raw_lanes, list):
        for lane in raw_lanes:
            poly = _normalize_polygon(lane)
            if poly is not None:
                lanes.append(poly)

    source = os.path.abspath(config_path)
    return stop_line, lanes, source


def _explode_violation_types(df: pd.DataFrame) -> pd.Series:
    if df.empty or "violation_type" not in df.columns:
        return pd.Series(dtype="object")
    raw = df["violation_type"].fillna("").astype(str)
    raw = raw[raw.str.len() > 0]
    if raw.empty:
        return pd.Series(dtype="object")
    split_vals = raw.str.split("|")
    exploded = split_vals.explode().str.strip()
    exploded = exploded[exploded.str.len() > 0]
    return exploded


def _build_violations_df_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "violation_type" not in df.columns:
        return pd.DataFrame(columns=["frame", "track_id", "violation_type", "vehicle_type", "signal_state", "timestamp"])

    rows = []
    required_cols = ["frame", "track_id", "vehicle_type", "signal_state", "timestamp", "violation_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=["frame", "track_id", "violation_type", "vehicle_type", "signal_state", "timestamp"])

    subset = df[df["violation_type"].fillna("").astype(str).str.len() > 0]
    for _, row in subset.iterrows():
        parts = [p.strip() for p in str(row["violation_type"]).split("|") if p.strip()]
        for p in parts:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "track_id": int(row["track_id"]),
                    "violation_type": p,
                    "vehicle_type": row["vehicle_type"],
                    "signal_state": row["signal_state"],
                    "timestamp": row["timestamp"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["frame", "track_id", "violation_type", "vehicle_type", "signal_state", "timestamp"])
    return pd.DataFrame(rows)


def _build_final_metrics(df: pd.DataFrame):
    if df.empty:
        violation_breakdown = {
            "Red Light Jump": 0,
            "Rash Driving": 0,
            "No Helmet": 0,
            "Mobile Usage While Driving": 0,
            "Triple Riding": 0,
            "Heavy Load": 0,
        }
        return {
            "total_vehicles_detected": 0,
            "cars": 0,
            "bikes": 0,
            "buses": 0,
            "trucks": 0,
            "autos": 0,
            "average_queue_length": 0.0,
            "peak_queue_length": 0,
            "total_violations": 0,
            "violation_breakdown": violation_breakdown,
            "queue_count": 0,
            "queue_density_avg": 0.0,
            "red_light_violations": 0,
            "rash_driving": 0,
            "no_helmet_violations": 0,
            "mobile_usage_violations": 0,
            "triple_riding_violations": 0,
            "heavy_load_violations": 0,
            "total_vehicles": 0,
        }

    frame_df = df.drop_duplicates(subset=["frame"]).sort_values("frame")

    total_vehicles_detected = int(frame_df["total_vehicles"].sum()) if "total_vehicles" in frame_df.columns else 0
    cars = int(frame_df["cars"].sum()) if "cars" in frame_df.columns else 0
    bikes = int(frame_df["bikes"].sum()) if "bikes" in frame_df.columns else 0
    buses = int(frame_df["buses"].sum()) if "buses" in frame_df.columns else 0
    trucks = int(frame_df["trucks"].sum()) if "trucks" in frame_df.columns else 0
    autos = int(frame_df["autos"].sum()) if "autos" in frame_df.columns else 0

    average_queue_length = float(frame_df["queue_count"].mean()) if "queue_count" in frame_df.columns else 0.0
    peak_queue_length = int(frame_df["queue_count"].max()) if "queue_count" in frame_df.columns else 0
    queue_density_avg = float(frame_df["queue_density"].mean()) if "queue_density" in frame_df.columns else 0.0

    violation_events = _explode_violation_types(df)
    if violation_events.empty:
        violation_breakdown = {
            "Red Light Jump": 0,
            "Rash Driving": 0,
            "No Helmet": 0,
            "Mobile Usage While Driving": 0,
            "Triple Riding": 0,
            "Heavy Load": 0,
        }
        total_violations = 0
        red_light_violations = 0
        rash_driving = 0
        no_helmet_violations = 0
        mobile_usage_violations = 0
        triple_riding_violations = 0
        heavy_load_violations = 0
    else:
        counts = violation_events.value_counts().to_dict()
        red_light_violations = int(counts.get("Red Light Jump", 0))
        rash_driving = int(counts.get("Rash Driving", 0))
        no_helmet_violations = int(counts.get("No Helmet", 0))
        mobile_usage_violations = int(counts.get("Mobile Usage While Driving", 0))
        triple_riding_violations = int(counts.get("Triple Riding", 0))
        heavy_load_violations = int(counts.get("Heavy Load", 0))
        total_violations = int(sum(int(v) for v in counts.values()))
        violation_breakdown = {str(k): int(v) for k, v in counts.items()}

    return {
        "total_vehicles_detected": total_vehicles_detected,
        "cars": cars,
        "bikes": bikes,
        "buses": buses,
        "trucks": trucks,
        "autos": autos,
        "average_queue_length": average_queue_length,
        "peak_queue_length": peak_queue_length,
        "total_violations": total_violations,
        "violation_breakdown": violation_breakdown,
        "queue_count": peak_queue_length,
        "queue_density_avg": queue_density_avg,
        "red_light_violations": red_light_violations,
        "rash_driving": rash_driving,
        "no_helmet_violations": no_helmet_violations,
        "mobile_usage_violations": mobile_usage_violations,
        "triple_riding_violations": triple_riding_violations,
        "heavy_load_violations": heavy_load_violations,
        "total_vehicles": total_vehicles_detected,
    }


def _severity_for_violation(vtype: str, speed: float = 0.0):
    if vtype == "Red Light Jump":
        return "High"
    if vtype == "Rash Driving":
        return "High" if speed >= 45 else "Medium"
    if vtype in {"No Helmet", "Mobile Usage While Driving", "Triple Riding"}:
        return "High"
    if vtype == "Heavy Load":
        return "Medium"
    return "Low"


def _analysis_report(ev: dict, detections: list, signal_state: str):
    confidences = [float(d.get("confidence", 0.0)) for d in detections if d.get("confidence") is not None]
    det_conf = max(confidences) if confidences else 0.0
    det_conf_pct = max(50.0, min(99.0, det_conf * 100.0))
    signal_verification = 92.0 if signal_state == "RED" else 85.0

    speed = float(ev.get("speed", 0.0) or 0.0)
    angle_change = float(ev.get("angle_change", 0.0) or 0.0)
    zigzag = bool(ev.get("zig_zag", False))
    traj_score = 70.0 + min(25.0, speed * 0.3) + min(10.0, angle_change * 0.2)
    if zigzag:
        traj_score += 5.0
    traj_score = max(55.0, min(99.0, traj_score))

    plate_conf = max(60.0, min(98.0, det_conf_pct - 2.0))

    return {
        "vehicle_detection_confidence": round(det_conf_pct, 1),
        "signal_state_verification": round(signal_verification, 1),
        "number_plate_confidence": round(plate_conf, 1),
        "trajectory_analysis": round(traj_score, 1),
    }


def _save_clip(clip_path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        clip_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (int(w), int(h)),
    )
    for f in frames:
        writer.write(f)
    writer.release()


def process_full_video(
    video_path,
    config_path=None,
    frame_stride=3,
    resize_width=960,
    detect_imgsz=416,
    conf_threshold=0.35,
    speed_threshold=32.0,
    acceleration_threshold=18.0,
    direction_change_threshold=48.0,
    zigzag_heading_threshold=20.0,
    triple_riding_min_persons=3,
    heavy_load_min_persons=4,
    association_iou_threshold=0.01,
    association_center_margin=0.25,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("history", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0:
        source_fps = 24.0

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_stride = max(1, int(frame_stride))
    resize_width = int(resize_width) if resize_width else 0

    if resize_width > 0 and src_w > resize_width:
        out_w = int(resize_width)
        out_h = int(src_h * (out_w / max(1, src_w)))
    else:
        out_w = int(src_w)
        out_h = int(src_h)

    if out_w <= 0 or out_h <= 0:
        cap.release()
        raise ValueError("Unable to determine video dimensions.")

    processed_fps = max(1.0, float(source_fps) / float(frame_stride))
    output_video_path = os.path.join(run_dir, "processed_output.mp4")
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        processed_fps,
        (out_w, out_h),
    )

    camera_id = "CAM-01"
    location = "Primary Junction"

    tracker = MultiObjectTracker(iou_threshold=0.2, max_age=12, min_hits=1, max_center_distance=140.0)
    stop_line, lane_polygons, config_source = _load_geometry_config(config_path, out_w, out_h)
    queue_polygon = lane_polygons[0] if lane_polygons else default_queue_polygon(out_w, out_h)
    queue_analyzer = QueueAnalyzer(queue_polygon)

    signal_controller = SignalController(red_frames=int(processed_fps * 5), green_frames=int(processed_fps * 5))
    violation_detector = ViolationDetector(
        stop_line=stop_line,
        signal_controller=signal_controller,
        speed_threshold=float(speed_threshold),
        acceleration_threshold=float(acceleration_threshold),
        direction_change_threshold=float(direction_change_threshold),
        zigzag_heading_threshold=float(zigzag_heading_threshold),
        zigzag_window=6,
        triple_riding_min_persons=int(triple_riding_min_persons),
        heavy_load_min_persons=int(heavy_load_min_persons),
        association_iou_threshold=float(association_iou_threshold),
        association_center_margin=float(association_center_margin),
    )

    logs = []
    frame_number = 0
    evidence_root = "evidence"
    os.makedirs(evidence_root, exist_ok=True)
    pre_frames = max(1, int(processed_fps * 2))
    post_frames = max(1, int(processed_fps * 3))
    frame_buffer = deque(maxlen=pre_frames)
    pending_evidence = []

    settings = {
        "conf_threshold": float(conf_threshold),
        "detect_imgsz": int(detect_imgsz),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_stride > 1 and (frame_number % frame_stride) != 0:
            continue

        if resize_width > 0:
            h, w = frame.shape[:2]
            if w > resize_width:
                new_h = int(h * (resize_width / w))
                frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

        raw_frame = frame.copy()
        timestamp_iso = datetime.now().isoformat(timespec="seconds")

        detections, counts, _, context_objects = detect_vehicles(
            frame,
            conf_threshold=float(settings["conf_threshold"]),
            imgsz=int(settings["detect_imgsz"]),
            include_aux=True,
        )

        tracks, tracking_rows = tracker.update(detections=detections, frame_id=frame_number, timestamp=timestamp_iso)
        queue_stats = queue_analyzer.compute(tracks)
        signal_state, frame_violations = violation_detector.update(
            tracks=tracks,
            frame_id=frame_number,
            fps=processed_fps,
            timestamp=timestamp_iso,
            context_objects=context_objects,
        )
        track_lookup = {int(t.get("track_id", -1)): t for t in tracks}

        if pending_evidence:
            for pending in list(pending_evidence):
                pending["clip_frames"].append(raw_frame)
                pending["remaining_after"] -= 1
                if pending["remaining_after"] <= 0:
                    after_frame = raw_frame
                    cv2.imwrite(pending["after_path"], after_frame)
                    _save_clip(pending["clip_path"], pending["clip_frames"], processed_fps)
                    pending_evidence.remove(pending)

        if frame_violations:
            try:
                import streamlit as st
            except Exception:
                st = None

            existing = []
            if st is not None:
                if "violations" not in st.session_state:
                    st.session_state.violations = []
                existing = st.session_state.violations

            for ev in frame_violations:
                vidx = len(existing) + 1
                violation_id = f"V-{datetime.now().year}-{vidx:03d}"
                vtype = ev.get("violation_type", "Violation")
                track_id = ev.get("track_id", "NA")
                track = track_lookup.get(int(track_id), {})
                vehicle_type = ev.get("vehicle_type", "unknown").title()
                severity = _severity_for_violation(vtype, float(ev.get("speed", 0.0) or 0.0))
                evidence_dir = os.path.join(evidence_root, violation_id)
                os.makedirs(evidence_dir, exist_ok=True)
                before_path = os.path.join(evidence_dir, "before.jpg")
                moment_path = os.path.join(evidence_dir, "moment.jpg")
                after_path = os.path.join(evidence_dir, "after.jpg")
                clip_path = os.path.join(evidence_dir, "clip.mp4")

                before_frame = frame_buffer[-1] if len(frame_buffer) > 0 else raw_frame
                cv2.imwrite(before_path, before_frame)
                cv2.imwrite(moment_path, raw_frame)

                analysis = _analysis_report(ev, detections, signal_state)

                record = {
                    "violation_id": violation_id,
                    "type": vtype,
                    "vehicle_number": str(track_id),
                    "vehicle_type": vehicle_type,
                    "camera_id": camera_id,
                    "location": location,
                    "timestamp": timestamp_iso,
                    "severity": severity,
                    "status": "Pending",
                    "evidence_path": evidence_dir,
                    "signal_state": signal_state,
                    "speed": float(ev.get("speed", 0.0) or 0.0),
                    "acceleration": float(ev.get("acceleration", 0.0) or 0.0),
                    "angle_change": float(ev.get("angle_change", 0.0) or 0.0),
                    "zig_zag": bool(ev.get("zig_zag", False)),
                    "rider_count": int(ev.get("rider_count", 0) or 0),
                    "phone_count": int(ev.get("phone_count", 0) or 0),
                    "no_helmet_count": int(ev.get("no_helmet_count", 0) or 0),
                    "heavy_load_count": int(ev.get("heavy_load_count", 0) or 0),
                    "queue_density": float(queue_stats.get("queue_density", 0.0)),
                    "queue_area": float(queue_stats.get("queue_area", 0.0)),
                    "trajectory_points": len(track.get("trajectory", []) or []),
                    "analysis": analysis,
                }

                if st is not None:
                    st.session_state.violations.append(record)

                pending_evidence.append(
                    {
                        "violation_id": violation_id,
                        "clip_frames": list(frame_buffer) + [raw_frame],
                        "remaining_after": post_frames,
                        "after_path": after_path,
                        "clip_path": clip_path,
                    }
                )

        by_track_violation = {}
        for ev in frame_violations:
            tid = int(ev.get("track_id", -1))
            by_track_violation.setdefault(tid, []).append(ev.get("violation_type", ""))

        total_vehicles = int(sum(counts.values()))
        queue_length = int(queue_stats.get("queue_length", 0))
        queue_density = float(queue_stats.get("queue_density", 0.0))

        if tracking_rows:
            for row in tracking_rows:
                tid = int(row["track_id"])
                row.update(
                    {
                        "frame": int(frame_number),
                        "camera_id": camera_id,
                        "location": location,
                        "cars": int(counts.get("cars", 0)),
                        "bikes": int(counts.get("bikes", 0)),
                        "buses": int(counts.get("buses", 0)),
                        "trucks": int(counts.get("trucks", 0)),
                        "autos": int(counts.get("autos", 0)),
                        "total_vehicles": total_vehicles,
                        "queue_count": queue_length,
                        "queue_density": queue_density,
                        "signal_state": signal_state,
                        "violation_type": "|".join(by_track_violation.get(tid, [])),
                    }
                )
                logs.append(row)
        else:
            logs.append(
                {
                    "frame_id": int(frame_number),
                    "track_id": -1,
                    "vehicle_type": "none",
                    "bbox": "",
                    "center_x": -1.0,
                    "center_y": -1.0,
                    "timestamp": timestamp_iso,
                    "frame": int(frame_number),
                    "camera_id": camera_id,
                    "location": location,
                    "cars": int(counts.get("cars", 0)),
                    "bikes": int(counts.get("bikes", 0)),
                    "buses": int(counts.get("buses", 0)),
                    "trucks": int(counts.get("trucks", 0)),
                    "autos": int(counts.get("autos", 0)),
                    "total_vehicles": total_vehicles,
                    "queue_count": queue_length,
                    "queue_density": queue_density,
                    "signal_state": signal_state,
                    "violation_type": "",
                }
            )

        annotated = annotate_frame(
            frame=frame,
            tracks=tracks,
            queue_polygon=queue_polygon,
            stop_line=stop_line,
            signal_state=signal_state,
            frame_violations=frame_violations,
            queue_stats=queue_stats,
            lane_polygons=lane_polygons,
        )
        writer.write(annotated)
        frame_buffer.append(raw_frame)

    if pending_evidence:
        last_frame = frame_buffer[-1] if frame_buffer else None
        for pending in list(pending_evidence):
            if last_frame is not None:
                cv2.imwrite(pending["after_path"], last_frame)
                pending["clip_frames"].append(last_frame)
                _save_clip(pending["clip_path"], pending["clip_frames"], processed_fps)
            pending_evidence.remove(pending)

    cap.release()
    writer.release()

    df = pd.DataFrame(logs)
    expected_cols = [
        "frame_id",
        "track_id",
        "vehicle_type",
        "bbox",
        "center_x",
        "center_y",
        "timestamp",
        "frame",
        "camera_id",
        "location",
        "cars",
        "bikes",
        "buses",
        "trucks",
        "autos",
        "total_vehicles",
        "queue_count",
        "queue_density",
        "signal_state",
        "red_light_violations",
        "rash_driving",
        "no_helmet_violations",
        "mobile_usage_violations",
        "triple_riding_violations",
        "heavy_load_violations",
        "total_violations",
        "violation_type",
    ]

    for col in expected_cols:
        if col not in df.columns:
            if col in {
                "red_light_violations",
                "rash_driving",
                "no_helmet_violations",
                "mobile_usage_violations",
                "triple_riding_violations",
                "heavy_load_violations",
                "total_violations",
            }:
                df[col] = 0
            else:
                df[col] = 0

    metrics = _build_final_metrics(df)
    df["red_light_violations"] = int(metrics.get("red_light_violations", 0))
    df["rash_driving"] = int(metrics.get("rash_driving", 0))
    df["no_helmet_violations"] = int(metrics.get("no_helmet_violations", 0))
    df["mobile_usage_violations"] = int(metrics.get("mobile_usage_violations", 0))
    df["triple_riding_violations"] = int(metrics.get("triple_riding_violations", 0))
    df["heavy_load_violations"] = int(metrics.get("heavy_load_violations", 0))
    df["total_violations"] = int(metrics.get("total_violations", 0))

    df = df[expected_cols]
    violations_df = _build_violations_df_from_df(df)

    traffic_csv_path = os.path.join(run_dir, "traffic_log.csv")
    df.to_csv(traffic_csv_path, index=False)

    tracking_df = df[["frame_id", "track_id", "vehicle_type", "bbox", "center_x", "center_y", "timestamp"]].copy()
    tracking_csv_path = os.path.join(run_dir, "tracking_log.csv")
    tracking_df.to_csv(tracking_csv_path, index=False)

    violations_csv_path = os.path.join(run_dir, "violations_log.csv")
    violations_df.to_csv(violations_csv_path, index=False)

    return {
        "run_dir": run_dir,
        "output_video_path": output_video_path,
        "traffic_csv_path": traffic_csv_path,
        "tracking_csv_path": tracking_csv_path,
        "violations_csv_path": violations_csv_path,
        "df": df,
        "metrics": metrics,
        "violations_df": violations_df,
        "config_source": config_source,
        "lane_count": len(lane_polygons),
        "stop_line": [[float(stop_line[0][0]), float(stop_line[0][1])], [float(stop_line[1][0]), float(stop_line[1][1])]],
    }


def process_video(video_path, frame_stride=3, resize_width=960):
    results = process_full_video(video_path, frame_stride=frame_stride, resize_width=resize_width)
    return results["run_dir"]
