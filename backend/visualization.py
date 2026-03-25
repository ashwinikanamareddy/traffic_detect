from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _as_np_points(points):
    return np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)


def annotate_frame(
    frame,
    tracks: List[Dict],
    queue_polygon,
    stop_line: Tuple[Tuple[float, float], Tuple[float, float]],
    signal_state: str,
    frame_violations: List[Dict],
    queue_stats: Dict,
    lane_polygons: Optional[List[List[Tuple[float, float]]]] = None,
):
    canvas = frame.copy()

    lane_colors = [
        (255, 165, 0),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 128),
    ]

    if lane_polygons:
        for idx, lane in enumerate(lane_polygons):
            lane_pts = _as_np_points(lane)
            if len(lane_pts) < 3:
                continue
            color = lane_colors[idx % len(lane_colors)]
            cv2.polylines(canvas, [lane_pts], True, color, 2)
            centroid = np.mean(lane_pts.reshape(-1, 2), axis=0)
            cv2.putText(
                canvas,
                f"Lane {idx + 1}",
                (int(centroid[0]), int(centroid[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    # Queue zone overlay
    poly = _as_np_points(queue_polygon)
    if len(poly) >= 3:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], (37, 99, 235))
        canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)
        cv2.polylines(canvas, [poly], True, (59, 130, 246), 2)

    # Stop line + signal state
    (x1, y1), (x2, y2) = stop_line
    sig_color = (0, 0, 255) if signal_state == "RED" else (0, 200, 0)
    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), sig_color, 3)
    cv2.putText(
        canvas,
        f"Signal: {signal_state}",
        (int(x1) + 8, max(24, int(y1) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        sig_color,
        2,
        cv2.LINE_AA,
    )

    violating_ids = {int(v.get("track_id", -1)) for v in frame_violations}

    type_colors = {
        "car": (56, 189, 248),
        "bike": (250, 204, 21),
        "bus": (16, 185, 129),
        "truck": (249, 115, 22),
    }

    for track in tracks:
        track_id = int(track["track_id"])
        x, y, w, h = track["bbox"]
        vehicle_type = track.get("vehicle_type", "vehicle")

        color = (0, 0, 255) if track_id in violating_ids else type_colors.get(vehicle_type, (255, 255, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        label = f"ID {track_id} | {vehicle_type.upper()}"
        cv2.putText(
            canvas,
            label,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    q_len = int(queue_stats.get("queue_length", 0))
    q_den = float(queue_stats.get("queue_density", 0.0))
    cv2.rectangle(canvas, (10, 10), (360, 60), (15, 23, 42), -1)
    cv2.putText(
        canvas,
        f"Queue Length: {q_len} | Density: {q_den:.6f}",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas
