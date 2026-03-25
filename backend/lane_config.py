import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Polygon = List[Point]


def _draw_lane_canvas(base_frame, lanes: List[Polygon], current_lane: Polygon):
    canvas = base_frame.copy()
    colors = [
        (255, 165, 0),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 128),
    ]

    for idx, lane in enumerate(lanes):
        color = colors[idx % len(colors)]
        pts = np.array(lane, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, color, 2)
        center = np.mean(np.array(lane, dtype=np.float32), axis=0)
        cv2.putText(
            canvas,
            f"Lane {idx + 1}",
            (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    color = colors[len(lanes) % len(colors)]
    for p in current_lane:
        cv2.circle(canvas, p, 4, color, -1)
    if len(current_lane) >= 2:
        pts = np.array(current_lane, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, color, 2)

    cv2.rectangle(canvas, (8, 8), (860, 92), (20, 20, 20), -1)
    cv2.rectangle(canvas, (8, 8), (860, 92), (90, 90, 90), 1)
    cv2.putText(
        canvas,
        "Lane Config: left click=add point | n=finish lane | q=save and exit | esc=cancel",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Lanes saved: {len(lanes)} | Current points: {len(current_lane)}",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def configure_lanes(frame) -> List[Polygon]:
    lanes: List[Polygon] = []
    current_lane: Polygon = []
    window_name = "Configure Lanes"

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            current_lane.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    save = False

    while True:
        cv2.imshow(window_name, _draw_lane_canvas(frame, lanes, current_lane))
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            save = False
            break
        if key == ord("n"):
            if len(current_lane) >= 3:
                lanes.append(current_lane.copy())
            current_lane.clear()
        if key == ord("q"):
            if len(current_lane) >= 3:
                lanes.append(current_lane.copy())
            save = True
            break

    cv2.destroyWindow(window_name)
    return lanes if save else []


def save_lanes_json(lanes: List[Polygon], json_path: str) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lanes": [[[int(x), int(y)] for x, y in lane] for lane in lanes]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def merge_lanes_into_config(lanes: List[Polygon], json_path: str) -> None:
    path = Path(json_path)
    data = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    data["lanes"] = [[[int(x), int(y)] for x, y in lane] for lane in lanes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
