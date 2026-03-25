import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

Point = Tuple[int, int]
Line = Tuple[Point, Point]


def _draw_stopline_canvas(base_frame, points: List[Point]):
    canvas = base_frame.copy()
    if len(points) == 1:
        cv2.circle(canvas, points[0], 5, (0, 0, 255), -1)
    if len(points) == 2:
        cv2.line(canvas, points[0], points[1], (0, 0, 255), 3)
        cx = (points[0][0] + points[1][0]) // 2
        cy = (points[0][1] + points[1][1]) // 2
        cv2.putText(
            canvas,
            "STOP LINE",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (8, 8), (860, 92), (20, 20, 20), -1)
    cv2.rectangle(canvas, (8, 8), (860, 92), (90, 90, 90), 1)
    cv2.putText(
        canvas,
        "Stop Line Config: left click=2 points | q=save and exit | esc=cancel",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Selected points: {len(points)}/2",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def configure_stop_line(frame) -> Optional[Line]:
    points: List[Point] = []
    window_name = "Configure Stop Line"

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((int(x), int(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    save = False
    while True:
        cv2.imshow(window_name, _draw_stopline_canvas(frame, points))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            save = False
            break
        if key == ord("q"):
            save = len(points) == 2
            break

    cv2.destroyWindow(window_name)
    if not save or len(points) != 2:
        return None
    return (points[0], points[1])


def merge_stopline_into_config(stop_line: Line, json_path: str) -> None:
    path = Path(json_path)
    data = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    data["stop_line"] = [[int(stop_line[0][0]), int(stop_line[0][1])], [int(stop_line[1][0]), int(stop_line[1][1])]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
