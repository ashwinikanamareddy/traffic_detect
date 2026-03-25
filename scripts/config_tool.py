import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Polygon = List[Point]


class TrafficConfigTool:
    def __init__(self, frame: np.ndarray) -> None:
        self.base_frame = frame
        self.window_name = "Traffic Config Tool"

        self.stop_line: List[Point] = []
        self.lanes: List[Polygon] = []
        self.current_lane: Polygon = []

        self.mode = "STOP_LINE"
        self.status_msg: Optional[str] = "Click two points to draw STOP LINE."
        self.msg_frames_left = 0

        self.lane_colors = [
            (255, 165, 0),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 255, 128),
            (255, 128, 0),
            (128, 128, 255),
        ]

    def set_status(self, msg: str, frames: int = 120) -> None:
        self.status_msg = msg
        self.msg_frames_left = frames

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        pt = (x, y)
        if self.mode == "STOP_LINE":
            if len(self.stop_line) < 2:
                self.stop_line.append(pt)
                if len(self.stop_line) == 2:
                    self.mode = "LANES"
                    self.set_status("STOP LINE fixed. Draw lanes. Press 'n' for next lane, 'q' to finish.")
                else:
                    self.set_status("Select second STOP LINE point.")
            return

        self.current_lane.append(pt)
        self.set_status(f"Lane {len(self.lanes) + 1}: point {len(self.current_lane)} added.")

    def _draw_stop_line(self, canvas: np.ndarray) -> None:
        if len(self.stop_line) == 1:
            cv2.circle(canvas, self.stop_line[0], 5, (0, 0, 255), -1)
            return

        if len(self.stop_line) == 2:
            p1, p2 = self.stop_line
            cv2.line(canvas, p1, p2, (0, 0, 255), 3)
            cx = (p1[0] + p2[0]) // 2
            cy = (p1[1] + p2[1]) // 2
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

    def _draw_lanes(self, canvas: np.ndarray) -> None:
        for idx, lane in enumerate(self.lanes):
            color = self.lane_colors[idx % len(self.lane_colors)]
            pts = np.array(lane, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, color, 2)

            arr = np.array(lane, dtype=np.float32)
            cx, cy = np.mean(arr[:, 0]), np.mean(arr[:, 1])
            cv2.putText(
                canvas,
                f"Lane {idx + 1}",
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        if not self.current_lane:
            return

        color = self.lane_colors[len(self.lanes) % len(self.lane_colors)]
        for p in self.current_lane:
            cv2.circle(canvas, p, 4, color, -1)

        if len(self.current_lane) >= 2:
            pts = np.array(self.current_lane, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, color, 2)

    def _draw_instructions(self, canvas: np.ndarray) -> None:
        lines: List[str] = []
        if self.mode == "STOP_LINE":
            lines.append("Mode: STOP_LINE")
            lines.append("Left click: choose 2 points")
            lines.append("q: finish early | esc: exit without save")
        else:
            lines.append("Mode: LANES")
            lines.append("Left click: add polygon point")
            lines.append("n: finalize lane (>=3 points)")
            lines.append("q: finish and save | esc: exit without save")

        panel_h = 24 * (len(lines) + 2)
        cv2.rectangle(canvas, (8, 8), (720, panel_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (8, 8), (720, panel_h), (90, 90, 90), 1)

        y = 30
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 24

        if self.status_msg and self.msg_frames_left > 0:
            cv2.putText(
                canvas,
                self.status_msg,
                (16, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            self.msg_frames_left -= 1

    def finalize_current_lane(self) -> None:
        if len(self.current_lane) < 3:
            self.set_status("Need at least 3 points to finalize a lane polygon.")
            return
        self.lanes.append(self.current_lane.copy())
        self.current_lane.clear()
        self.set_status(f"Lane {len(self.lanes)} finalized.")

    def render(self) -> np.ndarray:
        canvas = self.base_frame.copy()
        self._draw_stop_line(canvas)
        self._draw_lanes(canvas)
        self._draw_instructions(canvas)
        return canvas

    def run(self) -> bool:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        save_and_exit = False
        while True:
            cv2.imshow(self.window_name, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                save_and_exit = False
                break

            if key == ord("n"):
                if self.mode != "LANES":
                    self.set_status("Complete STOP LINE first.")
                else:
                    self.finalize_current_lane()

            elif key == ord("q"):
                if self.mode == "LANES" and len(self.current_lane) >= 3:
                    self.finalize_current_lane()
                save_and_exit = True
                break

        cv2.destroyAllWindows()
        return save_and_exit

    def get_config(self) -> dict:
        stop_line_out = [[int(x), int(y)] for x, y in self.stop_line[:2]]
        lanes_out = [[[int(x), int(y)] for x, y in lane] for lane in self.lanes]
        return {"stop_line": stop_line_out, "lanes": lanes_out}


def load_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Unable to read first frame from: {video_path}")

    return frame


def save_json(data: dict, out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Configuration saved to: {out_file.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive configuration tool for stop line and lane polygons."
    )
    parser.add_argument("--video", required=True, help="Path to video file.")
    parser.add_argument(
        "--output",
        default="traffic_config.json",
        help="Output JSON path (default: traffic_config.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_first_frame(args.video)

    tool = TrafficConfigTool(frame)
    should_save = tool.run()

    if should_save:
        save_json(tool.get_config(), args.output)
    else:
        print("[INFO] Exited without saving.")


if __name__ == "__main__":
    main()
