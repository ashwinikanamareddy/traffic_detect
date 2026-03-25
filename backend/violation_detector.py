from typing import Dict, List, Sequence, Tuple


Point = Tuple[float, float]
Line = Tuple[Point, Point]


def _line_side(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class RedLightViolationDetector:
    def __init__(self, stop_line: Line):
        self.stop_line = stop_line
        self.prev_side: Dict[int, float] = {}
        self.violated_ids = set()

    def update(self, tracks: List[Dict], signal_state: str, timestamp: str) -> List[Dict]:
        events = []
        signal_is_red = str(signal_state).upper() == "RED"
        for track in tracks:
            tid = int(track.get("track_id", -1))
            center = (float(track.get("center_x", 0.0)), float(track.get("center_y", 0.0)))
            side = _line_side(center, self.stop_line)
            prev = self.prev_side.get(tid)
            crossed = prev is not None and ((prev > 0 >= side) or (prev < 0 <= side))
            if crossed and signal_is_red and tid not in self.violated_ids:
                self.violated_ids.add(tid)
                events.append(
                    {
                        "track_id": tid,
                        "timestamp": timestamp,
                        "type": "Red Light Jump",
                    }
                )
            self.prev_side[tid] = side
        return events
