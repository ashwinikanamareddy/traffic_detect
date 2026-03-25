from typing import Dict, Iterable, List, Sequence, Tuple


Point = Tuple[float, float]
Polygon = List[Point]


def polygon_area(points: Sequence[Point]) -> float:
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if not polygon or len(polygon) < 3:
        return False

    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / max(1e-9, (y2 - y1)) + x1
        )
        if intersects:
            inside = not inside

    return inside


def default_queue_polygon(frame_width: int, frame_height: int) -> Polygon:
    y_top = int(frame_height * 0.58)
    y_bottom = int(frame_height * 0.98)
    x_margin = int(frame_width * 0.05)
    return [
        (float(x_margin), float(y_top)),
        (float(frame_width - x_margin), float(y_top)),
        (float(frame_width - x_margin), float(y_bottom)),
        (float(x_margin), float(y_bottom)),
    ]


class QueueAnalyzer:
    def __init__(self, queue_polygon: Iterable[Point]):
        self.queue_polygon: Polygon = [(float(x), float(y)) for x, y in queue_polygon]
        self._area = polygon_area(self.queue_polygon)

    @property
    def area(self) -> float:
        return self._area

    def update_polygon(self, queue_polygon: Iterable[Point]):
        self.queue_polygon = [(float(x), float(y)) for x, y in queue_polygon]
        self._area = polygon_area(self.queue_polygon)

    def is_vehicle_in_queue(self, center_x: float, center_y: float) -> bool:
        return point_in_polygon((float(center_x), float(center_y)), self.queue_polygon)

    def compute(self, tracks: List[Dict]):
        inside_ids = []
        for track in tracks:
            if self.is_vehicle_in_queue(track.get("center_x", 0.0), track.get("center_y", 0.0)):
                inside_ids.append(int(track.get("track_id", -1)))

        queue_length = len(inside_ids)
        queue_density = (queue_length / self._area) if self._area > 0 else 0.0

        return {
            "queue_length": int(queue_length),
            "queue_density": float(queue_density),
            "queue_area": float(self._area),
            "queue_track_ids": inside_ids,
        }
