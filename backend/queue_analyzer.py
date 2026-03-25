from typing import Dict, List, Sequence, Tuple


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


def queue_status_from_density(density: float) -> str:
    if density <= 5.0:
        return "Low"
    if density <= 10.0:
        return "Medium"
    return "High"


def lane_wise_queue_stats(tracks: List[Dict], lanes: List[Polygon]) -> List[Dict]:
    results = []
    for idx, lane in enumerate(lanes):
        area_px = polygon_area(lane)
        area_unit = max(area_px / 10000.0, 1e-6)
        vehicle_ids = []
        class_counts: Dict[str, int] = {}
        for tr in tracks:
            cx = float(tr.get("center_x", 0.0))
            cy = float(tr.get("center_y", 0.0))
            if point_in_polygon((cx, cy), lane):
                vehicle_ids.append(int(tr.get("track_id", -1)))
                vt = str(tr.get("vehicle_type", "unknown"))
                class_counts[vt] = class_counts.get(vt, 0) + 1

        count = len(vehicle_ids)
        density = float(count) / area_unit
        results.append(
            {
                "lane_index": idx,
                "lane_name": f"Lane {idx + 1}",
                "vehicle_count": int(count),
                "density": float(density),
                "queue_status": queue_status_from_density(float(density)),
                "area_px": float(area_px),
                "vehicle_ids": vehicle_ids,
                "class_counts": class_counts,
            }
        )
    return results
