from collections import defaultdict, deque
from typing import Dict, List, Tuple


Line = Tuple[Tuple[float, float], Tuple[float, float]]


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _bbox_to_xyxy(bbox) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _bbox_iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = _bbox_to_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_to_xyxy(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    a_area = float(max(1.0, (ax2 - ax1) * (ay2 - ay1)))
    b_area = float(max(1.0, (bx2 - bx1) * (by2 - by1)))
    return inter_area / max(1.0, a_area + b_area - inter_area)


def _center_inside_bbox(cx: float, cy: float, bbox, margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = _bbox_to_xyxy(bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * max(0.0, float(margin))
    my = bh * max(0.0, float(margin))
    return (x1 - mx) <= cx <= (x2 + mx) and (y1 - my) <= cy <= (y2 + my)


def _line_side(point: Tuple[float, float], line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _heading_deg(prev: Tuple[float, float], curr: Tuple[float, float]) -> float:
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if dx == 0 and dy == 0:
        return 0.0
    import math

    return float(math.degrees(math.atan2(dy, dx)))


class SignalController:
    def __init__(self, red_frames: int = 120, green_frames: int = 120):
        self.red_frames = max(1, int(red_frames))
        self.green_frames = max(1, int(green_frames))
        self.cycle = self.red_frames + self.green_frames

    def state_for_frame(self, frame_id: int) -> str:
        idx = int(frame_id) % self.cycle
        return "RED" if idx < self.red_frames else "GREEN"


class ViolationDetector:
    def __init__(
        self,
        stop_line: Line,
        signal_controller: SignalController,
        speed_threshold: float = 32.0,
        acceleration_threshold: float = 18.0,
        direction_change_threshold: float = 48.0,
        zigzag_heading_threshold: float = 20.0,
        zigzag_window: int = 6,
        triple_riding_min_persons: int = 3,
        heavy_load_min_persons: int = 4,
        association_iou_threshold: float = 0.01,
        association_center_margin: float = 0.25,
    ):
        self.stop_line = stop_line
        self.signal_controller = signal_controller

        self.speed_threshold = float(speed_threshold)
        self.acceleration_threshold = float(acceleration_threshold)
        self.direction_change_threshold = float(direction_change_threshold)
        self.zigzag_heading_threshold = float(zigzag_heading_threshold)
        self.zigzag_window = max(4, int(zigzag_window))
        self.triple_riding_min_persons = max(2, int(triple_riding_min_persons))
        self.heavy_load_min_persons = max(self.triple_riding_min_persons + 1, int(heavy_load_min_persons))
        self.association_iou_threshold = max(0.0, float(association_iou_threshold))
        self.association_center_margin = max(0.0, float(association_center_margin))

        self._prev_center: Dict[int, Tuple[float, float]] = {}
        self._prev_speed: Dict[int, float] = defaultdict(float)
        self._prev_side: Dict[int, float] = {}
        self._heading_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.zigzag_window))
        self._red_crossing_cooldown: Dict[int, int] = defaultdict(int)
        self._rash_cooldown: Dict[int, int] = defaultdict(int)
        self._triple_cooldown: Dict[int, int] = defaultdict(int)
        self._mobile_cooldown: Dict[int, int] = defaultdict(int)
        self._helmet_cooldown: Dict[int, int] = defaultdict(int)
        self._heavy_load_cooldown: Dict[int, int] = defaultdict(int)

    def _detect_zigzag(self, track_id: int) -> bool:
        headings = list(self._heading_history[track_id])
        if len(headings) < 4:
            return False

        diffs = []
        for i in range(1, len(headings)):
            diffs.append(_normalize_angle_deg(headings[i] - headings[i - 1]))

        sign_changes = 0
        for i in range(1, len(diffs)):
            a = diffs[i - 1]
            b = diffs[i]
            if abs(a) < self.zigzag_heading_threshold or abs(b) < self.zigzag_heading_threshold:
                continue
            if (a > 0 > b) or (a < 0 < b):
                sign_changes += 1

        return sign_changes >= 2

    def _associated_count(self, track_bbox, obj_list: List[Dict], iou_threshold: float = 0.01, center_margin: float = 0.2) -> int:
        if not obj_list:
            return 0
        count = 0
        for obj in obj_list:
            obox = obj.get("bbox")
            if not obox:
                continue
            iou = _bbox_iou_xywh(track_bbox, obox)
            ox, oy, ow, oh = obox
            ocx = float(ox) + float(ow) / 2.0
            ocy = float(oy) + float(oh) / 2.0
            if iou >= iou_threshold or _center_inside_bbox(ocx, ocy, track_bbox, margin=center_margin):
                count += 1
        return count

    def update(self, tracks: List[Dict], frame_id: int, fps: float, timestamp: str, context_objects: Dict[str, List[Dict]] = None):
        fps = max(1.0, float(fps))
        signal_state = self.signal_controller.state_for_frame(frame_id)
        context_objects = context_objects or {}

        persons = context_objects.get("person", []) or []
        phones = context_objects.get("cell_phone", []) or []
        no_helmets = context_objects.get("no_helmet", []) or []
        heavy_loads = context_objects.get("heavy_load", []) or []

        frame_events = []

        for track in tracks:
            track_id = int(track["track_id"])
            center = (float(track["center_x"]), float(track["center_y"]))
            vehicle_type = track.get("vehicle_type", "unknown")
            track_bbox = track.get("bbox")

            prev_center = self._prev_center.get(track_id)
            speed = 0.0
            acceleration = 0.0
            heading_change = 0.0

            if prev_center is not None:
                dist = _distance(prev_center, center)
                speed = dist * fps
                prev_speed = float(self._prev_speed.get(track_id, 0.0))
                acceleration = (speed - prev_speed) * fps

                prev_heading = self._heading_history[track_id][-1] if self._heading_history[track_id] else None
                heading = _heading_deg(prev_center, center)
                self._heading_history[track_id].append(heading)
                if prev_heading is not None:
                    heading_change = abs(_normalize_angle_deg(heading - prev_heading))
            else:
                self._heading_history[track_id].append(0.0)

            previous_side = self._prev_side.get(track_id)
            current_side = _line_side(center, self.stop_line)
            crossed = previous_side is not None and ((previous_side > 0 >= current_side) or (previous_side < 0 <= current_side))

            if self._red_crossing_cooldown[track_id] > 0:
                self._red_crossing_cooldown[track_id] -= 1

            if crossed and signal_state == "RED" and self._red_crossing_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "Red Light Jump",
                        "signal_state": signal_state,
                        "vehicle_type": vehicle_type,
                    }
                )
                self._red_crossing_cooldown[track_id] = int(fps)

            is_zigzag = self._detect_zigzag(track_id)
            rash = (
                (speed >= self.speed_threshold and abs(acceleration) >= self.acceleration_threshold)
                or (heading_change >= self.direction_change_threshold and speed >= (self.speed_threshold * 0.6))
                or is_zigzag
            )

            if self._rash_cooldown[track_id] > 0:
                self._rash_cooldown[track_id] -= 1

            if rash and self._rash_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "Rash Driving",
                        "signal_state": signal_state,
                        "vehicle_type": vehicle_type,
                        "speed": float(speed),
                        "acceleration": float(acceleration),
                        "angle_change": float(heading_change),
                        "zig_zag": bool(is_zigzag),
                    }
                )
                self._rash_cooldown[track_id] = int(fps * 0.5)

            if track_bbox:
                rider_count = self._associated_count(
                    track_bbox,
                    persons,
                    iou_threshold=self.association_iou_threshold,
                    center_margin=self.association_center_margin,
                )
                phone_count = self._associated_count(
                    track_bbox,
                    phones,
                    iou_threshold=max(0.001, self.association_iou_threshold * 0.25),
                    center_margin=self.association_center_margin,
                )
                no_helmet_count = self._associated_count(
                    track_bbox,
                    no_helmets,
                    iou_threshold=max(0.001, self.association_iou_threshold * 0.25),
                    center_margin=self.association_center_margin,
                )
                heavy_load_count = self._associated_count(
                    track_bbox,
                    heavy_loads,
                    iou_threshold=self.association_iou_threshold,
                    center_margin=max(0.2, self.association_center_margin * 0.8),
                )
            else:
                rider_count = 0
                phone_count = 0
                no_helmet_count = 0
                heavy_load_count = 0

            for cooldown_map in [self._triple_cooldown, self._mobile_cooldown, self._helmet_cooldown, self._heavy_load_cooldown]:
                if cooldown_map[track_id] > 0:
                    cooldown_map[track_id] -= 1

            if vehicle_type == "bike" and rider_count >= self.triple_riding_min_persons and self._triple_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "Triple Riding",
                        "signal_state": signal_state,
                        "vehicle_type": vehicle_type,
                        "rider_count": int(rider_count),
                    }
                )
                self._triple_cooldown[track_id] = int(fps * 1.2)

            if vehicle_type == "bike" and rider_count >= 1 and phone_count >= 1 and self._mobile_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "Mobile Usage While Driving",
                        "signal_state": signal_state,
                        "vehicle_type": vehicle_type,
                        "rider_count": int(rider_count),
                        "phone_count": int(phone_count),
                    }
                )
                self._mobile_cooldown[track_id] = int(fps * 1.0)

            if vehicle_type == "bike" and no_helmet_count >= 1 and self._helmet_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "No Helmet",
                        "signal_state": signal_state,
                        "vehicle_type": vehicle_type,
                        "rider_count": int(rider_count),
                        "no_helmet_count": int(no_helmet_count),
                    }
                )
                self._helmet_cooldown[track_id] = int(fps * 1.2)

            inferred_heavy_load = vehicle_type == "bike" and rider_count >= self.heavy_load_min_persons
            if (heavy_load_count >= 1 or inferred_heavy_load) and self._heavy_load_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "Heavy Load",
                        "signal_state": signal_state,
                        "vehicle_type": vehicle_type,
                        "rider_count": int(rider_count),
                        "heavy_load_count": int(heavy_load_count),
                    }
                )
                self._heavy_load_cooldown[track_id] = int(fps * 1.5)

            self._prev_center[track_id] = center
            self._prev_speed[track_id] = speed
            self._prev_side[track_id] = current_side

        return signal_state, frame_events
