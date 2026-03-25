from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


BBox = Tuple[int, int, int, int]


def _center_of(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return float(x + w / 2.0), float(y + h / 2.0)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    a_area = float(max(1, aw * ah))
    b_area = float(max(1, bw * bh))
    return inter_area / max(1.0, a_area + b_area - inter_area)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    vehicle_type: str
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    last_frame_id: int = 0
    centers: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.centers:
            self.centers.append(_center_of(self.bbox))

    def update(self, bbox: BBox, vehicle_type: str, frame_id: int):
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_frame_id = frame_id
        self.centers.append(_center_of(bbox))

    def mark_missed(self):
        self.age += 1
        self.time_since_update += 1


class MultiObjectTracker:
    """SORT-like tracker with IoU + center-distance assignment and short occlusion handling."""

    def __init__(self, iou_threshold: float = 0.2, max_age: int = 12, min_hits: int = 1, max_center_distance: float = 120.0):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.max_center_distance = float(max_center_distance)

        self._next_id = 1
        self._tracks: List[_Track] = []

    def _score(self, track: _Track, detection: Dict) -> float:
        det_bbox = detection["bbox"]
        track_center = _center_of(track.bbox)
        det_center = _center_of(det_bbox)

        iou = _bbox_iou(track.bbox, det_bbox)
        dist = _distance(track_center, det_center)

        if dist > self.max_center_distance and iou < self.iou_threshold:
            return -1.0

        class_bonus = 0.15 if track.vehicle_type == detection.get("type", "") else 0.0
        dist_score = max(0.0, 1.0 - (dist / max(1.0, self.max_center_distance)))
        return iou + 0.35 * dist_score + class_bonus

    def _build_matches(self, detections: List[Dict]):
        candidates = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                score = self._score(track, det)
                if score >= 0:
                    candidates.append((score, ti, di))

        candidates.sort(reverse=True, key=lambda x: x[0])

        matched_tracks = set()
        matched_dets = set()
        matches = []

        for score, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matches.append((ti, di, score))
            matched_tracks.add(ti)
            matched_dets.add(di)

        unmatched_track_idx = [i for i in range(len(self._tracks)) if i not in matched_tracks]
        unmatched_det_idx = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_track_idx, unmatched_det_idx

    def update(self, detections: List[Dict], frame_id: int, timestamp: str):
        """
        Returns:
            tracks_for_frame: list of active/updated tracks for the frame
            tracking_rows: rows with mandatory tracking fields
        """
        for track in self._tracks:
            track.mark_missed()

        matches, unmatched_tracks, unmatched_dets = self._build_matches(detections)

        updated_track_ids = set()

        for ti, di, _score in matches:
            det = detections[di]
            track = self._tracks[ti]
            track.update(det["bbox"], det.get("type", "unknown"), frame_id)
            updated_track_ids.add(track.track_id)

        for di in unmatched_dets:
            det = detections[di]
            track = _Track(
                track_id=self._next_id,
                bbox=det["bbox"],
                vehicle_type=det.get("type", "unknown"),
                hits=1,
                age=1,
                time_since_update=0,
                last_frame_id=frame_id,
                centers=[_center_of(det["bbox"])],
            )
            self._tracks.append(track)
            updated_track_ids.add(track.track_id)
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        tracks_for_frame = []
        tracking_rows = []

        for track in self._tracks:
            if track.time_since_update != 0:
                continue
            if track.hits < self.min_hits and frame_id > self.min_hits:
                continue

            cx, cy = _center_of(track.bbox)
            tracks_for_frame.append(
                {
                    "track_id": int(track.track_id),
                    "vehicle_type": track.vehicle_type,
                    "bbox": track.bbox,
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "trajectory": list(track.centers),
                }
            )
            tracking_rows.append(
                {
                    "frame_id": int(frame_id),
                    "track_id": int(track.track_id),
                    "vehicle_type": track.vehicle_type,
                    "bbox": f"{track.bbox[0]},{track.bbox[1]},{track.bbox[2]},{track.bbox[3]}",
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "timestamp": timestamp,
                }
            )

        return tracks_for_frame, tracking_rows
