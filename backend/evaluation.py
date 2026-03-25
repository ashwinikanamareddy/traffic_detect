from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from scipy.optimize import linear_sum_assignment


BBox = Tuple[float, float, float, float]


@dataclass
class DetectionMatchSummary:
    tp: int
    fp: int
    fn: int
    matched_pairs: List[Tuple[int, int, str, str]]

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 0.0 if denom == 0 else float(self.tp) / float(denom)

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 0.0 if denom == 0 else float(self.tp) / float(denom)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 0.0 if (p + r) == 0 else (2.0 * p * r) / (p + r)


def _first_present(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _parse_bbox(value) -> Optional[BBox]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s[0] == "(" and s[-1] == ")":
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = [float(p) for p in parts]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _prepare_detection_rows(df: pd.DataFrame) -> pd.DataFrame:
    frame_col = _first_present(df, ["frame", "frame_id"])
    bbox_col = _first_present(df, ["bbox"])
    obj_col = _first_present(df, ["track_id", "id"])
    cls_col = _first_present(df, ["vehicle_type", "class", "label"])
    if frame_col is None or bbox_col is None:
        return pd.DataFrame(columns=["frame", "bbox", "obj_id", "label"])

    rows = []
    for _, row in df.iterrows():
        bbox = _parse_bbox(row.get(bbox_col))
        if bbox is None:
            continue
        try:
            frame = int(float(row.get(frame_col)))
        except Exception:
            continue
        obj_id = str(row.get(obj_col)) if obj_col is not None else ""
        label = str(row.get(cls_col)) if cls_col is not None else ""
        rows.append({"frame": frame, "bbox": bbox, "obj_id": obj_id, "label": label})

    if not rows:
        return pd.DataFrame(columns=["frame", "bbox", "obj_id", "label"])
    return pd.DataFrame(rows)


def match_detections(pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold: float = 0.5) -> DetectionMatchSummary:
    pred = _prepare_detection_rows(pred_df)
    gt = _prepare_detection_rows(gt_df)

    if pred.empty and gt.empty:
        return DetectionMatchSummary(tp=0, fp=0, fn=0, matched_pairs=[])

    tp = 0
    fp = 0
    fn = 0
    matched_pairs: List[Tuple[int, int, str, str]] = []

    all_frames = sorted(set(pred["frame"].tolist()) | set(gt["frame"].tolist()))
    for frame in all_frames:
        p = pred[pred["frame"] == frame].reset_index(drop=True)
        g = gt[gt["frame"] == frame].reset_index(drop=True)

        if p.empty and g.empty:
            continue
        if p.empty:
            fn += len(g)
            continue
        if g.empty:
            fp += len(p)
            continue

        candidates = []
        for pi, prow in p.iterrows():
            for gi, grow in g.iterrows():
                iou = _bbox_iou(prow["bbox"], grow["bbox"])
                if iou >= iou_threshold:
                    candidates.append((iou, pi, gi))
        candidates.sort(key=lambda x: x[0], reverse=True)

        used_p = set()
        used_g = set()
        for _, pi, gi in candidates:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            tp += 1
            matched_pairs.append(
                (
                    frame,
                    pi,
                    str(p.iloc[pi]["obj_id"]),
                    str(g.iloc[gi]["obj_id"]),
                )
            )

        fp += len(p) - len(used_p)
        fn += len(g) - len(used_g)

    return DetectionMatchSummary(tp=tp, fp=fp, fn=fn, matched_pairs=matched_pairs)


def compute_idf1(pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_threshold: float = 0.5) -> Dict[str, float]:
    pred = _prepare_detection_rows(pred_df)
    gt = _prepare_detection_rows(gt_df)
    if pred.empty or gt.empty:
        return {"idtp": 0.0, "idfp": float(len(pred)), "idfn": float(len(gt)), "idf1": 0.0}

    det_match = match_detections(pred, gt, iou_threshold=iou_threshold)
    if not det_match.matched_pairs:
        return {
            "idtp": 0.0,
            "idfp": float(len(pred)),
            "idfn": float(len(gt)),
            "idf1": 0.0,
        }

    pair_counts: Dict[Tuple[str, str], int] = {}
    pred_ids = set()
    gt_ids = set()
    for _, _, pred_id, gt_id in det_match.matched_pairs:
        pair_counts[(gt_id, pred_id)] = pair_counts.get((gt_id, pred_id), 0) + 1
        pred_ids.add(pred_id)
        gt_ids.add(gt_id)

    gt_list = sorted(gt_ids)
    pred_list = sorted(pred_ids)
    if not gt_list or not pred_list:
        return {"idtp": 0.0, "idfp": float(len(pred)), "idfn": float(len(gt)), "idf1": 0.0}

    # Maximize assignment counts -> minimize negative counts
    import numpy as np

    mat = np.zeros((len(gt_list), len(pred_list)), dtype=float)
    for gi, gid in enumerate(gt_list):
        for pi, pid in enumerate(pred_list):
            mat[gi, pi] = -float(pair_counts.get((gid, pid), 0))

    gi_idx, pi_idx = linear_sum_assignment(mat)
    idtp = 0.0
    for gi, pi in zip(gi_idx, pi_idx):
        idtp += float(pair_counts.get((gt_list[gi], pred_list[pi]), 0))

    idfn = float(len(gt)) - idtp
    idfp = float(len(pred)) - idtp
    denom = (2.0 * idtp) + idfp + idfn
    idf1 = 0.0 if denom <= 0 else (2.0 * idtp) / denom
    return {"idtp": idtp, "idfp": idfp, "idfn": idfn, "idf1": idf1}


def compute_queue_errors(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, float]:
    frame_pred_col = _first_present(pred_df, ["frame", "frame_id"])
    frame_gt_col = _first_present(gt_df, ["frame", "frame_id"])
    if frame_pred_col is None or frame_gt_col is None:
        return {"queue_mae": 0.0, "queue_rmse": 0.0, "density_mae": 0.0, "count": 0.0}

    p = pred_df.copy()
    g = gt_df.copy()

    p["frame_key"] = pd.to_numeric(p[frame_pred_col], errors="coerce")
    g["frame_key"] = pd.to_numeric(g[frame_gt_col], errors="coerce")
    p = p.dropna(subset=["frame_key"])
    g = g.dropna(subset=["frame_key"])
    if p.empty or g.empty:
        return {"queue_mae": 0.0, "queue_rmse": 0.0, "density_mae": 0.0, "count": 0.0}

    merge_cols = ["frame_key"]
    p_cols = ["frame_key"]
    g_cols = ["frame_key"]

    if "queue_count" in p.columns and "queue_count" in g.columns:
        p_cols.append("queue_count")
        g_cols.append("queue_count")
    if "queue_density" in p.columns and "queue_density" in g.columns:
        p_cols.append("queue_density")
        g_cols.append("queue_density")

    m = p[p_cols].merge(g[g_cols], on="frame_key", suffixes=("_pred", "_gt"))
    if m.empty:
        return {"queue_mae": 0.0, "queue_rmse": 0.0, "density_mae": 0.0, "count": 0.0}

    out = {"queue_mae": 0.0, "queue_rmse": 0.0, "density_mae": 0.0, "count": float(len(m))}
    if "queue_count_pred" in m.columns and "queue_count_gt" in m.columns:
        e = (pd.to_numeric(m["queue_count_pred"], errors="coerce") - pd.to_numeric(m["queue_count_gt"], errors="coerce")).abs().dropna()
        if not e.empty:
            out["queue_mae"] = float(e.mean())
            out["queue_rmse"] = float((e.pow(2).mean()) ** 0.5)
    if "queue_density_pred" in m.columns and "queue_density_gt" in m.columns:
        ed = (pd.to_numeric(m["queue_density_pred"], errors="coerce") - pd.to_numeric(m["queue_density_gt"], errors="coerce")).abs().dropna()
        if not ed.empty:
            out["density_mae"] = float(ed.mean())
    return out


def compute_violation_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, float]:
    frame_p = _first_present(pred_df, ["frame", "frame_id"])
    frame_g = _first_present(gt_df, ["frame", "frame_id"])
    type_p = _first_present(pred_df, ["violation_type", "type"])
    type_g = _first_present(gt_df, ["violation_type", "type"])
    if frame_p is None or frame_g is None or type_p is None or type_g is None:
        return {"viol_precision": 0.0, "viol_recall": 0.0, "viol_f1": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}

    p = pred_df[[frame_p, type_p]].copy()
    g = gt_df[[frame_g, type_g]].copy()
    p.columns = ["frame", "type"]
    g.columns = ["frame", "type"]

    p["frame"] = pd.to_numeric(p["frame"], errors="coerce")
    g["frame"] = pd.to_numeric(g["frame"], errors="coerce")
    p = p.dropna(subset=["frame"])
    g = g.dropna(subset=["frame"])
    p["frame"] = p["frame"].astype(int)
    g["frame"] = g["frame"].astype(int)
    p["type"] = p["type"].astype(str).str.strip()
    g["type"] = g["type"].astype(str).str.strip()

    pred_set = set((int(r.frame), r.type) for r in p.itertuples(index=False))
    gt_set = set((int(r.frame), r.type) for r in g.itertuples(index=False))
    tp = float(len(pred_set & gt_set))
    fp = float(len(pred_set - gt_set))
    fn = float(len(gt_set - pred_set))
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "viol_precision": precision,
        "viol_recall": recall,
        "viol_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_all(
    pred_tracking_df: pd.DataFrame,
    gt_tracking_df: pd.DataFrame,
    pred_queue_df: Optional[pd.DataFrame] = None,
    gt_queue_df: Optional[pd.DataFrame] = None,
    pred_viol_df: Optional[pd.DataFrame] = None,
    gt_viol_df: Optional[pd.DataFrame] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    det = match_detections(pred_tracking_df, gt_tracking_df, iou_threshold=iou_threshold)
    idm = compute_idf1(pred_tracking_df, gt_tracking_df, iou_threshold=iou_threshold)

    out: Dict[str, float] = {
        "det_precision": det.precision,
        "det_recall": det.recall,
        "det_f1": det.f1,
        "det_tp": float(det.tp),
        "det_fp": float(det.fp),
        "det_fn": float(det.fn),
        "idf1": float(idm["idf1"]),
        "idtp": float(idm["idtp"]),
        "idfp": float(idm["idfp"]),
        "idfn": float(idm["idfn"]),
    }

    if pred_queue_df is not None and gt_queue_df is not None:
        out.update(compute_queue_errors(pred_queue_df, gt_queue_df))
    if pred_viol_df is not None and gt_viol_df is not None:
        out.update(compute_violation_metrics(pred_viol_df, gt_viol_df))
    return out
