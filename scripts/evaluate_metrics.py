import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.evaluation import evaluate_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic AI predictions against ground truth CSV files.")
    parser.add_argument("--pred-tracking", required=True, help="Path to predicted tracking CSV")
    parser.add_argument("--gt-tracking", required=True, help="Path to ground-truth tracking CSV")
    parser.add_argument("--pred-queue", help="Path to predicted queue/traffic CSV")
    parser.add_argument("--gt-queue", help="Path to ground-truth queue CSV")
    parser.add_argument("--pred-viol", help="Path to predicted violations CSV")
    parser.add_argument("--gt-viol", help="Path to ground-truth violations CSV")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    args = parser.parse_args()

    pred_tracking = pd.read_csv(args.pred_tracking)
    gt_tracking = pd.read_csv(args.gt_tracking)
    pred_queue = pd.read_csv(args.pred_queue) if args.pred_queue else None
    gt_queue = pd.read_csv(args.gt_queue) if args.gt_queue else None
    pred_viol = pd.read_csv(args.pred_viol) if args.pred_viol else None
    gt_viol = pd.read_csv(args.gt_viol) if args.gt_viol else None

    metrics = evaluate_all(
        pred_tracking_df=pred_tracking,
        gt_tracking_df=gt_tracking,
        pred_queue_df=pred_queue,
        gt_queue_df=gt_queue,
        pred_viol_df=pred_viol,
        gt_viol_df=gt_viol,
        iou_threshold=float(args.iou),
    )

    print("Evaluation Metrics")
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
