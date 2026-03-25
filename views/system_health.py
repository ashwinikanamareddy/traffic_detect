from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from backend.evaluation import evaluate_all


def _read_csv_safe(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return None


def _load_prediction_data():
    pred_tracking = None
    pred_queue = None
    pred_viol = None

    run_dir = st.session_state.get("run_dir")
    if run_dir and os.path.isdir(run_dir):
        t_path = os.path.join(run_dir, "tracking_log.csv")
        q_path = os.path.join(run_dir, "traffic_log.csv")
        v_path = os.path.join(run_dir, "violations_log.csv")
        if os.path.exists(t_path):
            pred_tracking = pd.read_csv(t_path)
        if os.path.exists(q_path):
            pred_queue = pd.read_csv(q_path)
        if os.path.exists(v_path):
            pred_viol = pd.read_csv(v_path)

    if pred_queue is None:
        df = st.session_state.get("df")
        if isinstance(df, pd.DataFrame) and not df.empty:
            pred_queue = df.copy()
            pred_tracking = df.copy() if pred_tracking is None else pred_tracking

    return pred_tracking, pred_queue, pred_viol


def _kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e7edf4;border-radius:14px;box-shadow:0 6px 18px rgba(15,23,42,0.05);padding:12px;">
            <div style="font-size:12px;color:#64748b;">{label}</div>
            <div style="font-size:28px;font-weight:800;color:#0f172a;line-height:1.1;">{value}</div>
            <div style="font-size:12px;color:#94a3b8;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show():
    st.markdown(
        """
        <div style="background:linear-gradient(180deg,#ffffff 0%,#f7fbff 100%);border:1px solid #e7edf4;border-radius:16px;box-shadow:0 8px 22px rgba(15,23,42,0.05);padding:16px 18px;margin-bottom:14px;">
            <h1 style="margin:0;font-size:34px;color:#0f172a;">System Health & Evaluation</h1>
            <p style="margin:6px 0 0 0;color:#64748b;">Upload ground-truth CSV files to compute objective metrics for detection, tracking, queue analytics and violations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_tracking, pred_queue, pred_viol = _load_prediction_data()
    if pred_tracking is None:
        st.warning("No prediction run found. Process a video first from Dashboard.")
        return

    st.markdown("### Ground Truth Upload")
    c1, c2, c3 = st.columns(3)
    gt_tracking_file = c1.file_uploader("GT Tracking CSV (required)", type=["csv"], key="gt_tracking_csv")
    gt_queue_file = c2.file_uploader("GT Queue CSV (optional)", type=["csv"], key="gt_queue_csv")
    gt_viol_file = c3.file_uploader("GT Violations CSV (optional)", type=["csv"], key="gt_viol_csv")

    with st.expander("Expected CSV schema"):
        st.markdown("`GT Tracking`: `frame` or `frame_id`, `bbox` (`x,y,w,h`), `track_id`")
        st.markdown("`GT Queue`: `frame` or `frame_id`, `queue_count`, optional `queue_density`")
        st.markdown("`GT Violations`: `frame` or `frame_id`, `violation_type`")

    gt_tracking = _read_csv_safe(gt_tracking_file)
    gt_queue = _read_csv_safe(gt_queue_file)
    gt_viol = _read_csv_safe(gt_viol_file)

    iou_threshold = st.slider("IoU threshold", min_value=0.3, max_value=0.9, value=0.5, step=0.05)
    run_eval = st.button("Run Evaluation", width="stretch")

    if not run_eval:
        st.info("Upload ground truth and click Run Evaluation.")
        return
    if gt_tracking is None:
        st.error("GT Tracking CSV is required to compute detection/tracking accuracy.")
        return

    metrics = evaluate_all(
        pred_tracking_df=pred_tracking,
        gt_tracking_df=gt_tracking,
        pred_queue_df=pred_queue,
        gt_queue_df=gt_queue,
        pred_viol_df=pred_viol,
        gt_viol_df=gt_viol,
        iou_threshold=float(iou_threshold),
    )

    st.markdown("### Accuracy Report")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        _kpi("Detection Precision", f"{metrics.get('det_precision', 0.0) * 100:.1f}%", "IoU matched TP/(TP+FP)")
    with r1c2:
        _kpi("Detection Recall", f"{metrics.get('det_recall', 0.0) * 100:.1f}%", "IoU matched TP/(TP+FN)")
    with r1c3:
        _kpi("Detection F1", f"{metrics.get('det_f1', 0.0) * 100:.1f}%", "Harmonic mean of P/R")
    with r1c4:
        _kpi("Tracking IDF1", f"{metrics.get('idf1', 0.0) * 100:.1f}%", "Identity consistency score")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        _kpi("Queue MAE", f"{metrics.get('queue_mae', 0.0):.3f}", "Lower is better")
    with r2c2:
        _kpi("Queue RMSE", f"{metrics.get('queue_rmse', 0.0):.3f}", "Lower is better")
    with r2c3:
        _kpi("Density MAE", f"{metrics.get('density_mae', 0.0):.4f}", "Lower is better")

    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        _kpi("Violation Precision", f"{metrics.get('viol_precision', 0.0) * 100:.1f}%", "Frame+type event match")
    with r3c2:
        _kpi("Violation Recall", f"{metrics.get('viol_recall', 0.0) * 100:.1f}%", "Frame+type event match")
    with r3c3:
        _kpi("Violation F1", f"{metrics.get('viol_f1', 0.0) * 100:.1f}%", "Harmonic mean of P/R")

    st.markdown("### Raw Metrics")
    st.dataframe(
        pd.DataFrame([metrics]),
        width="stretch",
        hide_index=True,
    )
