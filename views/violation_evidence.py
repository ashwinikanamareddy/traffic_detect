import io
import os
import zipfile
from datetime import datetime

import streamlit as st


def _severity_dot(severity: str):
    sev = (severity or "").lower()
    if sev == "high":
        return "dot-red"
    if sev == "medium":
        return "dot-yellow"
    return "dot-green"


def _badge(cls, text):
    return f"<span class='badge {cls}'>{text}</span>"


def _status_badge(status: str):
    val = (status or "").lower()
    if val == "verified":
        return _badge("badge-verified", "Verified \u2705")
    if val == "rejected":
        return _badge("badge-rejected", "Rejected \u274c")
    return _badge("badge-pending", "Pending")


def _severity_badge(severity: str):
    sev = (severity or "").lower()
    if sev == "high":
        return _badge("badge-high", "High")
    if sev == "medium":
        return _badge("badge-medium", "Medium")
    return _badge("badge-low", "Low")


def _find_violation(violations, violation_id):
    for idx, v in enumerate(violations):
        if v.get("violation_id") == violation_id:
            return idx, v
    return None, None


def _zip_evidence_folder(folder_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for f in files:
                full_path = os.path.join(root, f)
                rel = os.path.relpath(full_path, folder_path)
                zf.write(full_path, rel)
    buffer.seek(0)
    return buffer


def _resolve_evidence_path(violation: dict) -> str:
    evidence_path = str(violation.get("evidence_path", "") or "").strip()
    if evidence_path and os.path.exists(evidence_path):
        return evidence_path
    vid = str(violation.get("violation_id", "") or "").strip()
    if vid:
        alt = os.path.join("evidence", vid)
        if os.path.exists(alt):
            return alt
    return evidence_path


def show():
    st.markdown(
        """
        <style>
        .evi-root { padding: 8px 6px 18px 6px; }

        .panel {
            background: #ffffff;
            border: 1px solid #e7edf4;
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            padding: 14px;
        }

        .section-title {
            font-size: 14px;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 10px 0;
        }

        .muted {
            color: #64748b;
            font-size: 12px;
        }

        .list-card {
            border: 1px solid #eef2f7;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: #ffffff;
        }

        .list-card.active {
            border-color: #14b8a6;
            background: #ecfeff;
        }

        .list-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .dot-red { background: #ef4444; }
        .dot-yellow { background: #f59e0b; }
        .dot-green { background: #22c55e; }

        .list-id {
            font-weight: 800;
            color: #0f172a;
            font-size: 12px;
        }

        .list-sub {
            color: #64748b;
            font-size: 11px;
            margin-top: 2px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 10px;
            font-weight: 700;
        }

        .badge-high { background: #fee2e2; color: #991b1b; }
        .badge-medium { background: #ffedd5; color: #9a3412; }
        .badge-low { background: #fef9c3; color: #854d0e; }

        .badge-verified { background: #dcfce7; color: #166534; }
        .badge-rejected { background: #fee2e2; color: #991b1b; }
        .badge-pending { background: #e5e7eb; color: #4b5563; }

        .detail-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 10px;
        }

        .detail-item {
            background: #f8fafc;
            border: 1px solid #eef2f7;
            border-radius: 10px;
            padding: 8px 10px;
            font-size: 11px;
            color: #475569;
        }

        .detail-item strong {
            display: block;
            color: #0f172a;
            font-size: 12px;
        }

        .action-row {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }

        .img-card {
            border: 1px solid #eef2f7;
            border-radius: 12px;
            padding: 10px;
        }

        .report-item {
            border: 1px solid #eef2f7;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 8px;
            background: #f8fafc;
        }

        .report-title {
            font-weight: 700;
            color: #0f172a;
            font-size: 12px;
            margin-bottom: 4px;
        }

        .report-sub {
            color: #64748b;
            font-size: 11px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Violation Evidence Center")
    st.markdown("<div class='evi-root'>", unsafe_allow_html=True)

    violations = st.session_state.get("violations", [])
    if not violations:
        st.markdown(
            "<div class='panel'><div class='muted' style='text-align:center;'>No violations detected yet.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown("<div class='muted'>Review and manage violation evidence with visual proof</div>", unsafe_allow_html=True)
    with head_right:
        status_filter = st.selectbox("Status", ["All Status", "Pending", "Verified", "Rejected"], index=0, label_visibility="collapsed")

    if status_filter != "All Status":
        violations = [v for v in violations if str(v.get("status", "")).lower() == status_filter.lower()]

    if not violations:
        st.markdown(
            "<div class='panel'><div class='muted' style='text-align:center;'>No violations for selected filter.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    left, right = st.columns([1, 2])

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Violation List</div>", unsafe_allow_html=True)
        selected = st.session_state.get("selected_violation")
        for idx, v in enumerate(violations):
            vid = v.get("violation_id", f"V-{idx+1}")
            vtype = v.get("type", "Violation")
            loc = v.get("location", "Unknown")
            sev = v.get("severity", "Low")
            dot_cls = _severity_dot(sev)
            active_cls = "active" if selected == vid else ""
            st.markdown(
                f"""
                <div class="list-card {active_cls}">
                    <div class="list-top">
                        <div class="list-id">{vid}</div>
                        <span class="dot {dot_cls}"></span>
                    </div>
                    <div class="list-sub">{vtype}</div>
                    <div class="list-sub">{loc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Select", key=f"select_{vid}", width="stretch"):
                st.session_state.selected_violation = vid
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        selected_id = st.session_state.get("selected_violation") or violations[0].get("violation_id")
        sel_idx, sel = _find_violation(violations, selected_id)
        if sel is None:
            st.markdown("<div class='panel'><div class='muted'>No violation selected.</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        header = f"{sel.get('violation_id', 'Violation')} "
        st.markdown(
            f"""
            <div class="section-title">{header}{_severity_badge(sel.get('severity'))} {_status_badge(sel.get('status'))}</div>
            """,
            unsafe_allow_html=True,
        )

        ts = sel.get("timestamp")
        try:
            ts_display = datetime.fromisoformat(str(ts)).strftime("%H:%M:%S")
            date_display = datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d")
        except Exception:
            ts_display = str(ts)
            date_display = ""

        speed_val = sel.get("speed", None)
        speed_text = f"{float(speed_val):.1f} px/s" if speed_val is not None else "N/A"
        q_density = sel.get("queue_density", None)
        q_text = f"{float(q_density):.4f}" if q_density is not None else "N/A"

        st.markdown(
            f"""
            <div class="detail-grid">
                <div class="detail-item"><strong>Violation Type</strong>{sel.get('type', 'Violation')}</div>
                <div class="detail-item"><strong>Location</strong>{sel.get('location', 'Unknown')}</div>
                <div class="detail-item"><strong>Camera ID</strong>{sel.get('camera_id', 'CAM-01')}</div>
                <div class="detail-item"><strong>Timestamp</strong>{ts_display}</div>
                <div class="detail-item"><strong>Vehicle Number</strong>{sel.get('vehicle_number', 'N/A')}</div>
                <div class="detail-item"><strong>Vehicle Type</strong>{sel.get('vehicle_type', 'N/A')}</div>
                <div class="detail-item"><strong>Signal State</strong>{sel.get('signal_state', 'N/A')}</div>
                <div class="detail-item"><strong>Speed</strong>{speed_text}</div>
                <div class="detail-item"><strong>Queue Density</strong>{q_text}</div>
                <div class="detail-item"><strong>Date</strong>{date_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("\u2705 Verify Violation", width="stretch"):
                idx, _ = _find_violation(st.session_state.violations, sel.get("violation_id"))
                if idx is not None:
                    st.session_state.violations[idx]["status"] = "Verified"
        with action_col2:
            if st.button("\u274c Reject Violation", width="stretch"):
                idx, _ = _find_violation(st.session_state.violations, sel.get("violation_id"))
                if idx is not None:
                    st.session_state.violations[idx]["status"] = "Rejected"
        st.markdown("</div>", unsafe_allow_html=True)

        evidence_path = _resolve_evidence_path(sel)
        before_path = os.path.join(evidence_path, "before.jpg")
        moment_path = os.path.join(evidence_path, "moment.jpg")
        after_path = os.path.join(evidence_path, "after.jpg")
        clip_path = os.path.join(evidence_path, "clip.mp4")

        before_exists = os.path.exists(before_path)
        moment_exists = os.path.exists(moment_path)
        after_exists = os.path.exists(after_path)
        clip_exists = os.path.exists(clip_path)

        if before_exists or moment_exists or after_exists:
            st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>\U0001F4F8 Evidence Images</div>", unsafe_allow_html=True)
            img1, img2, img3 = st.columns(3)
            with img1:
                if before_exists:
                    st.caption("Before Violation")
                    st.image(before_path, width="stretch")
            with img2:
                if moment_exists:
                    st.caption("Violation Moment")
                    st.image(moment_path, width="stretch")
            with img3:
                if after_exists:
                    st.caption("After Violation")
                    st.image(after_path, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='panel' style='margin-top:12px;'><div class='muted'>Evidence images not found.</div></div>",
                unsafe_allow_html=True,
            )

        if clip_exists:
            st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>\U0001F3A5 Video Evidence</div>", unsafe_allow_html=True)
            try:
                with open(clip_path, "rb") as f:
                    st.video(f.read())
            except Exception:
                st.video(clip_path)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='panel' style='margin-top:12px;'><div class='muted'>Violation clip not found.</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>AI Analysis Report</div>", unsafe_allow_html=True)
        analysis = sel.get("analysis", {}) or {}
        if analysis:
            st.markdown(
                f"""
                <div class="report-item">
                    <div class="report-title">Vehicle Detection Confidence</div>
                    <div class="report-sub">{analysis.get('vehicle_detection_confidence', 'N/A')}%</div>
                </div>
                <div class="report-item">
                    <div class="report-title">Signal State Verification</div>
                    <div class="report-sub">{analysis.get('signal_state_verification', 'N/A')}%</div>
                </div>
                <div class="report-item">
                    <div class="report-title">Number Plate Recognition Confidence</div>
                    <div class="report-sub">{analysis.get('number_plate_confidence', 'N/A')}%</div>
                </div>
                <div class="report-item">
                    <div class="report-title">Trajectory Analysis</div>
                    <div class="report-sub">{analysis.get('trajectory_analysis', 'N/A')}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.write("No analysis available.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Export Evidence</div>", unsafe_allow_html=True)
        if evidence_path and os.path.exists(evidence_path):
            if st.button("Export Evidence", width="stretch"):
                zip_buffer = _zip_evidence_folder(evidence_path)
                st.download_button(
                    "Download Evidence ZIP",
                    data=zip_buffer,
                    file_name=f"{sel.get('violation_id', 'evidence')}.zip",
                    mime="application/zip",
                    width="stretch",
                )
        else:
            st.write("Evidence folder not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
