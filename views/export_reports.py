import os
from datetime import date

import streamlit as st

from backend import reports


def show():
    if "selected_report" not in st.session_state:
        st.session_state.selected_report = "Traffic Summary Report"
    if "export_format" not in st.session_state:
        st.session_state.export_format = "PDF"
    if "report_schedule" not in st.session_state:
        st.session_state.report_schedule = {}

    st.markdown(
        """
        <style>
        .exp-root { padding: 8px 6px 18px 6px; }

        .panel {
            background: #ffffff;
            border: 1px solid #e7edf4;
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            padding: 14px;
        }

        .title {
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }

        .subtitle {
            color: #64748b;
            font-size: 12px;
            margin-top: 4px;
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .report-card {
            border: 1px solid #eef2f7;
            border-radius: 12px;
            padding: 12px;
            background: #ffffff;
        }

        .report-card.active {
            border-color: #14b8a6;
            background: #ecfeff;
        }

        .card-title {
            font-size: 12px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .card-sub {
            font-size: 11px;
            color: #64748b;
        }

        .section-title {
            font-size: 14px;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 10px 0;
        }

        .format-btn {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 11px;
            text-align: center;
            margin-right: 6px;
            background: #ffffff;
        }

        .format-btn.active {
            border-color: #14b8a6;
            background: #ecfeff;
            font-weight: 700;
            color: #0f172a;
        }

        .recent-item {
            border: 1px solid #eef2f7;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: #ffffff;
        }

        .badge-ready {
            font-size: 10px;
            font-weight: 700;
            color: #166534;
            background: #dcfce7;
            border-radius: 999px;
            padding: 2px 8px;
        }

        .muted {
            color: #64748b;
            font-size: 11px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='exp-root'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Export Reports</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Generate and export analytics reports</div>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Select Report Type</div>", unsafe_allow_html=True)

        report_cards = [
            ("Traffic Summary Report", "Overall traffic flow and vehicle statistics", "📊"),
            ("Violation Report", "Detailed violation records and evidence", "⚠️"),
            ("Queue Analysis Report", "Queue length and density metrics", "📈"),
            ("Camera Performance Report", "Camera uptime and detection accuracy", "🎥"),
            ("Hourly Trends Report", "Hour-by-hour traffic patterns", "🕒"),
            ("Custom Report", "Build your own custom report", "🧩"),
        ]

        card_cols = st.columns(2)
        for idx, (name, desc, icon) in enumerate(report_cards):
            with card_cols[idx % 2]:
                active_cls = "active" if st.session_state.selected_report == name else ""
                st.markdown(
                    f"""
                    <div class="report-card {active_cls}">
                        <div class="card-title">{icon} {name}</div>
                        <div class="card-sub">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Select", key=f"report_{idx}", width="stretch"):
                    st.session_state.selected_report = name

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Report Configuration</div>", unsafe_allow_html=True)

        date_range = st.selectbox("Date Range", ["Today", "Last 7 Days", "Last 30 Days", "Custom Range"])
        start_date = None
        end_date = None
        if date_range == "Custom Range":
            d1, d2 = st.columns(2)
            start_date = d1.date_input("Start Date", value=date.today())
            end_date = d2.date_input("End Date", value=date.today())

        st.markdown("<div class='section-title' style='margin-top:6px;'>Export Format</div>", unsafe_allow_html=True)
        format_cols = st.columns(4)
        formats = ["PDF", "EXCEL", "CSV", "JSON"]
        for col, fmt in zip(format_cols, formats):
            with col:
                active = "active" if st.session_state.export_format == fmt else ""
                st.markdown(
                    f"<div class='format-btn {active}'>{fmt}</div>",
                    unsafe_allow_html=True,
                )
                if st.button(fmt, key=f"fmt_{fmt}", width="stretch"):
                    st.session_state.export_format = fmt

        st.markdown("<div class='section-title' style='margin-top:6px;'>Include Sections</div>", unsafe_allow_html=True)
        exec_sum = st.checkbox("Executive Summary", value=True)
        detail_stats = st.checkbox("Detailed Statistics", value=True)
        charts = st.checkbox("Charts and Graphs", value=True)
        raw_tables = st.checkbox("Raw Data Tables", value=False)
        evidence_imgs = st.checkbox("Violation Evidence Images", value=False)

        sections = []
        if exec_sum:
            sections.append("Executive Summary")
        if detail_stats:
            sections.append("Detailed Statistics")
        if charts:
            sections.append("Charts and Graphs")
        if raw_tables:
            sections.append("Raw Data Tables")
        if evidence_imgs:
            sections.append("Violation Evidence Images")

        report_request = {
            "type": date_range,
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        }

        if st.button("Generate and Download Report", width="stretch"):
            path = reports.generate_report(
                report_type=st.session_state.selected_report,
                date_range=report_request,
                format=st.session_state.export_format,
                sections=sections,
            )
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    data = f.read()
                st.download_button(
                    "Download Report",
                    data=data,
                    file_name=os.path.basename(path),
                    mime="application/octet-stream",
                    width="stretch",
                )
            else:
                st.warning("Unable to generate report with current settings.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Recent Reports</div>", unsafe_allow_html=True)
        recent = reports.get_recent_reports()
        if not recent:
            st.markdown("<div class='muted'>No reports generated yet.</div>", unsafe_allow_html=True)
        else:
            for idx, item in enumerate(recent):
                st.markdown(
                    f"""
                    <div class="recent-item">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-weight:700; font-size:12px;">{item['name']}</div>
                            <span class="badge-ready">Ready</span>
                        </div>
                        <div class="muted">{item['date']}</div>
                        <div class="muted">{item['format']} • {item['size']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if item.get("path") and os.path.exists(item["path"]):
                    with open(item["path"], "rb") as f:
                        data = f.read()
                    st.download_button(
                        "Download",
                        data=data,
                        file_name=os.path.basename(item["path"]),
                        mime="application/octet-stream",
                        key=f"recent_{idx}",
                        width="stretch",
                    )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel' style='margin-top:12px; background:#ecfeff;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Schedule Reports</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='muted'>Set up automated report generation and email delivery</div>",
            unsafe_allow_html=True,
        )
        if st.button("Configure Schedule", width="stretch"):
            st.session_state.report_schedule = {
                "enabled": True,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
