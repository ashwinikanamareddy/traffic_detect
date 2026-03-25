import streamlit as st
import importlib

from backend.auth import init_db


st.set_page_config(
    page_title="Traffic Intelligence System",
    page_icon="\U0001F6A6",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

if "processed" not in st.session_state:
    st.session_state.processed = False
if "df" not in st.session_state:
    st.session_state.df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_vehicles": 0,
        "queue_count": 0,
        "total_violations": 0,
        "red_light_violations": 0,
        "rash_driving": 0,
        "no_helmet_violations": 0,
        "mobile_usage_violations": 0,
        "triple_riding_violations": 0,
        "heavy_load_violations": 0,
        "autos": 0,
    }
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "vehicle_counts" not in st.session_state:
    st.session_state.vehicle_counts = {
        "cars": 0,
        "bikes": 0,
        "buses": 0,
        "trucks": 0,
        "autos": 0,
    }
if "live_paused" not in st.session_state:
    st.session_state.live_paused = False
if "live_video_path" not in st.session_state:
    st.session_state.live_video_path = None
if "live_frame_index" not in st.session_state:
    st.session_state.live_frame_index = 0
if "live_event_log" not in st.session_state:
    st.session_state.live_event_log = []
if "violations" not in st.session_state:
    st.session_state.violations = []
if "selected_violation" not in st.session_state:
    st.session_state.selected_violation = None


def _load_view_module(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        st.error(f"Failed to load module `{module_name}`.")
        st.exception(exc)
        return None


if not st.session_state.logged_in:
    login_module = _load_view_module("views.login")
    if login_module and hasattr(login_module, "show"):
        login_module.show()
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: #f4f7fb;
            border-right: 1px solid #e6edf6;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 16px;
        }
        .sb-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 8px 14px 8px;
        }
        .sb-logo {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            background-size: 36px 36px;
            background-position: center;
            background-repeat: no-repeat;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><circle cx='32' cy='32' r='29' fill='none' stroke='%23111827' stroke-width='2.6'/><path d='M19 44 32 40 45 44 41 52H23z' fill='%23111827'/><path d='M28 30h8c2.4 0 4.5 1.4 5.6 3.5l1.8 3.5v7.2a2 2 0 0 1-2 2h-1.5a2 2 0 0 1-2-2v-1h-12v1a2 2 0 0 1-2 2h-1.5a2 2 0 0 1-2-2V37l1.8-3.5A6.2 6.2 0 0 1 28 30z' fill='none' stroke='%23111827' stroke-width='2'/><path d='M21 41h5m17 0h-5' stroke='%23111827' stroke-width='2' stroke-linecap='round'/><path d='M18 19h8l1.2 1.2-2.6 1.8h-1.4l-5.2-1.8z' fill='%23111827'/><path d='M16 20.5h7.2c1.1 0 2.2.3 3.1.9l.8.5-2.1 3.2-.7-.5a3.3 3.3 0 0 0-2-.6H16z' fill='%23111827'/><rect x='43.2' y='17.8' width='6' height='15.6' rx='1.5' fill='%23111827'/><circle cx='46.2' cy='20.6' r='1.1' fill='white'/><circle cx='46.2' cy='25.6' r='1.1' fill='white'/><circle cx='46.2' cy='30.6' r='1.1' fill='white'/><path d='M20 18a20 20 0 0 1 24 0' fill='none' stroke='%23111827' stroke-width='2.2' stroke-linecap='round'/></svg>");
        }
        .sb-title {
            font-size: 15px;
            font-weight: 800;
            color: #0f172a;
        }
        .sb-sub {
            font-size: 12px;
            color: #64748b;
        }
        [data-testid="stSidebar"] .stRadio > div { gap: 6px; }
        [data-testid="stSidebar"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
            margin: 0;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: #ffffff;
            border: 1px solid #e6edf4;
            border-radius: 12px;
            padding: 11px 13px;
            margin: 0 0 8px 0;
            box-shadow: 0 2px 8px rgba(15,23,42,0.03);
            transition: all 0.2s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: #e7fbf8;
            border-color: #9cefe4;
            box-shadow: inset 0 0 0 1px #9cefe4, 0 6px 14px rgba(20, 184, 166, 0.10);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label > div:last-child {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 650;
            color: #0f172a;
            font-size: 15px;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:first-child > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='1.5' y='1.5' width='21' height='21' rx='4' fill='%23f3f4f6'/><rect x='6' y='14' width='3' height='5' rx='0.8' fill='%231f2937'/><rect x='10.5' y='12' width='3' height='7' rx='0.8' fill='%231f2937'/><rect x='15' y='9' width='3' height='10' rx='0.8' fill='%231f2937'/><path d='M8.2 10.8c2.7-.3 5.1-1.8 7.2-4.5' fill='none' stroke='%231f2937' stroke-width='1.3' stroke-linecap='round'/><circle cx='8.2' cy='10.8' r='0.9' fill='%231f2937'/><circle cx='15.4' cy='6.3' r='0.9' fill='%231f2937'/><path d='M7.8 4.4a3.2 3.2 0 1 1-2.3 5.4' fill='none' stroke='%231f2937' stroke-width='1.3' stroke-linecap='round'/></svg>");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(2) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='2' y='5' width='14' height='14' rx='1.8' fill='none' stroke='%23334155' stroke-width='1.8'/><path d='M10 10v4l3-2z' fill='%23334155'/><path d='M16 9l5-3v12l-5-3z' fill='none' stroke='%23334155' stroke-width='1.8' stroke-linejoin='round'/></svg>");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M3 21h18' stroke='%23334155' stroke-width='1.8' stroke-linecap='round'/><rect x='5' y='12' width='3.5' height='7' rx='0.8' fill='none' stroke='%23334155' stroke-width='1.8'/><rect x='10.5' y='9' width='3.5' height='10' rx='0.8' fill='%23334155'/><rect x='16' y='5' width='3.5' height='14' rx='0.8' fill='none' stroke='%23334155' stroke-width='1.8'/></svg>\");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(4) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M12 4 21 20H3z' fill='none' stroke='%23334155' stroke-width='1.8' stroke-linejoin='round'/><path d='M12 9v5' stroke='%23334155' stroke-width='1.8' stroke-linecap='round'/><circle cx='12' cy='17' r='1' fill='%23334155'/></svg>\");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M4 13h16l-1.2-4a2 2 0 0 0-1.9-1.4H7.1A2 2 0 0 0 5.2 9z' fill='none' stroke='%23334155' stroke-width='1.8'/><path d='M4 13v4h2m14-4v4h-2' stroke='%23334155' stroke-width='1.8' stroke-linecap='round'/><circle cx='8' cy='17' r='1.6' fill='%23334155'/><circle cx='16' cy='17' r='1.6' fill='%23334155'/></svg>\");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(6) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M5 3h10l4 4v14H5z' fill='none' stroke='%23334155' stroke-width='1.8' stroke-linejoin='round'/><path d='M15 3v4h4' fill='none' stroke='%23334155' stroke-width='1.8'/><path d='M8 10h6M8 13h6' stroke='%23334155' stroke-width='1.6' stroke-linecap='round'/><path d='M10 19h8' stroke='%23334155' stroke-width='1.8' stroke-linecap='round'/><path d='m16 16 3 3-3 3' fill='%23334155'/></svg>\");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M3 6h8l1 2h9v11H3z' fill='none' stroke='%23334155' stroke-width='1.8' stroke-linejoin='round'/><rect x='6.5' y='11' width='8' height='5' rx='1' fill='none' stroke='%23334155' stroke-width='1.6'/><path d='M18 11.5 21 6.5' stroke='%23334155' stroke-width='1.6' stroke-linecap='round'/><path d='M17 17.5h5' stroke='%23334155' stroke-width='1.6' stroke-linecap='round'/></svg>\");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(8) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M4 4v16h10' fill='none' stroke='%23334155' stroke-width='1.8' stroke-linecap='round'/><rect x='6' y='13' width='2.8' height='5' rx='0.6' fill='%23334155'/><rect x='10' y='10' width='2.8' height='8' rx='0.6' fill='%23334155'/><rect x='14' y='7' width='2.8' height='11' rx='0.6' fill='%23334155'/><circle cx='17.5' cy='14.5' r='4.2' fill='none' stroke='%23334155' stroke-width='1.8'/><path d='m20.3 17.3 2 2' stroke='%23334155' stroke-width='1.8' stroke-linecap='round'/></svg>\");
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(9) > div:last-child::before {
            content: "";
            width: 20px;
            height: 20px;
            border-radius: 5px;
            flex: 0 0 20px;
            background-size: 20px 20px;
            background-repeat: no-repeat;
            background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='m12 3 1 .6 1.2-.3 1 .9.3 1.2 1 .6v1.2l.9.9-.3 1.2.6 1-.6 1 .3 1.2-.9.9v1.2l-1 .6-.3 1.2-1 .9-1.2-.3-1 .6-1-.6-1.2.3-1-.9-.3-1.2-1-.6v-1.2l-.9-.9.3-1.2-.6-1 .6-1-.3-1.2.9-.9V6.4l1-.6.3-1.2 1-.9 1.2.3z' fill='none' stroke='%23334155' stroke-width='1.4'/><circle cx='12' cy='12' r='3.8' fill='none' stroke='%23334155' stroke-width='1.4'/><path d='m10.6 12 1 1 2-2' fill='none' stroke='%23334155' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/></svg>\");
        }
        [data-testid="stSidebar"] input[type="radio"] {
            position: absolute !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            pointer-events: none !important;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child,
        [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child * {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label svg {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-logo"></div>
            <div>
                <div class="sb-title">TrafficAI</div>
                <div class="sb-sub">Intelligence System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu_items = [
        ("Dashboard Overview", "Dashboard", ""),
        ("Manual Config + Analysis", "ManualDashboard", ""),
        ("Live Video Feed", "LiveFeed", ""),
        ("Queue Analytics", "Queue", ""),
        ("Violation Detection", "Violations", ""),
        ("Vehicle Statistics", "Statistics", ""),
        ("Export Reports", "Export", ""),
        ("Violation Evidence", "ViolationEvidence", ""),
        ("Trends & Insights", "Insights", ""),
        ("System Health", "Health", ""),
    ]

    labels = [label if not icon else f"{icon} {label}" for label, key, icon in menu_items]
    keys = [key for _, key, _ in menu_items]
    current_key = st.session_state.get("page", "Dashboard")
    default_index = keys.index(current_key) if current_key in keys else 0

    selection = st.radio("Navigation", labels, index=default_index, label_visibility="collapsed")
    sel_index = labels.index(selection)
    st.session_state.page = keys[sel_index]

    # Logout control removed from sidebar per updated UI requirement.

page = st.session_state.get("page", "Dashboard")

page_to_module = {
    "Dashboard": "views.dashboard",
    "ManualDashboard": "views.manual_dashboard",
    "Upload": "views.live_feed",
    "LiveFeed": "views.live_feed",
    "Queue": "views.queue",
    "Violations": "views.violations",
    "ViolationEvidence": "views.violation_evidence",
    "Statistics": "views.statistics",
    "Export": "views.export",
    "Insights": "views.insights",
    "Health": "views.system_health",
}

module_name = page_to_module.get(page, "views.dashboard")
page_module = _load_view_module(module_name)
if page_module and hasattr(page_module, "show"):
    page_module.show()
