import os

import pandas as pd
import streamlit as st

from backend.process_video import process_video


def show():
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
        }

    st.title("Upload & Process Traffic Video")

    uploaded_video = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov"],
    )

    col1, col2 = st.columns(2)
    run_btn = col1.button("Run Analysis", width="stretch")
    reset_btn = col2.button("Reset", width="stretch")

    if reset_btn:
        st.session_state.processed = False
        st.session_state.df = None
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
        }
        st.success("System reset successfully.")

    if run_btn and not uploaded_video:
        st.warning("Please upload a video file before running analysis.")
        return

    if run_btn and uploaded_video:
        os.makedirs("uploads", exist_ok=True)
        safe_name = os.path.basename(uploaded_video.name)
        video_path = os.path.join("uploads", safe_name)

        with open(video_path, "wb") as file_obj:
            file_obj.write(uploaded_video.getbuffer())

        with st.spinner("Processing video..."):
            run_dir = process_video(video_path)

        csv_path = os.path.join(run_dir, "traffic_log.csv")
        df = pd.read_csv(csv_path)

        st.session_state.df = df

        if df.empty:
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
            }
        else:
            last_row = df.iloc[-1]
            st.session_state.metrics = {
                "total_vehicles": int(last_row.get("total_vehicles", 0)),
                "queue_count": int(last_row.get("queue_count", 0)),
                "total_violations": int(last_row.get("total_violations", 0)),
                "red_light_violations": int(last_row.get("red_light_violations", 0)),
                "rash_driving": int(last_row.get("rash_driving", 0)),
                "no_helmet_violations": int(last_row.get("no_helmet_violations", 0)),
                "mobile_usage_violations": int(last_row.get("mobile_usage_violations", 0)),
                "triple_riding_violations": int(last_row.get("triple_riding_violations", 0)),
                "heavy_load_violations": int(last_row.get("heavy_load_violations", 0)),
            }

        st.session_state.processed = True
        st.success("Video processed successfully.")
