import os
from datetime import datetime
from typing import Optional

import cv2


def save_uploaded_video(uploaded_file, upload_dir: str = "uploads") -> str:
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = os.path.basename(uploaded_file.name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(upload_dir, f"{ts}_{safe_name}")
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def get_first_frame(video_path: str) -> Optional[object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame
