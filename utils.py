import cv2
import base64
import requests
import logging
import numpy as np

IMGBB_API_KEY = "4e6fcd6b91a0e2cea4ae0c98e8e62ed3"

def upload_frame_to_imgbb(frame: np.ndarray, api_key: str = IMGBB_API_KEY) -> str:
    """Uploads an image frame (as numpy array) to imgbb and returns the image URL."""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        b64_str = base64.b64encode(buffer).decode()
        payload = {
            "key": api_key,
            "image": b64_str
        }
        resp = requests.post("https://api.imgbb.com/1/upload", data=payload)
        resp.raise_for_status()
        return resp.json()["data"]["url"]
    except Exception as e:
        logging.error(f"imgbb upload failed: {e}")
        return None
