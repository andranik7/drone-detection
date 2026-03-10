"""Gradio webapp for drone detection inference."""

import io
import logging
import os

import gradio as gr
import requests
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def detect_drones(image: Image.Image, confidence: float) -> Image.Image:
    """Send image to API and draw bounding boxes on the result."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    logger.info("Sending image to API (confidence=%.2f)", confidence)
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("image.png", buf, "image/png")},
        params={"confidence": confidence},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    logger.info("API returned %d detections", data["count"])

    draw = ImageDraw.Draw(image)
    for det in data["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.text((x1, y1 - 15), label, fill="red")

    return image


demo = gr.Interface(
    fn=detect_drones,
    inputs=[
        gr.Image(type="pil", label="Upload an image"),
        gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=gr.Image(type="pil", label="Detection result"),
    title="Drone Detection",
    description="Upload an image to detect drones using YOLOv8.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
