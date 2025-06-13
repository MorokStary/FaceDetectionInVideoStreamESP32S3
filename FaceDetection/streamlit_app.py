"""Simple Streamlit interface for adding face embeddings and running
face recognition on uploaded video files.

The app reuses the YOLOv8 face detector and Facenet based
embedding model from the existing scripts.
"""

from __future__ import annotations

import os
import pickle
from tempfile import NamedTemporaryFile
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO


@st.cache_resource
def load_models():
    """Load detection and recognition models."""
    detector = YOLO("detection/weights/best.pt")
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return detector, mtcnn, resnet


def load_embeddings() -> Dict[str, List[np.ndarray]]:
    """Load known embeddings from disk."""
    path = os.path.join("known_embeddings.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_embeddings(data: Dict[str, List[np.ndarray]]) -> None:
    """Persist embeddings to disk."""
    with open("known_embeddings.pkl", "wb") as f:
        pickle.dump(data, f)


def compare_embeddings(embedding: np.ndarray, known: Dict[str, List[np.ndarray]]) -> str:
    """Return the name of the closest embedding if within threshold."""
    threshold = 0.2
    min_dist = float("inf")
    match = "Unknown"

    for name, embeds in known.items():
        for known_embedding in embeds:
            dist = np.linalg.norm(embedding - known_embedding)
            if dist < min_dist:
                min_dist = dist
                if dist < threshold:
                    match = name
    return match


def compute_embeddings(images: List[Image.Image], detector: YOLO, mtcnn: MTCNN, resnet: InceptionResnetV1) -> List[np.ndarray]:
    """Create embeddings from a list of PIL images."""
    embeddings: List[np.ndarray] = []
    for img in images:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = detector(frame)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            face = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = mtcnn(face_rgb)
            if tensor is not None:
                emb = (
                    resnet(tensor.unsqueeze(0))
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )
                embeddings.append(emb)
    return embeddings


def recognise_in_video(path: str, detector: YOLO, mtcnn: MTCNN, resnet: InceptionResnetV1, known: Dict[str, List[np.ndarray]]) -> str:
    """Process a video file and save annotated copy."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = cv2.VideoWriter(
        output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector(frame)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            face = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = mtcnn(face_rgb)
            if tensor is not None:
                emb = (
                    resnet(tensor.unsqueeze(0))
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )
                name = compare_embeddings(emb, known)
            else:
                name = "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        writer.write(frame)

    writer.release()
    cap.release()
    return output.name


def main() -> None:
    st.title("Face recognition demo")

    detector, mtcnn, resnet = load_models()
    embeddings = load_embeddings()

    st.sidebar.header("Add new user")
    name = st.sidebar.text_input("Name")
    images = st.sidebar.file_uploader(
        "Face images", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
    )
    if st.sidebar.button("Add user"):
        if not name or not images:
            st.sidebar.error("Provide name and at least one image")
        else:
            pics = [Image.open(img) for img in images]
            embs = compute_embeddings(pics, detector, mtcnn, resnet)
            if embs:
                embeddings[name] = embs
                save_embeddings(embeddings)
                st.sidebar.success(f"Embeddings saved for {name}")
            else:
                st.sidebar.error("No faces detected in images")

    st.header("Recognise faces in video")
    video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if video is not None:
        tmp = NamedTemporaryFile(delete=False)
        tmp.write(video.read())
        tmp.close()
        result_path = recognise_in_video(tmp.name, detector, mtcnn, resnet, embeddings)
        if result_path:
            st.video(result_path)
        else:
            st.error("Could not read video")


if __name__ == "__main__":
    main()

