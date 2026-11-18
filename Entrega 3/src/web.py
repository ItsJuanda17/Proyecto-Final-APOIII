"""
API web para demostrar el clasificador de actividades humanas.

Recibe un video (subido o de ejemplo), lo procesa con MediaPipe + el modelo
entrenado y devuelve la actividad detectada por frame, un resumen y algunos
frames anotados con el esqueleto.
"""
import base64
import os
import tempfile
import threading
from collections import Counter
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.inference import RealTimeActivityDetector

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR.parents[1] / "models" / "best_model.pkl"))
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", BASE_DIR.parents[1] / "samples"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "30"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "450"))  # ~15 s a 30 FPS
MAX_SIDE = 864  # lado largo de los videos de entrenamiento
PREVIEW_FRAMES = 6

app = FastAPI(title="Análisis de Actividades Humanas", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_detector: RealTimeActivityDetector | None = None
_lock = threading.Lock()  # un solo detector compartido; se procesa un video a la vez


def get_detector() -> RealTimeActivityDetector:
    global _detector
    if _detector is None:
        _detector = RealTimeActivityDetector(str(MODEL_PATH))
    return _detector


def _jpeg_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def analyze_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    timeline: list[dict] = []
    previews: list[tuple[int, str]] = []  # solo unas pocas miniaturas, para cuidar la memoria
    est_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES, MAX_FRAMES)
    preview_step = max(1, (est_frames - 32) // PREVIEW_FRAMES)

    with _lock:
        detector = get_detector()
        detector.reset()
        try:
            n = 0
            while n < MAX_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                scale = MAX_SIDE / max(h, w)
                if scale < 1:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                annotated, activity, confidence = detector.process_frame(frame)
                if activity:
                    timeline.append(
                        {"frame": n, "t": round(n / fps, 2), "activity": activity,
                         "confidence": round(confidence, 3)}
                    )
                    if len(previews) < PREVIEW_FRAMES and (
                        not previews or n - previews[-1][0] >= preview_step
                    ):
                        previews.append((n, _jpeg_b64(annotated)))
                n += 1
        finally:
            cap.release()
            detector.reset()

    if n == 0:
        raise HTTPException(status_code=400, detail="El video no tiene frames legibles.")

    counts = Counter(item["activity"] for item in timeline)
    total = sum(counts.values())
    summary = [
        {
            "activity": act,
            "frames": c,
            "percent": round(100 * c / total, 1),
            "avg_confidence": round(
                100 * sum(i["confidence"] for i in timeline if i["activity"] == act) / c, 1
            ),
        }
        for act, c in counts.most_common()
    ]

    # Segmentos consecutivos con la misma actividad
    segments: list[dict] = []
    for item in timeline:
        if segments and segments[-1]["activity"] == item["activity"]:
            segments[-1]["end"] = item["t"]
        else:
            segments.append({"activity": item["activity"], "start": item["t"], "end": item["t"]})

    preview_images = [{"frame": idx, "image": img} for idx, img in previews]

    return {
        "frames_processed": n,
        "fps": round(fps, 1),
        "frames_classified": total,
        "dominant_activity": summary[0]["activity"] if summary else None,
        "summary": summary,
        "segments": segments,
        "timeline": timeline,
        "previews": preview_images,
        "note": None if total else "No se detectó una persona con suficientes frames (se necesitan ~32).",
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.exists()}


@app.get("/api/samples")
def list_samples() -> dict:
    names = sorted(p.stem for p in SAMPLES_DIR.glob("*.mp4")) if SAMPLES_DIR.exists() else []
    return {"samples": names}


@app.get("/api/samples/{name}/video")
def sample_video(name: str) -> FileResponse:
    path = SAMPLES_DIR / f"{Path(name).name}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video de ejemplo no encontrado.")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/samples/{name}/analyze")
def analyze_sample(name: str) -> dict:
    path = SAMPLES_DIR / f"{Path(name).name}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video de ejemplo no encontrado.")
    return analyze_video(str(path))


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"El video supera {MAX_UPLOAD_MB} MB.")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return await run_in_threadpool(analyze_video, tmp_path)
    finally:
        os.unlink(tmp_path)
