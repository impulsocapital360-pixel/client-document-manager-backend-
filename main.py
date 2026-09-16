"""
Client Document Manager — Servicio de procesamiento de imágenes (Fase 3)

Servicio SIN ESTADO: no guarda nada en disco, no tiene credenciales de
Supabase ni de GoHighLevel. Solo recibe la URL firmada de una imagen,
la procesa en memoria, y devuelve el resultado. Nunca registra en logs
el contenido de las imágenes.
"""

import base64
import os
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import process_auto, process_manual

# ------------------------------------------------------------------
# Falla al arrancar si falta cualquier variable de seguridad, en vez
# de arrancar "abierto" en silencio cuando alguien olvida configurarla.
# ------------------------------------------------------------------
API_KEYS = {k.strip() for k in os.environ.get("PROCESSING_API_KEYS", "").split(",") if k.strip()}
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN")
ALLOWED_IMAGE_HOST = os.environ.get("ALLOWED_IMAGE_HOST")  # ej: xxxxx.supabase.co

_missing = [
    name
    for name, value in [
        ("PROCESSING_API_KEYS", API_KEYS),
        ("ALLOWED_ORIGIN", ALLOWED_ORIGIN),
        ("ALLOWED_IMAGE_HOST", ALLOWED_IMAGE_HOST),
    ]
    if not value
]
if _missing:
    raise RuntimeError(
        "Faltan variables de entorno obligatorias: " + ", ".join(_missing) +
        ". El servicio se niega a arrancar sin ellas (fail-closed) en vez de "
        "quedar abierto por accidente."
    )

app = FastAPI(title="Client Document Manager — Processing Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def check_api_key(x_api_key: str | None):
    # x_api_key debe estar en el conjunto de llaves válidas — cada quien
    # (tu frontend, un tercero) usa la suya, y se puede revocar una sola
    # quitándola de PROCESSING_API_KEYS sin tocar las demás.
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="unauthorized")


def check_image_url(url: str):
    """Bloquea SSRF: solo se permite descargar imágenes del propio
    proyecto de Supabase, nunca una URL arbitraria que mande quien sea."""
    host = urlparse(url).hostname or ""
    if host != ALLOWED_IMAGE_HOST:
        raise HTTPException(status_code=400, detail="image_url_host_not_allowed")


class ProcessRequest(BaseModel):
    image_url: str


class ProcessManualRequest(BaseModel):
    image_url: str
    corners: list[list[float]]  # 4 puntos [[x,y], [x,y], [x,y], [x,y]]


async def download_image(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="could_not_download_image")
        return resp.content


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_endpoint(body: ProcessRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    check_image_url(body.image_url)
    image_bytes = await download_image(body.image_url)
    result = process_auto(image_bytes)

    if result["manual_required"]:
        return {
            "manual_required": True,
            "confidence": result["confidence"],
            "message": result["message"],
        }

    return {
        "manual_required": False,
        "confidence": result["confidence"],
        "quality_flags": result["quality_flags"],
        "detected_corners": result["detected_corners"],
        "processed_image_base64": base64.b64encode(result["processed_image_bytes"]).decode("ascii"),
    }


@app.post("/process-manual")
async def process_manual_endpoint(body: ProcessManualRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    check_image_url(body.image_url)
    if len(body.corners) != 4:
        raise HTTPException(status_code=400, detail="exactly_4_corners_required")

    image_bytes = await download_image(body.image_url)
    result = process_manual(image_bytes, body.corners)

    return {
        "quality_flags": result["quality_flags"],
        "processed_image_base64": base64.b64encode(result["processed_image_bytes"]).decode("ascii"),
    }
