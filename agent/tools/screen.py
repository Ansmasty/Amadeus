"""
agent/tools/screen.py

Herramientas de percepción visual para AMADEUS:
  - read_screen_text: captura la pantalla y extrae texto con OCR (rápido,
    ideal para leer código, mensajes de error o cualquier texto visible).
  - describe_screen: captura la pantalla y la envía a un modelo multimodal
    local de Ollama para describir layout, UI, gráficos o errores visuales.

Ambas herramientas son de solo lectura (no requieren confirmación HITL) y
nunca guardan la captura en disco: todo se procesa en memoria.
"""
import base64
import io
import os

from langchain_core.tools import tool

_MAX_OCR_CHARS = 6000


def _grab_screenshot():
    """Captura la pantalla completa y devuelve un objeto PIL.Image."""
    from PIL import ImageGrab
    return ImageGrab.grab()


def _screenshot_to_base64_png(image, max_dimension: int = 1400) -> str:
    """Redimensiona (si hace falta) y codifica la imagen como PNG base64."""
    w, h = image.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── OCR: lectura de texto/código en pantalla ─────────────────────────────────


@tool
def read_screen_text() -> str:
    """
    Captura la pantalla actual y extrae todo el texto visible usando OCR.
    Úsala cuando el usuario pregunte por código, mensajes de error, o
    cualquier texto que esté viendo en pantalla ("¿qué dice este error?",
    "analiza el código que tengo abierto", "lee esto de la pantalla").
    No requiere abrir ni mover ninguna ventana; lee lo que ya está visible.
    """
    try:
        import pytesseract
    except ImportError:
        return (
            "Error: falta la librería 'pytesseract'. Instálala con "
            "'pip install pytesseract Pillow' y asegúrate de tener "
            "Tesseract OCR instalado en el sistema."
        )

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        image = _grab_screenshot()
    except Exception as exc:
        return f"Error al capturar la pantalla: {exc}"

    try:
        text = pytesseract.image_to_string(image, lang=os.getenv("OCR_LANG", "spa+eng"))
    except Exception as exc:
        return (
            f"Error al ejecutar OCR: {exc}\n"
            "Verifica que Tesseract OCR esté instalado y en el PATH, o define "
            "TESSERACT_CMD en tu .env con la ruta al ejecutable."
        )

    text = text.strip()
    if not text:
        return "No se detectó texto legible en la pantalla actual."

    if len(text) > _MAX_OCR_CHARS:
        text = text[:_MAX_OCR_CHARS] + "\n\n[...texto truncado por longitud...]"

    return f"Texto detectado en pantalla:\n\n{text}"


# ─── Visión: descripción de la pantalla con modelo multimodal ────────────────


@tool
def describe_screen(question: str = "") -> str:
    """
    Captura la pantalla actual y la analiza con un modelo de IA con visión
    (local, vía Ollama). Úsala cuando el usuario pida entender algo visual
    que el OCR no puede capturar bien: layout de una interfaz, un gráfico,
    un diagrama, colores, o para responder preguntas generales sobre lo que
    se ve en pantalla ("¿qué estoy viendo?", "describe esta ventana").
    Si el usuario no da una pregunta específica, describe la pantalla en general.
    Requiere tener un modelo multimodal descargado en Ollama (ver .env: OLLAMA_VISION_MODEL).
    """
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    vision_model = os.getenv("OLLAMA_VISION_MODEL", "llava")

    try:
        image = _grab_screenshot()
    except Exception as exc:
        return f"Error al capturar la pantalla: {exc}"

    try:
        img_b64 = _screenshot_to_base64_png(image)
    except Exception as exc:
        return f"Error al procesar la captura: {exc}"

    prompt_text = question.strip() or (
        "Describe brevemente lo que se ve en esta captura de pantalla: "
        "qué aplicación o ventana está abierta, y cualquier elemento relevante "
        "(código, errores, texto importante, gráficos)."
    )

    try:
        vlm = ChatOllama(model=vision_model, base_url=base_url, temperature=0.2)
        message = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
        ])
        response = vlm.invoke([message])
        result = getattr(response, "content", "")
        if isinstance(result, list):
            result = " ".join(
                b.get("text", "") for b in result if isinstance(b, dict)
            )
        result = (result or "").strip()
        if not result:
            return "El modelo de visión no devolvió una descripción."
        return result
    except Exception as exc:
        return (
            f"Error al analizar la pantalla con el modelo de visión: {exc}\n"
            f"Verifica que el modelo '{vision_model}' esté descargado "
            f"(ejecuta: ollama pull {vision_model}) y que Ollama esté corriendo."
        )
