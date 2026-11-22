"""
agent/llm_factory.py

Fábrica de modelos de lenguaje (LLM) para AMADEUS.
Soporta una arquitectura dual permitiendo instanciar modelos locales
mediante Ollama o modelos en la nube mediante Google Gemini.
"""
import os
from typing import Final

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

PROVEEDOR_LOCAL: Final[str] = "local"
PROVEEDOR_GEMINI: Final[str] = "gemini"
MODELO_OLLAMA_POR_DEFECTO: Final[str] = "llama3.2:3b"
URL_OLLAMA_POR_DEFECTO: Final[str] = "http://127.0.0.1:11434"
MODELO_GEMINI_POR_DEFECTO: Final[str] = "gemini-2.5-flash"
TEMPERATURA_AGENTE: Final[float] = 0.1
MAX_TOKENS_RESPUESTA: Final[int] = 4096


def crear_modelo_llm(proveedor: str) -> BaseChatModel:
    """
    Crea y retorna una instancia de modelo de chat según el proveedor indicado.

    Args:
        proveedor (str): Identificador del proveedor a utilizar ("local" para
            Ollama o "gemini" para Google Generative AI).

    Returns:
        BaseChatModel: Instancia configurada de ChatOllama o ChatGoogleGenerativeAI.

    Raises:
        ValueError: Si se selecciona "gemini" y no está configurada la variable
            de entorno GEMINI_API_KEY, o si el proveedor especificado no es válido.
    """
    proveedor_normalizado: str = proveedor.strip().lower()

    if proveedor_normalizado == PROVEEDOR_LOCAL:
        url_base: str = os.getenv(
            "OLLAMA_BASE_URL",
            URL_OLLAMA_POR_DEFECTO,
        ).strip() or URL_OLLAMA_POR_DEFECTO
        nombre_modelo: str = os.getenv(
            "OLLAMA_MODEL",
            MODELO_OLLAMA_POR_DEFECTO,
        ).strip() or MODELO_OLLAMA_POR_DEFECTO

        return ChatOllama(
            model=nombre_modelo,
            base_url=url_base,
            temperature=TEMPERATURA_AGENTE,
            num_predict=MAX_TOKENS_RESPUESTA,
        )

    if proveedor_normalizado == PROVEEDOR_GEMINI:
        clave_api: str = os.getenv("GEMINI_API_KEY", "").strip()
        if not clave_api:
            raise ValueError(
                "No se encontró la clave de API para Gemini. "
                "Por favor, configura la variable de entorno 'GEMINI_API_KEY' "
                "en tu archivo .env antes de iniciar con el modelo Gemini."
            )

        nombre_modelo_gemini: str = os.getenv(
            "GEMINI_MODEL",
            MODELO_GEMINI_POR_DEFECTO,
        ).strip() or MODELO_GEMINI_POR_DEFECTO

        return ChatGoogleGenerativeAI(
            model=nombre_modelo_gemini,
            google_api_key=clave_api,
            temperature=TEMPERATURA_AGENTE,
        )

    raise ValueError(
        f"Proveedor de LLM no soportado: '{proveedor}'. "
        f"Usa '{PROVEEDOR_LOCAL}' (Ollama) o '{PROVEEDOR_GEMINI}' (Google AI)."
    )
