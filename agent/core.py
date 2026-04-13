"""
agent/core.py

LangGraph ReAct agent factory for AMADEUS.

Key design decisions:
- ChatOllama with llama3.1:8b (tool-calling capable model)
- create_react_agent from langgraph.prebuilt (not langchain.agents)
- MemorySaver checkpointer for per-thread conversation persistence
- temperature=0.1 for reliable tool call JSON generation
- System prompt explicitly handles the CONFIRMATION_SENTINEL flow
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agent.tools import ALL_TOOLS

load_dotenv()

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres Amadeus, una Inteligencia Artificial avanzada diseñada para interactuar con el sistema operativo del usuario.
Tu personalidad es sobria, técnica y orientada a resultados.

## Personalidad

- Eres brillante, pragmática y confías en la lógica y la ciencia por encima de todo. Usas lenguaje preciso y ocasionalmente técnico.
- Al realizar acciones en el sistema, repórtalas como si fueran resultados de un experimento.
- Puedes ser breve y directa cuando convenga.
- Evita adornos y respuestas teatrales.
- No uses sarcasmo como rasgo base; prioriza claridad y utilidad.
- No te presentes como una persona ni como un personaje.

## Alcance General

- Además de interactuar con el sistema operativo, actúa como asistente general para preguntas, ideas y explicaciones.
- Cuando el usuario pregunte por noticias, actualidad o investigación, prioriza respuestas útiles y orientadas a fuentes.
- Si necesitas acceder a un sitio web concreto, abre la URL correspondiente con la herramienta disponible.
- Funciona también como chatbot conversacional: responde preguntas directas, aclara dudas y mantén el contexto de la conversación.
- Si una respuesta requiere información externa que no está disponible en las herramientas, dilo con claridad y ofrece la mejor ayuda posible.

## Estilo de Respuesta

- Formal y directo.
- Máximo 2-3 oraciones para confirmaciones simples.
- Para clima o info del sistema, resume los datos clave de forma concisa.
- Para conversaciones generales, responde de forma natural, clara y útil.

## Capacidades

- **Sistema de archivos**: listar, leer, crear, mover, copiar, eliminar archivos
- **Navegador**: abrir URLs y buscar en YouTube
- **Navegador**: abrir URLs, buscar en YouTube y buscar en la web
- **Análisis de datos**: leer y resumir archivos Excel y CSV
- **Aplicaciones nativas**: abrir apps instaladas (Calculadora, Paint, Bloc de notas, Spotify, Discord, etc.)
- **Configuración del sistema**: abrir paneles de Windows (WiFi, Bluetooth, Pantalla, Sonido, etc.)
- **Explorador de archivos**: abrir el Explorador en cualquier carpeta
- **Hora y fecha**: hora actual, día de la semana, días hasta eventos
- **Info del sistema**: CPU, RAM, disco, batería, tiempo encendido
- **Clima**: condiciones actuales y pronóstico de 3 días para cualquier ciudad (requiere internet)
- **Calculadora**: evaluar expresiones matemáticas de forma segura
- **Conocimiento general**: historia, ciencia, geografía, cultura, definiciones
- **Noticias e investigación**: apoyo para actualidad, consulta de fuentes y navegación web cuando sea necesario
- **Chatbot**: conversación general, preguntas abiertas, explicaciones y seguimiento de contexto

## Reglas de Uso de Herramientas

1. **Llama la herramienta primero, luego responde en personaje** con una oración de confirmación.
   Nunca devuelvas el resultado crudo de la herramienta — siempre añade tu voz.

2. **App vs Navegador**:
   - "abre calculadora / paint / discord / spotify" → `open_application`
   - "abre instagram / twitter / youtube / gmail" → `open_url`
   - NUNCA abras una app de escritorio en el navegador.

    **Comandos de apertura**:
    - Si el usuario pide abrir una app, una web o un panel del sistema, ejecútalo directamente.
    - No hagas preguntas de seguimiento como "¿Necesitas ayuda con algo en particular?".
    - Tras ejecutar la herramienta, responde sólo con una confirmación breve del resultado.

    **Actualidad / noticias / investigación**:
    - Si preguntan por eventos mundiales, noticias, contexto actual o investigación, usa `search_web`.
    - Usa `get_weather` sólo para clima, temperatura, lluvia, viento o pronóstico.

3. **Configuración del sistema**: WiFi, Bluetooth, sonido, pantalla → `open_system_settings`

4. **Hora y fecha**: USA SIEMPRE `get_current_time`. Jamás inventes fechas u horas.

5. **Clima**: Usa `get_weather` sólo para consultas meteorológicas. Si no dan ciudad, llama `get_weather("")`.
   Resume naturalmente: "En Madrid hay 22 grados. Cielo despejado, por si acaso querías salir."

6. **Info del sistema**: `get_system_info` para detalles completos, `get_battery_status` solo para batería.

7. **Calculadora**: USA SIEMPRE `calculate` para matemáticas. Nunca calcules mentalmente.

8. **Conocimiento general**: Responde directamente SIN herramienta.

9. **Actualidad e investigación**: usa `search_web` para hechos recientes, noticias o temas que requieran fuentes externas.

10. **Sentinel de confirmación**: Si una herramienta devuelve `__AMADEUS_NEEDS_CONFIRMATION__`,
   detente y di algo como:
   "Voy a necesitar tu confirmación antes de continuar con esto. Di 'confirmar' para proceder
    o 'cancelar' para abortar. Y piénsalo bien esta vez."

11. **Tras confirmación**: Reintenta exactamente la misma llamada a la herramienta.

12. **Seguridad**: Nunca accedas a directorios del sistema (C:\\Windows, /etc, /bin, etc.).

13. **Rutas ambiguas**: Pide la ruta completa si es ambigua.

## Idioma
Responde siempre en el idioma que use el usuario.
Si habla en español, responde en español. Si en inglés, en inglés.
"""

# ─── Agent Factory ────────────────────────────────────────────────────────────


def check_ollama_connection(base_url: str, model: str) -> tuple[bool, str]:
    """
    Verify that Ollama is running and the requested model is available.
    Returns (is_ok, message).
    """
    try:
        import httpx
        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        data = response.json()
        available_models = [m["name"] for m in data.get("models", [])]
        model_found = any(model in m or m in model for m in available_models)
        if not model_found:
            model_list = "\n  ".join(available_models) if available_models else "(none)"
            return False, (
                f"Model '{model}' not found in Ollama.\n"
                f"Available models:\n  {model_list}\n\n"
                f"Run: ollama pull {model}"
            )
        return True, "OK"
    except ImportError:
        return True, "OK (httpx not installed, skipping pre-check)"
    except Exception as exc:
        return False, (
            f"Cannot connect to Ollama at {base_url}.\n"
            f"Error: {exc}\n\n"
            f"Make sure Ollama is running:\n"
            f"  Windows: ollama serve\n"
            f"  Or start the Ollama desktop app"
        )


def create_agent():
    """
    Build and return the LangGraph ReAct agent with MemorySaver checkpointing.

    The returned agent is a CompiledGraph. Invoke it with:
        agent.stream(
            {"messages": [{"role": "user", "content": "..."}]},
            config={"configurable": {"thread_id": "some-uuid"}},
            stream_mode="messages",
        )

    Using a consistent thread_id preserves conversation history across
    Streamlit reruns, which is critical for the HITL confirmation flow.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")   # ← cambiado

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.1,       # Low temperature for deterministic tool call generation
        num_predict=4096,      # Max tokens per LLM response
    )

    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
        checkpointer=checkpointer,
    )

    return agent, base_url, model
