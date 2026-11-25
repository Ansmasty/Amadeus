"""
agent/core.py

Fábrica del agente ReAct con LangGraph para AMADEUS.
Soporta inyección de modelos LLM tanto locales (Ollama) como en la nube (Gemini).
"""
import os
from typing import Any, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agent.llm_factory import PROVEEDOR_LOCAL, crear_modelo_llm
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

- **Sistema de archivos**: listar, leer, crear, mover, copiar, eliminar, escribir y editar archivos
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
- **Percepción de pantalla**: leer texto/código visible en pantalla (OCR) o describir visualmente lo que se muestra (layout, gráficos, errores)
- **Asistente de programación**: revisar estado/diff/historial de git, buscar código dentro de un proyecto, ejecutar comandos de terminal (build, test, instalar dependencias)
- **Autonomía e interacción (Computer Use)**: ejecutar scripts de Python (`ejecutar_codigo_python`) para procesar formatos desconocidos (.pdf, .docx, .zip), automatizar sitios web con `playwright` y controlar aplicaciones de escritorio con `pyautogui` y `pyperclip`

## Autonomía e Interacción (Computer Use)

- **Procesamiento de Archivos:** Para leer o modificar archivos de formatos desconocidos (.pdf, .docx, .zip), escribe un script de Python usando `ejecutar_codigo_python` que importe la librería adecuada, lea el contenido y lo imprima. Si te falta una librería, usa subprocess para instalarla vía pip.
- **Automatización Web:** Si el usuario te pide realizar acciones en sitios web (ej. 'Entra a Instagram y envía un mensaje', 'Busca X en Amazon'), NO digas que no puedes. Escribe un script usando `playwright` (síncrono) para levantar Chromium, navegar al sitio, interactuar con los selectores del DOM y extraer el resultado. Ejecuta el script con `ejecutar_codigo_python`.
- **Automatización de Escritorio:** Para interactuar con aplicaciones nativas o clics físicos, usa `ejecutar_codigo_python` con la librería `pyautogui`. Puedes coordinar esto con las herramientas de `screen.py` para entender dónde están los elementos antes de mover el ratón.
- **Resiliencia:** Si un script de automatización falla (ej. selector de Playwright no encontrado, error de sintaxis), analiza el error del `stderr`, corrige el código y vuelve a intentarlo.

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

14. **Pantalla**:
    - Si el usuario pregunta por código, texto, mensajes de error o cualquier
      contenido textual visible en pantalla → `read_screen_text` (OCR, más preciso para texto).
    - Si el usuario pregunta por algo visual (layout, un gráfico, "qué app tengo abierta",
      "describe la pantalla") → `describe_screen`.
    - Si `read_screen_text` no aporta información útil, intenta con `describe_screen` como respaldo.

15. **Programación**:
    - Para ver cambios pendientes de un proyecto → `git_status`. Para el detalle → `git_diff`.
      Para el historial de commits → `git_log`.
    - Para encontrar dónde está definida o usada una función/variable/texto en el código
      → `search_in_project`.
    - Para correr tests, instalar dependencias, o cualquier comando que el usuario pida
      explícitamente en terminal → `run_terminal_command`. SIEMPRE requiere confirmación;
      nunca lo uses para acciones que no fueron pedidas explícitamente por el usuario.
    - Todas estas herramientas requieren una ruta de proyecto dentro de los directorios
      permitidos (`ALLOWED_BASE_DIRS`). Si el usuario no la da y hay una ruta usada
      recientemente en la conversación, puedes reutilizarla; si no, pídela.

16. **Modificar código (write_file / edit_file)**:
    - Prefiere SIEMPRE `edit_file` sobre `write_file` para cambios puntuales
      (arreglar un bug, renombrar algo, agregar una función): es más seguro
      porque solo toca el fragmento exacto y el usuario ve claramente qué cambia.
    - Usa `write_file` solo para reescribir un archivo completo o crear uno nuevo.
    - Antes de editar, lee el archivo con `read_file` para copiar el fragmento
      EXACTO (indentación incluida) que usarás en `find`. Un `find` que no
      coincide exactamente fallará.
    - Ambas requieren confirmación del usuario si el archivo ya existe — es esperado,
      nunca lo trates como un error, simplemente sigue el flujo normal del sentinel.

17. **Analizar o modificar "el código en pantalla"** (flujo de trabajo típico):
    a. Si el usuario adjuntó una imagen/captura desde la interfaz, ya tendrás una
       transcripción de su contenido en el mensaje — úsala como si fuera el código.
    b. Si no, usa `read_screen_text` para leer el código visible (OCR). Si el resultado
       no es suficientemente claro (texto cortado, mezcla de UI), usa `describe_screen`
       como respaldo para entender qué archivo/proyecto se está viendo.
    c. Para modificar ese código de verdad (no solo explicarlo), necesitas la ruta real
       del archivo en disco. Si el usuario no la ha dado, pregúntala directamente
       (por ejemplo: "¿en qué ruta está ese archivo?"); no la inventes ni asumas una.
    d. Una vez tengas la ruta, usa `read_file` para confirmar el contenido exacto,
       luego `edit_file` (preferido) o `write_file` para aplicar el cambio.
    e. Explica en una frase qué cambiaste y por qué, sin repetir el código completo
       salvo que el usuario lo pida.

## Idioma
Responde siempre en el idioma que use el usuario.
Si habla en español, responde en español. Si en inglés, en inglés.
"""


# ─── Funciones del Núcleo ─────────────────────────────────────────────────────


def verificar_conexion_ollama(url_base: str, modelo: str) -> Tuple[bool, str]:
    """
    Verifica que el servidor local de Ollama esté activo y que el modelo exista.

    Args:
        url_base (str): URL base donde escucha el servidor de Ollama.
        modelo (str): Nombre del modelo de Ollama requerido.

    Returns:
        Tuple[bool, str]: Tupla con (estado_conexion, mensaje_detalle).
    """
    try:
        import httpx

        respuesta = httpx.get(f"{url_base}/api/tags", timeout=3.0)
        respuesta.raise_for_status()
        datos: dict[str, Any] = respuesta.json()
        modelos_disponibles: List[str] = [
            m["name"] for m in datos.get("models", [])
        ]
        modelo_encontrado: bool = any(
            modelo in m or m in modelo for m in modelos_disponibles
        )
        if not modelo_encontrado:
            lista_modelos: str = (
                "\n  ".join(modelos_disponibles)
                if modelos_disponibles
                else "(ninguno)"
            )
            return False, (
                f"El modelo '{modelo}' no se encontró en Ollama.\n"
                f"Modelos disponibles:\n  {lista_modelos}\n\n"
                f"Ejecuta: ollama pull {modelo}"
            )
        return True, "OK"
    except ImportError:
        return True, "OK (httpx no instalado, omitiendo verificación previa)"
    except Exception as exc:
        return False, (
            f"No se puede conectar con Ollama en {url_base}.\n"
            f"Error: {exc}\n\n"
            f"Asegúrate de que Ollama esté en ejecución:\n"
            f"  Windows: ollama serve\n"
            f"  O inicia la aplicación de escritorio de Ollama"
        )


def crear_agente(
    modelo_llm: Optional[Union[BaseChatModel, str]] = None,
    proveedor: str = PROVEEDOR_LOCAL,
) -> Tuple[Any, str, str]:
    """
    Construye y retorna el agente ReAct de LangGraph con persistencia en memoria.

    Permite inyectar una instancia de `BaseChatModel` ya creada o bien indicar
    el nombre del proveedor ("local" o "gemini") para delegar su creación a la
    fábrica `crear_modelo_llm`.

    Args:
        modelo_llm (Optional[Union[BaseChatModel, str]]): Instancia del modelo
            LLM inyectado, o nombre del proveedor si se pasa como cadena.
        proveedor (str): Proveedor a utilizar cuando `modelo_llm` es None.
            Por defecto "local".

    Returns:
        Tuple[Any, str, str]: Tupla con (grafo_agente_compilado, origen_o_url, nombre_modelo).
    """
    proveedor_efectivo: str = proveedor
    instancia_llm: BaseChatModel

    if isinstance(modelo_llm, BaseChatModel):
        instancia_llm = modelo_llm
    elif isinstance(modelo_llm, str):
        proveedor_efectivo = modelo_llm.strip().lower()
        instancia_llm = crear_modelo_llm(proveedor_efectivo)
    else:
        proveedor_efectivo = proveedor.strip().lower()
        instancia_llm = crear_modelo_llm(proveedor_efectivo)

    nombre_modelo: str = str(
        getattr(instancia_llm, "model", None)
        or getattr(instancia_llm, "model_name", None)
        or (
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            if proveedor_efectivo == "gemini"
            else os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        )
    )
    origen_endpoint: str = (
        "Google Gemini API"
        if proveedor_efectivo == "gemini"
        else os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )

    almacen_memoria = MemorySaver()

    agente = create_react_agent(
        model=instancia_llm,
        tools=ALL_TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
        checkpointer=almacen_memoria,
    )

    return agente, origen_endpoint, nombre_modelo


# Alias de compatibilidad hacia atrás con módulos externos (ej. gui.py)
check_ollama_connection = verificar_conexion_ollama
create_agent = crear_agente
