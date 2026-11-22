"""
main.py

AMADEUS - Asistente de IA con arquitectura LLM dual (Ollama local / Gemini en la nube)
y soporte de entrada/salida por voz.
Ejecutar con: run.bat  o  python main.py

Modos disponibles:
  - Texto:      python main.py
  - Voz:        python main.py --voice
  - Voz+Texto:  python main.py --voice --text-fallback
"""
import argparse
import logging
import os
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Set

from colorlog import ColoredFormatter
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


# ─── Configuración de Logging ─────────────────────────────────────────────────


def configurar_registro_logs() -> None:
    """Configura el formateador de logs con colores en consola."""
    manejador = logging.StreamHandler()
    manejador.setFormatter(
        ColoredFormatter(
            "%(log_color)s[%(levelname)s]%(reset)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    logger_raiz = logging.getLogger()
    logger_raiz.setLevel(logging.WARNING)  # Silencia logs de LangChain/httpx
    logging.getLogger("amadeus").setLevel(logging.INFO)
    logger_raiz.addHandler(manejador)


configurar_registro_logs()
registro_log = logging.getLogger("amadeus")

# ─── Imports del agente ───────────────────────────────────────────────────────

from agent.core import crear_agente, verificar_conexion_ollama  # noqa: E402
from agent.hitl import (  # noqa: E402
    PendingAction,
    cancel,
    confirm,
    get_pending_action,
    reset_all,
)
from agent.llm_factory import (  # noqa: E402
    PROVEEDOR_GEMINI,
    PROVEEDOR_LOCAL,
    crear_modelo_llm,
)
from agent.tools.browser import open_url  # noqa: E402
from agent.tools.system_apps import (  # noqa: E402
    open_application,
    open_file_explorer,
    open_system_settings,
)

# ─── Constantes ───────────────────────────────────────────────────────────────

BANNER: str = r"""
    ___    __  ______    ____  _______  _______
   /   |  /  |/ /   \  / __ \/ ____/ / / ___/
  / /| | / /|_/ / /| | / / / / __/ / / /\__ \ 
 / ___ |/ /  / / ___ |/ /_/ / /___/ /_/ /__/ / 
/_/  |_|/_/  /_/_/  |_/_____/_____\____/____/  

        Asistente de IA — Arquitectura Dual (Local / Nube)
"""

COMANDOS_SALIDA: Set[str] = {"salir", "exit", "quit", "adiós", "adios", "bye"}
FRASES_CANCELACION: Set[str] = {"cancelar", "cancel", "no", "cancela"}
FRASES_CONFIRMACION: Set[str] = {
    "confirmar",
    "confirm",
    "sí",
    "si",
    "yes",
    "adelante",
    "procede",
}

MAPA_APLICACIONES_WEB: Dict[str, str] = {
    "instagram": "https://www.instagram.com",
    "insta": "https://www.instagram.com",
    "youtube": "https://www.youtube.com",
    "youtube music": "https://music.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "whatsapp web": "https://web.whatsapp.com",
    "reddit": "https://www.reddit.com",
    "linkedin": "https://www.linkedin.com",
    "discord web": "https://discord.com/app",
    "spotify web": "https://open.spotify.com",
}

PALABRAS_CLAVE_PANEL_SISTEMA: Set[str] = {
    "wifi",
    "bluetooth",
    "pantalla",
    "display",
    "sonido",
    "sound",
    "privacidad",
    "updates",
    "actualizaciones",
    "apps",
    "aplicaciones",
    "explorador",
    "explorer",
    "file explorer",
}


# ─── Menú CLI de Selección de Proveedor LLM ──────────────────────────────────


def seleccionar_proveedor_llm() -> str:
    """
    Muestra un menú interactivo en consola para que el usuario elija el motor LLM.

    Returns:
        str: Proveedor seleccionado ("local" para Ollama o "gemini" para Google AI).
    """
    modelo_local: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    modelo_gemini: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    print("─" * 60)
    print("🧠 Selección de Motor de Inteligencia Artificial:")
    print(f"  [1] Modelo Local (Ollama - {modelo_local})")
    print(f"  [2] Modelo Gemini (Google AI - {modelo_gemini})")
    print("─" * 60)

    while True:
        try:
            opcion: str = input(
                "Selecciona una opción [1/2] (por defecto 1): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Saliendo de AMADEUS.")
            sys.exit(0)

        if opcion in {"", "1", "local", "ollama"}:
            return PROVEEDOR_LOCAL
        if opcion in {"2", "gemini", "google", "google ai"}:
            return PROVEEDOR_GEMINI

        print("⚠️  Opción no válida. Por favor, ingresa '1' (Local) o '2' (Gemini).")


# ─── Clase Principal de la Aplicación ────────────────────────────────────────


class AplicacionAmadeus:
    """Controlador principal de la interfaz de línea de comandos de AMADEUS."""

    def __init__(
        self,
        modo_voz: bool = False,
        respaldo_texto: bool = False,
        proveedor_llm: Optional[str] = None,
    ) -> None:
        self.modo_voz: bool = modo_voz
        self.respaldo_texto: bool = respaldo_texto
        self.proveedor_llm: Optional[str] = proveedor_llm
        self.id_hilo: str = str(uuid.uuid4())
        self.agente: Any = None

        # Componentes de voz (se inicializan solo si se necesitan)
        self._entrada_voz: Any = None
        self._salida_voz: Any = None

    # ─── Inicialización ───────────────────────────────────────────────────────

    def inicializar(self) -> bool:
        """
        Inicializa el modelo LLM seleccionado, el agente y los componentes de voz.

        Returns:
            bool: True si la inicialización fue exitosa, False en caso de error.
        """
        print(BANNER)

        proveedor: str = self.proveedor_llm or seleccionar_proveedor_llm()
        self.proveedor_llm = proveedor

        if proveedor == PROVEEDOR_LOCAL:
            url_base: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            nombre_modelo: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

            print(f"\n[*] Conectando con Ollama ({nombre_modelo})...")
            conectado, mensaje_conexion = verificar_conexion_ollama(
                url_base,
                nombre_modelo,
            )
            if not conectado:
                print(f"\n❌ Error de conexión:\n{mensaje_conexion}")
                return False
            print(f"[✓] Ollama conectado — modelo: {nombre_modelo}")
        else:
            nombre_modelo = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            print(f"\n[*] Configurando Modelo Gemini ({nombre_modelo})...")

        try:
            modelo_llm: BaseChatModel = crear_modelo_llm(proveedor)
        except ValueError as exc:
            print(f"\n❌ Error de configuración del modelo:\n{exc}")
            return False
        except Exception as exc:
            print(f"\n❌ Error inesperado al instanciar el modelo LLM:\n{exc}")
            return False

        if proveedor == PROVEEDOR_GEMINI:
            print(f"[✓] Gemini configurado — modelo: {nombre_modelo}")

        # Inyectar el modelo seleccionado en el núcleo del agente
        self.agente, _, _ = crear_agente(
            modelo_llm=modelo_llm,
            proveedor=proveedor,
        )
        print("[✓] Agente AMADEUS listo")

        # Inicializar voz si corresponde
        if self.modo_voz:
            self._inicializar_voz()

        print("\n" + "─" * 60)
        if self.modo_voz:
            print("🎤 Modo VOZ activado. Habla para interactuar con AMADEUS.")
        else:
            print("⌨️  Modo TEXTO. Escribe tu mensaje y pulsa Enter.")
        print('Escribe/di "salir" para terminar.')
        print("─" * 60 + "\n")

        return True

    def _inicializar_voz(self) -> None:
        """Inicializa los módulos de síntesis y reconocimiento de voz."""
        print("[*] Inicializando síntesis de voz...")
        try:
            from voice.output import VoiceOutput

            self._salida_voz = VoiceOutput()
            print("[✓] Síntesis de voz lista")
        except Exception as exc:
            print(f"[!] No se pudo inicializar la voz: {exc}")
            self._salida_voz = None

        print("[*] Inicializando reconocimiento de voz...")
        try:
            from voice.input import VoiceInput

            self._entrada_voz = VoiceInput()
            print("[✓] Reconocimiento de voz listo")
        except Exception as exc:
            print(f"[!] No se pudo inicializar el micrófono: {exc}")
            self._entrada_voz = None
            if not self.respaldo_texto:
                print("[!] Activando respaldo a entrada por texto.")
                self.respaldo_texto = True

    # ─── Síntesis de voz ──────────────────────────────────────────────────────

    def _hablar(self, texto: str) -> None:
        """Convierte texto a voz si TTS está habilitado y disponible."""
        if os.getenv("NO_TTS", "false").lower() == "true":
            return
        if self._salida_voz is not None:
            self._salida_voz.speak(texto)

    # ─── Bucle principal ──────────────────────────────────────────────────────

    def ejecutar(self) -> None:
        """Ejecuta el bucle principal de conversación con el usuario."""
        reset_all()

        while True:
            try:
                entrada_usuario: Optional[str] = self._obtener_entrada_usuario()
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                self._hablar("¡Hasta luego!")
                break

            if entrada_usuario is None:
                continue

            texto_limpio: str = entrada_usuario.strip().lower()
            if not texto_limpio:
                continue

            if texto_limpio in COMANDOS_SALIDA:
                despedida: str = "¡Hasta luego! Ha sido un placer ayudarte."
                print(f"\n🤖 AMADEUS: {despedida}")
                self._hablar(despedida)
                break

            print(f"\n👤 Tú: {entrada_usuario}")

            accion_pendiente: Optional[PendingAction] = get_pending_action()
            if accion_pendiente:
                self._manejar_confirmacion(accion_pendiente, texto_limpio)
            else:
                if self._intentar_comando_directo(entrada_usuario):
                    continue
                self._ejecutar_agente(entrada_usuario)

    # ─── Entrada de usuario ───────────────────────────────────────────────────

    def _obtener_entrada_usuario(self) -> Optional[str]:
        """Obtiene la entrada del usuario por micrófono o teclado."""
        if self.modo_voz and self._entrada_voz:
            texto: Optional[str] = self._entrada_voz.listen()
            if texto:
                return texto
            if self.respaldo_texto:
                return self._obtener_entrada_texto()
            return None
        return self._obtener_entrada_texto()

    def _obtener_entrada_texto(self) -> Optional[str]:
        """Lee una línea de texto ingresada por el usuario en consola."""
        try:
            return input("\n💬 Tú: ").strip()
        except EOFError:
            return "salir"

    # ─── Confirmación HITL ────────────────────────────────────────────────────

    def _manejar_confirmacion(
        self,
        accion_pendiente: PendingAction,
        entrada_minusculas: str,
    ) -> None:
        """Procesa la respuesta del usuario ante una acción pendiente de confirmación."""
        if entrada_minusculas in FRASES_CONFIRMACION:
            confirm(accion_pendiente.key)
            mensaje_confirmacion: str = "Entendido, procediendo con la acción..."
            print(f"\n🤖 AMADEUS: {mensaje_confirmacion}")
            self._hablar(mensaje_confirmacion)
            self._ejecutar_agente(
                "El usuario ha confirmado la acción. "
                "Por favor, procede exactamente con la misma llamada a la herramienta que hiciste antes."
            )
        elif entrada_minusculas in FRASES_CANCELACION:
            cancel(accion_pendiente.key)
            reset_all()
            mensaje_cancelacion: str = (
                "Acción cancelada. No se ha modificado ningún archivo."
            )
            print(f"\n🤖 AMADEUS: {mensaje_cancelacion}")
            self._hablar(mensaje_cancelacion)
        else:
            recordatorio: str = (
                f"Tengo pendiente tu confirmación para: «{accion_pendiente.description}». "
                f"Di 'confirmar' para continuar o 'cancelar' para abortar."
            )
            print(f"\n🤖 AMADEUS: {recordatorio}")
            self._hablar(recordatorio)

    def _intentar_comando_directo(self, entrada_usuario: str) -> bool:
        """Ejecuta comandos directos de apertura sin invocar al LLM."""
        normalizado: str = re.sub(r"\s+", " ", entrada_usuario.strip().lower())
        coincidencia = re.match(
            r"^(abre|abrir|open|launch|lanzar)\s+(.+)$",
            normalizado,
        )
        if not coincidencia:
            return False

        objetivo: str = coincidencia.group(2).strip()
        objetivo = re.sub(r"^(la|el|los|las|the|my|mi|mis)\s+", "", objetivo)

        if objetivo in MAPA_APLICACIONES_WEB:
            resultado: str = open_url.invoke(
                {"url": MAPA_APLICACIONES_WEB[objetivo]}
            )
            self._anunciar_accion_directa(objetivo, resultado)
            return True

        if objetivo in PALABRAS_CLAVE_PANEL_SISTEMA:
            resultado = open_system_settings.invoke({"setting": objetivo})
            self._anunciar_accion_directa(objetivo, resultado)
            return True

        if objetivo in {"explorador", "explorer", "file explorer"}:
            resultado = open_file_explorer.invoke({"path": ""})
            self._anunciar_accion_directa(objetivo, resultado)
            return True

        resultado = open_application.invoke({"app_name": objetivo})
        self._anunciar_accion_directa(objetivo, resultado)
        return True

    def _anunciar_accion_directa(self, objetivo: str, resultado: str) -> None:
        """Muestra y pronuncia el resultado de una acción directa."""
        if (
            resultado.startswith("✓")
            or resultado.startswith("Opened")
            or resultado.startswith("Abierto")
        ):
            mensaje: str = f"He abierto {objetivo}."
        else:
            mensaje = f"No pude abrir {objetivo}."
        print(f"\n🤖 AMADEUS: {mensaje}")
        self._hablar(mensaje)

    # ─── Invocación del agente ────────────────────────────────────────────────

    def _ejecutar_agente(self, mensaje_usuario: str) -> None:
        """Invoca el agente LangGraph en modo streaming y procesa la respuesta."""
        configuracion: Dict[str, Any] = {
            "configurable": {"thread_id": self.id_hilo}
        }
        carga_entrada: Dict[str, Any] = {
            "messages": [{"role": "user", "content": mensaje_usuario}]
        }

        partes_respuesta: List[str] = []
        herramienta_actual: Optional[str] = None
        encabezado_impreso: bool = False

        try:
            for fragmento in self.agente.stream(
                carga_entrada,
                config=configuracion,
                stream_mode="messages",
            ):
                mensaje, metadatos = fragmento
                nodo: str = metadatos.get("langgraph_node", "")

                # ── Indicador de herramienta en uso ──────────────────────
                if (
                    nodo == "tools"
                    and hasattr(mensaje, "name")
                    and mensaje.name
                ):
                    if mensaje.name != herramienta_actual:
                        herramienta_actual = mensaje.name
                        if not encabezado_impreso:
                            print("\n🤖 AMADEUS: ", end="", flush=True)
                            encabezado_impreso = True
                        print(
                            f"\n   🔧 [{herramienta_actual}]",
                            end="",
                            flush=True,
                        )

                # ── Tokens de texto del agente ────────────────────────────
                contenido: Any = getattr(mensaje, "content", None)
                tipo_mensaje: str = getattr(mensaje, "type", "")

                if (
                    contenido
                    and isinstance(contenido, str)
                    and tipo_mensaje in {"ai", "AIMessageChunk"}
                ):
                    if not encabezado_impreso:
                        print("\n🤖 AMADEUS: ", end="", flush=True)
                        encabezado_impreso = True
                    print(contenido, end="", flush=True)
                    partes_respuesta.append(contenido)

                elif (
                    contenido
                    and isinstance(contenido, list)
                    and tipo_mensaje in {"ai", "AIMessageChunk"}
                ):
                    for bloque in contenido:
                        if (
                            isinstance(bloque, dict)
                            and bloque.get("type") == "text"
                        ):
                            parte_texto: str = bloque.get("text", "")
                            if parte_texto:
                                if not encabezado_impreso:
                                    print("\n🤖 AMADEUS: ", end="", flush=True)
                                    encabezado_impreso = True
                                print(parte_texto, end="", flush=True)
                                partes_respuesta.append(parte_texto)

        except KeyboardInterrupt:
            print("\n[Interrumpido]")
            return
        except Exception as exc:
            mensaje_error: str = f"\n❌ Error del agente: {exc}"
            print(mensaje_error)
            self._hablar("Ha ocurrido un error. Por favor, inténtalo de nuevo.")
            return

        if not partes_respuesta:
            respuesta_estado: Optional[str] = self._obtener_ultimo_mensaje_ia()
            if respuesta_estado:
                print(f"\n🤖 AMADEUS: {respuesta_estado}", flush=True)
                partes_respuesta.append(respuesta_estado)

        if encabezado_impreso or partes_respuesta:
            print()

        respuesta_final: str = "".join(partes_respuesta)

        accion_pendiente: Optional[PendingAction] = get_pending_action()
        if accion_pendiente:
            aviso_confirmacion: str = (
                f"{respuesta_final}\n\n"
                f"⚠️  CONFIRMACIÓN REQUERIDA: {accion_pendiente.description}\n"
                f"   Di 'confirmar' para continuar o 'cancelar' para abortar."
            )
            print(f"\r🤖 AMADEUS: {aviso_confirmacion}")
            self._hablar(
                f"{respuesta_final}. {accion_pendiente.description}. "
                f"Di confirmar para continuar o cancelar para abortar."
            )
        else:
            if respuesta_final:
                self._hablar(respuesta_final)
            reset_all()

    def _obtener_ultimo_mensaje_ia(self) -> Optional[str]:
        """
        Obtiene el último mensaje AI del estado del grafo como mecanismo de respaldo.

        Returns:
            Optional[str]: Texto del último mensaje generado por el modelo, o None.
        """
        try:
            configuracion: Dict[str, Any] = {
                "configurable": {"thread_id": self.id_hilo}
            }
            estado = self.agente.get_state(configuracion)
            mensajes: List[Any] = estado.values.get("messages", [])
            for msg in reversed(mensajes):
                tipo_msg: str = getattr(msg, "type", "")
                contenido: Any = getattr(msg, "content", "")
                if (
                    tipo_msg == "ai"
                    and isinstance(contenido, str)
                    and contenido.strip()
                ):
                    return contenido.strip()
                if tipo_msg == "ai" and isinstance(contenido, list):
                    textos: List[str] = [
                        b.get("text", "")
                        for b in contenido
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    unido: str = " ".join(t for t in textos if t).strip()
                    if unido:
                        return unido
        except Exception:
            pass
        return None


# Alias de compatibilidad
AmadeusApp = AplicacionAmadeus


def analizar_argumentos() -> argparse.Namespace:
    """Analiza y retorna los argumentos de línea de comandos."""
    analizador = argparse.ArgumentParser(
        description="AMADEUS — Asistente de IA (Local Ollama / Google Gemini)"
    )
    analizador.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="Activar modo de entrada por voz (micrófono)",
    )
    analizador.add_argument(
        "--text-fallback",
        "-t",
        action="store_true",
        help="Si la voz falla, usar teclado como respaldo",
    )
    analizador.add_argument(
        "--no-tts",
        action="store_true",
        help="Desactivar síntesis de voz (solo texto en pantalla)",
    )
    analizador.add_argument(
        "--mic",
        "-m",
        type=int,
        default=None,
        metavar="INDEX",
        help=(
            "Índice del micrófono a usar (evita el menú de selección). "
            "Usa --list-mics para ver los índices disponibles."
        ),
    )
    analizador.add_argument(
        "--list-mics",
        action="store_true",
        help="Listar todos los micrófonos disponibles y salir",
    )
    return analizador.parse_args()


def main() -> None:
    """Punto de entrada principal de la aplicación CLI."""
    argumentos = analizar_argumentos()

    if argumentos.list_mics:
        from voice.input import list_microphones

        microfonos = list_microphones()
        print("\n🎙️  Micrófonos disponibles:")
        print("─" * 50)
        for indice, nombre in microfonos:
            print(f"  [{indice:2d}] {nombre}")
        print("─" * 50)
        print("\nUsa:  python main.py --voice --mic <INDEX>")
        return

    if argumentos.mic is not None:
        os.environ["MICROPHONE_INDEX"] = str(argumentos.mic)

    if argumentos.no_tts:
        os.environ["NO_TTS"] = "true"

    aplicacion = AplicacionAmadeus(
        modo_voz=argumentos.voice,
        respaldo_texto=argumentos.text_fallback,
    )

    if not aplicacion.inicializar():
        sys.exit(1)

    aplicacion.ejecutar()


if __name__ == "__main__":
    main()
