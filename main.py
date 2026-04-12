"""
main.py

AMADEUS - Asistente de IA Local con entrada/salida por voz.
Ejecutar con: run.bat  o  python main.py

Modos disponibles:
  - Texto:      python main.py
  - Voz:        python main.py --voice
  - Voz+Texto:  python main.py --voice --text-fallback
"""
import argparse
import os
import sys
import uuid

from colorlog import ColoredFormatter
import logging

from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

def _setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        "%(log_color)s[%(levelname)s]%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))
    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # Silencia logs de LangChain/httpx
    logging.getLogger("amadeus").setLevel(logging.INFO)
    root.addHandler(handler)

_setup_logging()
log = logging.getLogger("amadeus")

# ─── Imports del agente ───────────────────────────────────────────────────────

from agent.core import create_agent, check_ollama_connection
from agent.hitl import (
    CONFIRMATION_SENTINEL,
    PendingAction,
    cancel,
    confirm,
    get_pending_action,
    reset_all,
)

# ─── Constantes ───────────────────────────────────────────────────────────────

BANNER = r"""
    ___    __  ______    ____  _______  _______
   /   |  /  |/ /   \  / __ \/ ____/ / / ___/
  / /| | / /|_/ / /| | / / / / __/ / / /\__ \ 
 / ___ |/ /  / / ___ |/ /_/ / /___/ /_/ /__/ / 
/_/  |_/_/  /_/_/  |_/_____/_____/\____/____/  

        Asistente de IA Local — 100% Privado
"""

EXIT_COMMANDS  = {"salir", "exit", "quit", "adiós", "adios", "bye"}
CANCEL_PHRASES = {"cancelar", "cancel", "no", "cancela"}
CONFIRM_PHRASES = {"confirmar", "confirm", "sí", "si", "yes", "adelante", "procede"}


# ─── Clase principal ──────────────────────────────────────────────────────────

class AmadeusApp:
    def __init__(self, voice_mode: bool = False, text_fallback: bool = False):
        self.voice_mode = voice_mode
        self.text_fallback = text_fallback
        self.thread_id = str(uuid.uuid4())
        self.agent = None

        # Componentes de voz (se inicializan solo si se necesitan)
        self._voice_input = None
        self._voice_output = None

    # ─── Inicialización ───────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Inicializa el agente y los componentes de voz. Retorna False si falla."""
        print(BANNER)

        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        model    = os.getenv("OLLAMA_MODEL", "llama3.2:3b")   # ← cambiado

        print(f"[*] Conectando con Ollama ({model})...")
        ok, msg = check_ollama_connection(base_url, model)
        if not ok:
            print(f"\n❌ Error de conexión:\n{msg}")
            return False
        print(f"[✓] Ollama conectado — modelo: {model}")

        # Crear agente
        self.agent, _, _ = create_agent()
        print("[✓] Agente AMADEUS listo")

        # Inicializar voz
        if self.voice_mode:
            self._init_voice()

        print("\n" + "─" * 60)
        if self.voice_mode:
            print("🎤 Modo VOZ activado. Habla para interactuar con AMADEUS.")
        else:
            print("⌨️  Modo TEXTO. Escribe tu mensaje y pulsa Enter.")
        print('Escribe/di "salir" para terminar.')
        print("─" * 60 + "\n")

        return True

    def _init_voice(self):
        """Inicializa los módulos de voz."""
        print("[*] Inicializando síntesis de voz...")
        try:
            from voice.output import VoiceOutput
            self._voice_output = VoiceOutput()
            print("[✓] Síntesis de voz lista")
        except Exception as exc:
            print(f"[!] No se pudo inicializar la voz: {exc}")
            self._voice_output = None

        print("[*] Inicializando reconocimiento de voz...")
        try:
            from voice.input import VoiceInput
            self._voice_input = VoiceInput()
            print("[✓] Reconocimiento de voz listo")
        except Exception as exc:
            print(f"[!] No se pudo inicializar el micrófono: {exc}")
            self._voice_input = None
            if not self.text_fallback:
                print("[!] Activando fallback a texto.")
                self.text_fallback = True

    # ─── Síntesis de voz ──────────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        """Convierte texto a voz si TTS está habilitado y disponible."""
        if os.getenv("NO_TTS", "false").lower() == "true":
            return
        if self._voice_output is not None:
            self._voice_output.speak(text)

    # ─── Bucle principal ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Bucle principal de conversación."""
        reset_all()

        while True:
            try:
                user_input = self._get_user_input()
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                self._speak("¡Hasta luego!")
                break

            if user_input is None:
                # Sin entrada (timeout de voz o error)
                continue

            stripped = user_input.strip().lower()
            if not stripped:
                continue

            # Comando de salida
            if stripped in EXIT_COMMANDS:
                farewell = "¡Hasta luego! Ha sido un placer ayudarte."
                print(f"\n🤖 AMADEUS: {farewell}")
                self._speak(farewell)
                break

            print(f"\n👤 Tú: {user_input}")

            # ¿Hay una acción pendiente de confirmación?
            pending = get_pending_action()
            if pending:
                self._handle_confirmation(pending, stripped)
            else:
                self._run_agent(user_input)

    # ─── Entrada de usuario ───────────────────────────────────────────────────

    def _get_user_input(self) -> str | None:
        """Obtiene entrada del usuario (voz o texto)."""
        if self.voice_mode and self._voice_input:
            text = self._voice_input.listen()
            if text:
                return text
            # Fallback a texto si está habilitado
            if self.text_fallback:
                return self._get_text_input()
            return None
        return self._get_text_input()

    def _get_text_input(self) -> str | None:
        """Obtiene entrada de texto del usuario."""
        try:
            return input("\n💬 Tú: ").strip()
        except EOFError:
            return "salir"

    # ─── Confirmación HITL ────────────────────────────────────────────────────

    def _handle_confirmation(self, pending: PendingAction, user_input_lower: str) -> None:
        """Maneja la respuesta del usuario ante una acción pendiente."""
        if user_input_lower in CONFIRM_PHRASES:
            confirm(pending.key)
            confirmation_msg = "Entendido, procediendo con la acción..."
            print(f"\n🤖 AMADEUS: {confirmation_msg}")
            self._speak(confirmation_msg)
            self._run_agent(
                "El usuario ha confirmado la acción. "
                "Por favor, procede exactamente con la misma llamada a la herramienta que hiciste antes."
            )
        elif user_input_lower in CANCEL_PHRASES:
            cancel(pending.key)
            reset_all()
            cancel_msg = "Acción cancelada. No se ha modificado ningún archivo."
            print(f"\n🤖 AMADEUS: {cancel_msg}")
            self._speak(cancel_msg)
        else:
            # El usuario dijo otra cosa, recordarle que hay una confirmación pendiente
            reminder = (
                f"Tengo pendiente tu confirmación para: «{pending.description}». "
                f"Di 'confirmar' para continuar o 'cancelar' para abortar."
            )
            print(f"\n🤖 AMADEUS: {reminder}")
            self._speak(reminder)

    # ─── Invocación del agente ────────────────────────────────────────────────

    def _run_agent(self, user_message: str) -> None:
        """Invoca el agente LangGraph y procesa la respuesta."""
        config = {"configurable": {"thread_id": self.thread_id}}
        input_payload = {
            "messages": [{"role": "user", "content": user_message}]
        }

        full_response_parts: list[str] = []
        current_tool: str | None = None
        printed_header = False

        try:
            for chunk in self.agent.stream(
                input_payload,
                config=config,
                stream_mode="messages"
            ):
                message, metadata = chunk
                node = metadata.get("langgraph_node", "")

                # ── Indicador de herramienta en uso ──────────────────────
                if node == "tools" and hasattr(message, "name") and message.name:
                    if message.name != current_tool:
                        current_tool = message.name
                        if not printed_header:
                            print("\n🤖 AMADEUS: ", end="", flush=True)
                            printed_header = True
                        print(f"\n   🔧 [{current_tool}]", end="", flush=True)

                # ── Tokens de texto del agente ────────────────────────────
                # Capturar contenido de tipo string en mensajes AI,
                # independientemente del nodo (algunos modelos emiten en "agent",
                # otros en "__end__" o directamente como AIMessageChunk)
                content = getattr(message, "content", None)
                msg_type = getattr(message, "type", "")

                if content and isinstance(content, str) and msg_type == "ai":
                    if not printed_header:
                        print("\n🤖 AMADEUS: ", end="", flush=True)
                        printed_header = True
                    print(content, end="", flush=True)
                    full_response_parts.append(content)

                # ── Contenido en formato lista (tool_use + text mezclados) ─
                elif content and isinstance(content, list) and msg_type == "ai":
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_part = block.get("text", "")
                            if text_part:
                                if not printed_header:
                                    print("\n🤖 AMADEUS: ", end="", flush=True)
                                    printed_header = True
                                print(text_part, end="", flush=True)
                                full_response_parts.append(text_part)

        except KeyboardInterrupt:
            print("\n[Interrumpido]")
            return
        except Exception as exc:
            error_msg = f"\n❌ Error del agente: {exc}"
            print(error_msg)
            self._speak("Ha ocurrido un error. Por favor, inténtalo de nuevo.")
            return

        # Si no se imprimió nada, obtener la respuesta final del estado del grafo
        if not full_response_parts:
            final_from_state = self._get_last_ai_message()
            if final_from_state:
                print(f"\n🤖 AMADEUS: {final_from_state}", flush=True)
                full_response_parts.append(final_from_state)

        if printed_header or full_response_parts:
            print()  # Salto de línea final

        final_response = "".join(full_response_parts)

        # Verificar si el agente registró una acción pendiente de confirmación
        pending = get_pending_action()
        if pending:
            confirmation_prompt = (
                f"{final_response}\n\n"
                f"⚠️  CONFIRMACIÓN REQUERIDA: {pending.description}\n"
                f"   Di 'confirmar' para continuar o 'cancelar' para abortar."
            )
            print(f"\r🤖 AMADEUS: {confirmation_prompt}")
            self._speak(
                f"{final_response}. {pending.description}. "
                f"Di confirmar para continuar o cancelar para abortar."
            )
        else:
            if final_response:
                self._speak(final_response)
            reset_all()

    def _get_last_ai_message(self) -> str | None:
        """
        Obtiene el último mensaje AI del estado del grafo como fallback.
        Útil cuando el stream no emite tokens de texto (modelos pequeños).
        """
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.agent.get_state(config)
            messages = state.values.get("messages", [])
            # Recorrer al revés buscando el último AIMessage con contenido
            for msg in reversed(messages):
                msg_type = getattr(msg, "type", "")
                content  = getattr(msg, "content", "")
                if msg_type == "ai" and isinstance(content, str) and content.strip():
                    return content.strip()
                if msg_type == "ai" and isinstance(content, list):
                    # Extraer bloques de texto
                    texts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    joined = " ".join(t for t in texts if t).strip()
                    if joined:
                        return joined
        except Exception:
            pass
        return None

# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMADEUS — Asistente de IA Local"
    )
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Activar modo de entrada por voz (micrófono)"
    )
    parser.add_argument(
        "--text-fallback", "-t",
        action="store_true",
        help="Si la voz falla, usar teclado como fallback"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Desactivar síntesis de voz (solo texto en pantalla)"
    )
    parser.add_argument(
        "--mic", "-m",
        type=int,
        default=None,
        metavar="INDEX",
        help="Índice del micrófono a usar (evita el menú de selección). "
             "Usa --list-mics para ver los índices disponibles."
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="Listar todos los micrófonos disponibles y salir"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Listar micrófonos y salir
    if args.list_mics:
        from voice.input import list_microphones
        mics = list_microphones()
        print("\n🎙️  Micrófonos disponibles:")
        print("─" * 50)
        for idx, name in mics:
            print(f"  [{idx:2d}] {name}")
        print("─" * 50)
        print("\nUsa:  python main.py --voice --mic <INDEX>")
        return

    # Pre-seleccionar micrófono si se pasó --mic
    if args.mic is not None:
        os.environ["MICROPHONE_INDEX"] = str(args.mic)

    voice_mode = args.voice
    text_fallback = args.text_fallback

    if args.no_tts:
        os.environ["NO_TTS"] = "true"

    app = AmadeusApp(
        voice_mode=voice_mode,
        text_fallback=text_fallback,
    )

    if not app.setup():
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
