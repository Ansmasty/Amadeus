"""
gui.py

AMADEUS — Interfaz gráfica de escritorio (CustomTkinter).

Este módulo es una capa de presentación alternativa a main.py (CLI).
NO modifica agent/core.py, agent/hitl.py, agent/tools/*, security/* ni
voice/*: reutiliza exactamente las mismas funciones que usa el modo texto,
así que la funcionalidad operativa del agente permanece idéntica.

Ejecutar con: run_gui.bat   o   python gui.py
"""
from __future__ import annotations

import os
import queue
import re
import threading
import uuid

import customtkinter as ctk
from dotenv import load_dotenv

load_dotenv()

from agent.core import create_agent, check_ollama_connection
from agent.tools.browser import open_url
from agent.tools.system_apps import open_application, open_system_settings, open_file_explorer
from agent.hitl import (
    PendingAction,
    cancel,
    confirm,
    get_pending_action,
    reset_all,
)

# ─── Constantes reutilizadas de main.py (mismo comportamiento que el CLI) ────

WEB_APP_MAP = {
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

SYSTEM_PANEL_KEYWORDS = {
    "wifi", "bluetooth", "pantalla", "display", "sonido", "sound",
    "privacidad", "updates", "actualizaciones", "apps", "aplicaciones",
    "explorador", "explorer", "file explorer",
}

# ─── Paleta visual ────────────────────────────────────────────────────────────

BG_MAIN      = "#0b0f14"
BG_PANEL     = "#11161d"
BG_INPUT     = "#151b23"
ACCENT       = "#3fb6ff"
ACCENT_DIM   = "#1c3a4d"
USER_BUBBLE  = "#1d2733"
AI_BUBBLE    = "#0f2530"
TEXT_MAIN    = "#e6edf3"
TEXT_DIM     = "#7d8b99"
TOOL_TAG     = "#f5a623"
ERROR_COLOR  = "#ff5f5f"
OK_COLOR     = "#3ddc84"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ─── Widget: burbuja de mensaje ───────────────────────────────────────────────

class MessageBubble(ctk.CTkFrame):
    """Una burbuja de chat individual (usuario, Amadeus o etiqueta de herramienta)."""

    def __init__(self, master, role: str, text: str = ""):
        super().__init__(master, fg_color="transparent")
        self.role = role
        is_user = role == "user"

        bubble_color = USER_BUBBLE if is_user else AI_BUBBLE
        border_color = TEXT_DIM if is_user else ACCENT_DIM
        anchor = "e" if is_user else "w"

        self.columnconfigure(0, weight=1)
        self.bubble = ctk.CTkFrame(
            self, fg_color=bubble_color, corner_radius=14,
            border_width=1, border_color=border_color,
        )
        self.bubble.grid(row=0, column=0, sticky=anchor, padx=8, pady=4)

        label_name = "Tú" if is_user else "AMADEUS"
        name_color = TEXT_DIM if is_user else ACCENT
        ctk.CTkLabel(
            self.bubble, text=label_name, text_color=name_color,
            font=ctk.CTkFont(size=11, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))

        self.text_var = ctk.StringVar(value=text)
        self.text_label = ctk.CTkLabel(
            self.bubble, textvariable=self.text_var, text_color=TEXT_MAIN,
            font=ctk.CTkFont(size=13), justify="left", anchor="w",
            wraplength=560,
        )
        self.text_label.grid(row=1, column=0, sticky="w", padx=12, pady=(2, 10))

    def append(self, chunk: str) -> None:
        self.text_var.set(self.text_var.get() + chunk)

    def add_tool_tag(self, tool_name: str) -> None:
        tag = ctk.CTkLabel(
            self.bubble, text=f"🔧 {tool_name}", text_color=TOOL_TAG,
            font=ctk.CTkFont(size=11, family="Consolas"), anchor="w",
        )
        tag.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 8))


# ─── Aplicación principal ─────────────────────────────────────────────────────

class AmadeusGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AMADEUS — Asistente de IA Local")
        self.geometry("880x720")
        self.minsize(640, 520)
        self.configure(fg_color=BG_MAIN)

        self.thread_id = str(uuid.uuid4())
        self.agent = None
        self._voice_input = None
        self._voice_output = None
        self.voice_mode = False
        self._listening = False

        self._ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self._current_bubble: MessageBubble | None = None
        self._current_tool: str | None = None

        self._build_header()
        self._build_chat_area()
        self._build_confirmation_bar()
        self._build_input_bar()

        self.after(150, self._poll_queue)
        self.after(300, self._boot)

    # ─── Construcción de la interfaz ──────────────────────────────────────────

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=72)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", padx=20, pady=8)
        ctk.CTkLabel(
            title_box, text="AMADEUS", text_color=ACCENT,
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box, text="Asistente de IA Local — 100% Privado",
            text_color=TEXT_DIM, font=ctk.CTkFont(size=11),
        ).pack(anchor="w")

        status_box = ctk.CTkFrame(header, fg_color="transparent")
        status_box.pack(side="right", padx=20)
        self.status_dot = ctk.CTkLabel(
            status_box, text="●", text_color=TEXT_DIM, font=ctk.CTkFont(size=16),
        )
        self.status_dot.pack(side="left", padx=(0, 6))
        self.status_label = ctk.CTkLabel(
            status_box, text="Iniciando…", text_color=TEXT_DIM,
            font=ctk.CTkFont(size=12),
        )
        self.status_label.pack(side="left")

    def _build_chat_area(self) -> None:
        self.chat_frame = ctk.CTkScrollableFrame(
            self, fg_color=BG_MAIN, scrollbar_button_color=ACCENT_DIM,
        )
        self.chat_frame.pack(side="top", fill="both", expand=True, padx=6, pady=(6, 0))
        self.chat_frame.columnconfigure(0, weight=1)
        self._row = 0

    def _build_confirmation_bar(self) -> None:
        self.confirm_bar = ctk.CTkFrame(self, fg_color="#241a0f", corner_radius=10)
        self.confirm_label = ctk.CTkLabel(
            self.confirm_bar, text="", text_color=TOOL_TAG,
            font=ctk.CTkFont(size=12), wraplength=760, justify="left",
        )
        self.confirm_label.pack(side="top", fill="x", padx=14, pady=(10, 6))

        btn_row = ctk.CTkFrame(self.confirm_bar, fg_color="transparent")
        btn_row.pack(side="top", pady=(0, 10))
        ctk.CTkButton(
            btn_row, text="✓ Confirmar", fg_color=OK_COLOR, text_color="#0b0f14",
            hover_color="#33c470", width=120, command=self._on_confirm_click,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            btn_row, text="✕ Cancelar", fg_color=ERROR_COLOR, text_color="#0b0f14",
            hover_color="#e04b4b", width=120, command=self._on_cancel_click,
        ).pack(side="left", padx=6)
        # Oculto hasta que haya una acción pendiente
        self.confirm_bar.pack_forget()

    def _build_input_bar(self) -> None:
        self.input_bar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=64)
        self.input_bar.pack(side="bottom", fill="x")
        self.input_bar.pack_propagate(False)

        self.mic_button = ctk.CTkButton(
            self.input_bar, text="🎤", width=44, height=40, fg_color=BG_INPUT,
            hover_color=ACCENT_DIM, command=self._on_mic_click,
        )
        self.mic_button.pack(side="left", padx=(14, 6), pady=12)

        self.entry = ctk.CTkEntry(
            self.input_bar, placeholder_text="Escribe tu mensaje…", height=40,
            fg_color=BG_INPUT, border_color=ACCENT_DIM, text_color=TEXT_MAIN,
            font=ctk.CTkFont(size=13),
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=6, pady=12)
        self.entry.bind("<Return>", lambda _e: self._on_send_click())

        self.send_button = ctk.CTkButton(
            self.input_bar, text="Enviar", width=90, height=40, fg_color=ACCENT,
            hover_color="#2c93cc", text_color="#0b0f14",
            font=ctk.CTkFont(weight="bold"), command=self._on_send_click,
        )
        self.send_button.pack(side="left", padx=(6, 14), pady=12)

    # ─── Arranque del agente (en segundo plano) ───────────────────────────────

    def _boot(self) -> None:
        self._set_status("Conectando con Ollama…", TOOL_TAG)
        self._add_system_note(
            "Inicializando AMADEUS. Esto puede tardar unos segundos…"
        )
        threading.Thread(target=self._boot_worker, daemon=True).start()

    def _boot_worker(self) -> None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        ok, msg = check_ollama_connection(base_url, model)
        if not ok:
            self._ui_queue.put(("status", "Error de conexión", ERROR_COLOR))
            self._ui_queue.put(("system", f"❌ No se pudo conectar con Ollama:\n{msg}"))
            return

        self.agent, _, _ = create_agent()

        try:
            from voice.output import VoiceOutput
            self._voice_output = VoiceOutput()
        except Exception:
            self._voice_output = None

        try:
            from voice.input import VoiceInput
            self._voice_input = VoiceInput()
        except Exception:
            self._voice_input = None
            self._ui_queue.put(("mic_disabled",))

        self._ui_queue.put(("status", f"Conectado — {model}", OK_COLOR))
        self._ui_queue.put(("system", "AMADEUS listo. Escribe o usa el micrófono para empezar."))

    # ─── Cola de eventos hilo-seguros ─────────────────────────────────────────

    def _poll_queue(self) -> None:
        try:
            while True:
                event = self._ui_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.after(60, self._poll_queue)

    def _handle_event(self, event: tuple) -> None:
        kind = event[0]

        if kind == "status":
            _, text, color = event
            self._set_status(text, color)

        elif kind == "system":
            self._add_system_note(event[1])

        elif kind == "mic_disabled":
            self.mic_button.configure(state="disabled")

        elif kind == "start_bubble":
            self._current_bubble = MessageBubble(self.chat_frame, role="assistant")
            self._current_bubble.grid(row=self._row, column=0, sticky="ew", pady=2)
            self._row += 1
            self._scroll_to_bottom()

        elif kind == "token":
            if self._current_bubble:
                self._current_bubble.append(event[1])
                self._scroll_to_bottom()

        elif kind == "tool":
            if self._current_bubble:
                self._current_bubble.add_tool_tag(event[1])
                self._scroll_to_bottom()

        elif kind == "final_text":
            # Fallback: modelos pequeños que no emiten tokens en streaming
            if not self._current_bubble:
                self._current_bubble = MessageBubble(self.chat_frame, role="assistant")
                self._current_bubble.grid(row=self._row, column=0, sticky="ew", pady=2)
                self._row += 1
            self._current_bubble.append(event[1])
            self._scroll_to_bottom()

        elif kind == "error":
            self._add_system_note(f"❌ {event[1]}", color=ERROR_COLOR)
            self._set_ready()

        elif kind == "pending":
            self._show_confirmation_bar(event[1])
            self._set_ready()

        elif kind == "done":
            self._set_ready()

    # ─── Envío de mensajes ─────────────────────────────────────────────────────

    def _on_send_click(self) -> None:
        text = self.entry.get().strip()
        if not text or self.agent is None:
            return
        self.entry.delete(0, "end")
        self._dispatch_user_message(text)

    def _dispatch_user_message(self, text: str) -> None:
        bubble = MessageBubble(self.chat_frame, role="user", text=text)
        bubble.grid(row=self._row, column=0, sticky="ew", pady=2)
        self._row += 1
        self._scroll_to_bottom()

        stripped = text.strip().lower()

        pending = get_pending_action()
        if pending:
            self._resolve_confirmation(pending, stripped)
            return

        if self._maybe_handle_direct_command(text):
            return

        self._set_busy("AMADEUS está pensando…")
        threading.Thread(target=self._run_agent_worker, args=(text,), daemon=True).start()

    def _maybe_handle_direct_command(self, user_input: str) -> bool:
        """Réplica exacta de la lógica de main.py para comandos directos."""
        normalized = re.sub(r"\s+", " ", user_input.strip().lower())
        match = re.match(r"^(abre|abrir|open|launch|lanzar)\s+(.+)$", normalized)
        if not match:
            return False

        target = match.group(2).strip()
        target = re.sub(r"^(la|el|los|las|the|my|mi|mis)\s+", "", target)

        if target in WEB_APP_MAP:
            result = open_url.invoke({"url": WEB_APP_MAP[target]})
        elif target in SYSTEM_PANEL_KEYWORDS:
            result = open_system_settings.invoke({"setting": target})
        elif target in {"explorador", "explorer", "file explorer"}:
            result = open_file_explorer.invoke({"path": ""})
        else:
            result = open_application.invoke({"app_name": target})

        ok = result.startswith("✓") or result.startswith("Opened") or result.startswith("Abierto")
        message = f"He abierto {target}." if ok else f"No pude abrir {target}."
        self._ui_queue.put(("start_bubble",))
        self._ui_queue.put(("token", message))
        self._speak(message)
        return True

    # ─── Invocación del agente (hilo de trabajo) ──────────────────────────────

    def _run_agent_worker(self, user_message: str) -> None:
        config = {"configurable": {"thread_id": self.thread_id}}
        input_payload = {"messages": [{"role": "user", "content": user_message}]}

        full_response_parts: list[str] = []
        current_tool: str | None = None
        started = False

        try:
            for chunk in self.agent.stream(input_payload, config=config, stream_mode="messages"):
                message, metadata = chunk
                node = metadata.get("langgraph_node", "")

                if node == "tools" and getattr(message, "name", None):
                    if message.name != current_tool:
                        current_tool = message.name
                        if not started:
                            self._ui_queue.put(("start_bubble",))
                            started = True
                        self._ui_queue.put(("tool", current_tool))

                content = getattr(message, "content", None)
                msg_type = getattr(message, "type", "")

                if content and isinstance(content, str) and msg_type == "ai":
                    if not started:
                        self._ui_queue.put(("start_bubble",))
                        started = True
                    self._ui_queue.put(("token", content))
                    full_response_parts.append(content)

                elif content and isinstance(content, list) and msg_type == "ai":
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_part = block.get("text", "")
                            if text_part:
                                if not started:
                                    self._ui_queue.put(("start_bubble",))
                                    started = True
                                self._ui_queue.put(("token", text_part))
                                full_response_parts.append(text_part)

        except Exception as exc:
            self._ui_queue.put(("error", f"Error del agente: {exc}"))
            self._speak("Ha ocurrido un error. Por favor, inténtalo de nuevo.")
            return

        if not full_response_parts:
            final_from_state = self._get_last_ai_message()
            if final_from_state:
                self._ui_queue.put(("final_text", final_from_state))
                full_response_parts.append(final_from_state)

        final_response = "".join(full_response_parts)

        pending = get_pending_action()
        if pending:
            self._ui_queue.put(("pending", pending))
            self._speak(
                f"{final_response}. {pending.description}. "
                f"Confirma para continuar o cancela para abortar."
            )
        else:
            if final_response:
                self._speak(final_response)
            reset_all()
            self._ui_queue.put(("done",))

    def _get_last_ai_message(self) -> str | None:
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.agent.get_state(config)
            messages = state.values.get("messages", [])
            for msg in reversed(messages):
                msg_type = getattr(msg, "type", "")
                content = getattr(msg, "content", "")
                if msg_type == "ai" and isinstance(content, str) and content.strip():
                    return content.strip()
                if msg_type == "ai" and isinstance(content, list):
                    texts = [b.get("text", "") for b in content
                             if isinstance(b, dict) and b.get("type") == "text"]
                    joined = " ".join(t for t in texts if t).strip()
                    if joined:
                        return joined
        except Exception:
            pass
        return None

    # ─── Confirmaciones HITL ───────────────────────────────────────────────────

    def _show_confirmation_bar(self, pending: PendingAction) -> None:
        self.confirm_label.configure(text=f"⚠️  Confirmación requerida: {pending.description}")
        self.confirm_bar.pack(side="bottom", fill="x", padx=14, pady=(0, 6), before=self.input_bar)

    def _resolve_confirmation(self, pending: PendingAction, text_lower: str) -> None:
        if text_lower in {"confirmar", "confirm", "sí", "si", "yes", "adelante", "procede"}:
            self._on_confirm_click()
        elif text_lower in {"cancelar", "cancel", "no", "cancela"}:
            self._on_cancel_click()
        else:
            self._add_system_note(
                f"Tengo pendiente tu confirmación: «{pending.description}». "
                f"Usa los botones o escribe 'confirmar' / 'cancelar'.",
                color=TOOL_TAG,
            )

    def _on_confirm_click(self) -> None:
        pending = get_pending_action()
        if not pending:
            return
        confirm(pending.key)
        self.confirm_bar.pack_forget()
        self._add_system_note("Entendido, procediendo con la acción…")
        self._set_busy("AMADEUS está procesando…")
        threading.Thread(
            target=self._run_agent_worker,
            args=("El usuario ha confirmado la acción. Por favor, procede exactamente "
                  "con la misma llamada a la herramienta que hiciste antes.",),
            daemon=True,
        ).start()

    def _on_cancel_click(self) -> None:
        pending = get_pending_action()
        if pending:
            cancel(pending.key)
        reset_all()
        self.confirm_bar.pack_forget()
        self._add_system_note("Acción cancelada. No se ha modificado ningún archivo.")
        self._speak("Acción cancelada. No se ha modificado ningún archivo.")

    # ─── Voz ────────────────────────────────────────────────────────────────────

    def _on_mic_click(self) -> None:
        if self._voice_input is None or self._listening:
            return
        self._listening = True
        self.mic_button.configure(fg_color=ACCENT, text="●")
        threading.Thread(target=self._listen_worker, daemon=True).start()

    def _listen_worker(self) -> None:
        text = self._voice_input.listen()
        self._ui_queue.put(("mic_result", text))

    def _speak(self, text: str) -> None:
        if os.getenv("NO_TTS", "false").lower() == "true":
            return
        if self._voice_output is not None:
            threading.Thread(target=self._voice_output.speak, args=(text,), daemon=True).start()

    # ─── Utilidades de UI ───────────────────────────────────────────────────────

    def _add_system_note(self, text: str, color: str = TEXT_DIM) -> None:
        note = ctk.CTkLabel(
            self.chat_frame, text=text, text_color=color,
            font=ctk.CTkFont(size=11, slant="italic"), wraplength=680, justify="left",
        )
        note.grid(row=self._row, column=0, sticky="w", padx=14, pady=4)
        self._row += 1
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=text, text_color=color)
        self.status_dot.configure(text_color=color)

    def _set_busy(self, text: str) -> None:
        self.send_button.configure(state="disabled")
        self._set_status(text, TOOL_TAG)

    def _set_ready(self) -> None:
        self.send_button.configure(state="normal")
        self._current_bubble = None
        self._current_tool = None
        self._listening = False
        self.mic_button.configure(fg_color=BG_INPUT, text="🎤")
        self._set_status("Conectado", OK_COLOR)


def main() -> None:
    app = AmadeusGUI()

    # Manejar resultado del micrófono en el hilo principal
    original_handle = app._handle_event

    def patched_handle(event):
        if event[0] == "mic_result":
            text = event[1]
            app._listening = False
            app.mic_button.configure(fg_color=BG_INPUT, text="🎤")
            if text:
                app._dispatch_user_message(text)
            return
        original_handle(event)

    app._handle_event = patched_handle
    app.mainloop()


if __name__ == "__main__":
    main()