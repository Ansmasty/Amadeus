"""
gui.py

AMADEUS — Interfaz gráfica de escritorio (CustomTkinter) con soporte de
arquitectura LLM dual (Ollama Local / Google Gemini en la nube).

Incluye un selector visual (ComboBox) en la barra superior para elegir el
motor de IA antes de instanciar el agente o cambiar de proveedor en caliente.

Ejecutar con: run_gui.bat   o   python gui.py [--proveedor local|gemini]
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import threading
import uuid
from pathlib import Path
from tkinter import filedialog
from typing import Any, Dict, Final, List, Optional, Pattern, Set, Tuple

import customtkinter as ctk
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from PIL import Image

RUTA_ENV: Final[Path] = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=RUTA_ENV, override=True)

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
from agent.tools.screen import analyze_image, grab_screenshot  # noqa: E402
from agent.tools.system_apps import (  # noqa: E402
    open_application,
    open_file_explorer,
    open_system_settings,
)

# ─── Constantes de Comandos Directos y Expresiones ───────────────────────────

PATRON_FIN_ORACION: Final[Pattern[str]] = re.compile(r"[.!?]+(?=\s)|\n+")

OPCION_MOTOR_LOCAL: Final[str] = "Modelo Local (Ollama)"
OPCION_MOTOR_GEMINI: Final[str] = "Modelo Gemini (Google AI)"

MAPA_OPCIONES_PROVEEDOR: Final[Dict[str, str]] = {
    OPCION_MOTOR_LOCAL: PROVEEDOR_LOCAL,
    OPCION_MOTOR_GEMINI: PROVEEDOR_GEMINI,
}

MAPA_APLICACIONES_WEB: Final[Dict[str, str]] = {
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

PALABRAS_CLAVE_PANEL_SISTEMA: Final[Set[str]] = {
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

# ─── Paleta Visual ────────────────────────────────────────────────────────────

COLOR_FONDO_PRINCIPAL: Final[str] = "#0b0f14"
COLOR_FONDO_PANEL: Final[str] = "#11161d"
COLOR_FONDO_ENTRADA: Final[str] = "#151b23"
COLOR_ACENTO: Final[str] = "#3fb6ff"
COLOR_ACENTO_ATENUADO: Final[str] = "#1c3a4d"
COLOR_BURBUJA_USUARIO: Final[str] = "#1d2733"
COLOR_BURBUJA_IA: Final[str] = "#0f2530"
COLOR_TEXTO_PRINCIPAL: Final[str] = "#e6edf3"
COLOR_TEXTO_ATENUADO: Final[str] = "#7d8b99"
COLOR_ETIQUETA_TOOL: Final[str] = "#f5a623"
COLOR_ERROR: Final[str] = "#ff5f5f"
COLOR_EXITO: Final[str] = "#3ddc84"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ─── Widget: Burbuja de Mensaje ───────────────────────────────────────────────


class BurbujaMensaje(ctk.CTkFrame):
    """Una burbuja de chat individual (usuario, Amadeus o etiqueta de herramienta)."""

    def __init__(
        self,
        master: Any,
        rol: str,
        texto: str = "",
        miniatura: Optional[ctk.CTkImage] = None,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self.rol: str = rol
        es_usuario: bool = rol == "user"

        color_burbuja: str = (
            COLOR_BURBUJA_USUARIO if es_usuario else COLOR_BURBUJA_IA
        )
        color_borde: str = (
            COLOR_TEXTO_ATENUADO if es_usuario else COLOR_ACENTO_ATENUADO
        )
        anclaje: str = "e" if es_usuario else "w"

        self.columnconfigure(0, weight=1)
        self.burbuja = ctk.CTkFrame(
            self,
            fg_color=color_burbuja,
            corner_radius=14,
            border_width=1,
            border_color=color_borde,
        )
        self.burbuja.grid(row=0, column=0, sticky=anclaje, padx=8, pady=4)

        nombre_etiqueta: str = "Tú" if es_usuario else "AMADEUS"
        color_nombre: str = COLOR_TEXTO_ATENUADO if es_usuario else COLOR_ACENTO
        ctk.CTkLabel(
            self.burbuja,
            text=nombre_etiqueta,
            text_color=color_nombre,
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))

        siguiente_fila: int = 1
        if miniatura is not None:
            etiqueta_imagen = ctk.CTkLabel(
                self.burbuja,
                image=miniatura,
                text="",
            )
            etiqueta_imagen.image = miniatura  # type: ignore[attr-defined]
            etiqueta_imagen.grid(
                row=siguiente_fila,
                column=0,
                sticky="w",
                padx=12,
                pady=(4, 2),
            )
            siguiente_fila += 1

        self._fila_texto: int = siguiente_fila
        self.variable_texto = ctk.StringVar(value=texto)
        self.etiqueta_texto = ctk.CTkLabel(
            self.burbuja,
            textvariable=self.variable_texto,
            text_color=COLOR_TEXTO_PRINCIPAL,
            font=ctk.CTkFont(size=13),
            justify="left",
            anchor="w",
            wraplength=560,
        )
        self.etiqueta_texto.grid(
            row=siguiente_fila,
            column=0,
            sticky="w",
            padx=12,
            pady=(2, 10),
        )

    def agregar_texto(self, fragmento: str) -> None:
        """Concatena un fragmento de texto a la burbuja actual."""
        self.variable_texto.set(self.variable_texto.get() + fragmento)

    def agregar_etiqueta_herramienta(self, nombre_herramienta: str) -> None:
        """Muestra una insignia visual indicando la herramienta ejecutada."""
        etiqueta = ctk.CTkLabel(
            self.burbuja,
            text=f"🔧 {nombre_herramienta}",
            text_color=COLOR_ETIQUETA_TOOL,
            font=ctk.CTkFont(size=11, family="Consolas"),
            anchor="w",
        )
        etiqueta.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 8))


# Alias de compatibilidad
MessageBubble = BurbujaMensaje


# ─── Aplicación Gráfica Principal ─────────────────────────────────────────────


class InterfazAmadeus(ctk.CTk):
    """Ventana principal de la interfaz gráfica de AMADEUS con selector de motor LLM."""

    def __init__(
        self,
        proveedor_inicial: Optional[str] = None,
        auto_iniciar: bool = False,
    ) -> None:
        super().__init__()
        self.title("AMADEUS — Asistente de IA (Local / Gemini)")
        self.geometry("940x740")
        self.minsize(720, 540)
        self.configure(fg_color=COLOR_FONDO_PRINCIPAL)

        self.id_hilo: str = str(uuid.uuid4())
        self.agente: Any = None
        self.proveedor_activo: Optional[str] = None
        self.modelo_activo: str = ""
        self._inicializando_motor: bool = False

        self._entrada_voz: Any = None
        self._salida_voz: Any = None
        self._voz_inicializada: bool = False
        self._escuchando: bool = False

        self._cola_ui: "queue.Queue[Tuple[Any, ...]]" = queue.Queue()
        self._burbuja_actual: Optional[BurbujaMensaje] = None
        self._herramienta_actual: Optional[str] = None
        self._imagen_pendiente: Optional[Image.Image] = None
        self._etiqueta_imagen_pendiente: str = ""
        self._mensaje_en_espera: Optional[str] = None

        self._construir_encabezado(proveedor_inicial)
        self._construir_area_chat()
        self._construir_barra_confirmacion()
        self._construir_barra_adjuntos()
        self._construir_barra_entrada()

        self.after(100, self._procesar_cola_ui)

        if proveedor_inicial or auto_iniciar:
            self.after(250, self._al_conectar_motor_click)
        else:
            self.after(200, self._mostrar_bienvenida_seleccion)

    # ─── Construcción de la Interfaz ──────────────────────────────────────────

    def _construir_encabezado(
        self,
        proveedor_inicial: Optional[str] = None,
    ) -> None:
        """Construye la barra superior con título, selector ComboBox de LLM y estado."""
        encabezado = ctk.CTkFrame(
            self,
            fg_color=COLOR_FONDO_PANEL,
            corner_radius=0,
            height=78,
        )
        encabezado.pack(side="top", fill="x")
        encabezado.pack_propagate(False)

        caja_titulo = ctk.CTkFrame(encabezado, fg_color="transparent")
        caja_titulo.pack(side="left", padx=20, pady=8)
        ctk.CTkLabel(
            caja_titulo,
            text="AMADEUS",
            text_color=COLOR_ACENTO,
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            caja_titulo,
            text="Arquitectura Dual — Ollama Local / Google Gemini",
            text_color=COLOR_TEXTO_ATENUADO,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w")

        # ── Selector visual de motor LLM (ComboBox + Botón de conexión) ──
        caja_motor = ctk.CTkFrame(encabezado, fg_color="transparent")
        caja_motor.pack(side="left", padx=16, pady=12)

        ctk.CTkLabel(
            caja_motor,
            text="Motor IA:",
            text_color=COLOR_TEXTO_ATENUADO,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", padx=(0, 8))

        opcion_predeterminada: str = (
            OPCION_MOTOR_GEMINI
            if (proveedor_inicial or "").strip().lower() == PROVEEDOR_GEMINI
            else OPCION_MOTOR_LOCAL
        )

        self.selector_motor = ctk.CTkComboBox(
            caja_motor,
            values=[OPCION_MOTOR_LOCAL, OPCION_MOTOR_GEMINI],
            width=230,
            height=34,
            fg_color=COLOR_FONDO_ENTRADA,
            border_color=COLOR_ACENTO_ATENUADO,
            button_color=COLOR_ACENTO_ATENUADO,
            button_hover_color=COLOR_ACENTO,
            text_color=COLOR_TEXTO_PRINCIPAL,
            font=ctk.CTkFont(size=12),
            command=self._al_cambiar_selector_motor,
        )
        self.selector_motor.set(opcion_predeterminada)
        self.selector_motor.configure(state="readonly")
        self.selector_motor.pack(side="left", padx=(0, 8))

        self.boton_conectar = ctk.CTkButton(
            caja_motor,
            text="Iniciar Motor",
            width=115,
            height=34,
            fg_color=COLOR_ACENTO,
            hover_color="#2c93cc",
            text_color="#0b0f14",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._al_conectar_motor_click,
        )
        self.boton_conectar.pack(side="left")

        # ── Indicador de estado ──
        caja_estado = ctk.CTkFrame(encabezado, fg_color="transparent")
        caja_estado.pack(side="right", padx=20)
        self.punto_estado = ctk.CTkLabel(
            caja_estado,
            text="●",
            text_color=COLOR_ETIQUETA_TOOL,
            font=ctk.CTkFont(size=16),
        )
        self.punto_estado.pack(side="left", padx=(0, 6))
        self.etiqueta_estado = ctk.CTkLabel(
            caja_estado,
            text="Selecciona un motor",
            text_color=COLOR_TEXTO_ATENUADO,
            font=ctk.CTkFont(size=12),
        )
        self.etiqueta_estado.pack(side="left")

    def _construir_area_chat(self) -> None:
        """Construye el contenedor desplazable de mensajes del chat."""
        self.marco_chat = ctk.CTkScrollableFrame(
            self,
            fg_color=COLOR_FONDO_PRINCIPAL,
            scrollbar_button_color=COLOR_ACENTO_ATENUADO,
        )
        self.marco_chat.pack(
            side="top",
            fill="both",
            expand=True,
            padx=6,
            pady=(6, 0),
        )
        self.marco_chat.columnconfigure(0, weight=1)
        self._fila: int = 0

    def _construir_barra_confirmacion(self) -> None:
        """Construye la barra inferior de confirmación HITL para acciones destructivas."""
        self.barra_confirmacion = ctk.CTkFrame(
            self,
            fg_color="#241a0f",
            corner_radius=10,
        )
        self.etiqueta_confirmacion = ctk.CTkLabel(
            self.barra_confirmacion,
            text="",
            text_color=COLOR_ETIQUETA_TOOL,
            font=ctk.CTkFont(size=12),
            wraplength=760,
            justify="left",
        )
        self.etiqueta_confirmacion.pack(
            side="top",
            fill="x",
            padx=14,
            pady=(10, 6),
        )

        fila_botones = ctk.CTkFrame(
            self.barra_confirmacion,
            fg_color="transparent",
        )
        fila_botones.pack(side="top", pady=(0, 10))
        ctk.CTkButton(
            fila_botones,
            text="✓ Confirmar",
            fg_color=COLOR_EXITO,
            text_color="#0b0f14",
            hover_color="#33c470",
            width=120,
            command=self._al_confirmar_click,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            fila_botones,
            text="✕ Cancelar",
            fg_color=COLOR_ERROR,
            text_color="#0b0f14",
            hover_color="#e04b4b",
            width=120,
            command=self._al_cancelar_click,
        ).pack(side="left", padx=6)
        self.barra_confirmacion.pack_forget()

    def _construir_barra_adjuntos(self) -> None:
        """Construye la barra de previsualización de imágenes adjuntas."""
        self.barra_adjuntos = ctk.CTkFrame(
            self,
            fg_color=COLOR_FONDO_PANEL,
            corner_radius=10,
        )

        self.etiqueta_miniatura_adjunto = ctk.CTkLabel(
            self.barra_adjuntos,
            text="",
            image=None,
        )
        self.etiqueta_miniatura_adjunto.pack(
            side="left",
            padx=(10, 8),
            pady=8,
        )

        self.etiqueta_nombre_adjunto = ctk.CTkLabel(
            self.barra_adjuntos,
            text="",
            text_color=COLOR_TEXTO_ATENUADO,
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.etiqueta_nombre_adjunto.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 8),
        )

        ctk.CTkButton(
            self.barra_adjuntos,
            text="✕",
            width=28,
            height=28,
            fg_color=COLOR_FONDO_ENTRADA,
            hover_color=COLOR_ERROR,
            command=self._limpiar_adjunto,
        ).pack(side="right", padx=10, pady=8)

        self.barra_adjuntos.pack_forget()

    def _construir_barra_entrada(self) -> None:
        """Construye la barra inferior de entrada de texto, voz y capturas."""
        self.barra_entrada = ctk.CTkFrame(
            self,
            fg_color=COLOR_FONDO_PANEL,
            corner_radius=0,
            height=64,
        )
        self.barra_entrada.pack(side="bottom", fill="x")
        self.barra_entrada.pack_propagate(False)

        self.boton_microfono = ctk.CTkButton(
            self.barra_entrada,
            text="🎤",
            width=44,
            height=40,
            fg_color=COLOR_FONDO_ENTRADA,
            hover_color=COLOR_ACENTO_ATENUADO,
            command=self._al_microfono_click,
        )
        self.boton_microfono.pack(side="left", padx=(14, 4), pady=12)

        self.boton_adjuntar = ctk.CTkButton(
            self.barra_entrada,
            text="📎",
            width=40,
            height=40,
            fg_color=COLOR_FONDO_ENTRADA,
            hover_color=COLOR_ACENTO_ATENUADO,
            command=self._al_adjuntar_click,
        )
        self.boton_adjuntar.pack(side="left", padx=4, pady=12)

        self.boton_captura = ctk.CTkButton(
            self.barra_entrada,
            text="📷",
            width=40,
            height=40,
            fg_color=COLOR_FONDO_ENTRADA,
            hover_color=COLOR_ACENTO_ATENUADO,
            command=self._al_capturar_click,
        )
        self.boton_captura.pack(side="left", padx=(4, 6), pady=12)

        self.campo_entrada = ctk.CTkEntry(
            self.barra_entrada,
            placeholder_text="Escribe tu mensaje…",
            height=40,
            fg_color=COLOR_FONDO_ENTRADA,
            border_color=COLOR_ACENTO_ATENUADO,
            text_color=COLOR_TEXTO_PRINCIPAL,
            font=ctk.CTkFont(size=13),
        )
        self.campo_entrada.pack(
            side="left",
            fill="x",
            expand=True,
            padx=6,
            pady=12,
        )
        self.campo_entrada.bind(
            "<Return>",
            lambda _evento: self._al_enviar_click(),
        )

        self.boton_enviar = ctk.CTkButton(
            self.barra_entrada,
            text="Enviar",
            width=90,
            height=40,
            fg_color=COLOR_ACENTO,
            hover_color="#2c93cc",
            text_color="#0b0f14",
            font=ctk.CTkFont(weight="bold"),
            command=self._al_enviar_click,
        )
        self.boton_enviar.pack(side="left", padx=(6, 14), pady=12)

    # ─── Selección e Inicialización del Motor LLM ─────────────────────────────

    def _mostrar_bienvenida_seleccion(self) -> None:
        """Muestra una nota inicial invitando a elegir el motor antes de instanciar."""
        modelo_local: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        modelo_gemini: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._agregar_nota_sistema(
            "Bienvenido a AMADEUS. Selecciona el motor de IA en el selector superior "
            f"('Modelo Local (Ollama: {modelo_local})' o 'Modelo Gemini ({modelo_gemini})') "
            "y pulsa 'Iniciar Motor' (o envía un mensaje para iniciarlo automáticamente)."
        )

    def _obtener_proveedor_seleccionado(self) -> str:
        """Retorna el identificador del proveedor ('local' o 'gemini') según el ComboBox."""
        opcion_actual: str = self.selector_motor.get()
        return MAPA_OPCIONES_PROVEEDOR.get(opcion_actual, PROVEEDOR_LOCAL)

    def _al_cambiar_selector_motor(self, _opcion_elegida: str) -> None:
        """Callback ejecutado cuando el usuario cambia la opción del ComboBox."""
        if not self._inicializando_motor:
            self._al_conectar_motor_click()

    def _al_conectar_motor_click(self) -> None:
        """Inicia o cambia la instancia del modelo LLM y del agente en segundo plano."""
        if self._inicializando_motor:
            return

        proveedor: str = self._obtener_proveedor_seleccionado()
        self._inicializando_motor = True
        self.boton_conectar.configure(state="disabled", text="Conectando…")
        self.boton_enviar.configure(state="disabled")

        nombre_proveedor_ui: str = (
            "Google Gemini" if proveedor == PROVEEDOR_GEMINI else "Ollama Local"
        )
        self._establecer_estado(
            f"Iniciando {nombre_proveedor_ui}…",
            COLOR_ETIQUETA_TOOL,
        )
        self._agregar_nota_sistema(
            f"Configurando motor {nombre_proveedor_ui}…"
        )
        threading.Thread(
            target=self._trabajador_arranque_motor,
            args=(proveedor,),
            daemon=True,
        ).start()

    def _trabajador_arranque_motor(self, proveedor: str) -> None:
        """Instancia el modelo mediante `crear_modelo_llm` y lo inyecta en `crear_agente`."""
        try:
            if proveedor == PROVEEDOR_LOCAL:
                url_base: str = os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://127.0.0.1:11434",
                )
                nombre_modelo: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                conectado, mensaje = verificar_conexion_ollama(
                    url_base,
                    nombre_modelo,
                )
                if not conectado:
                    self._cola_ui.put(
                        ("fallo_motor", f"No se pudo conectar con Ollama:\n{mensaje}")
                    )
                    return
            else:
                nombre_modelo = (
                    os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
                    or "gemini-2.5-flash"
                )

            modelo_llm: BaseChatModel = crear_modelo_llm(proveedor)
            agente_compilado, _, modelo_resuelto = crear_agente(
                modelo_llm=modelo_llm,
                proveedor=proveedor,
            )

            self.agente = agente_compilado
            self.proveedor_activo = proveedor
            self.modelo_activo = modelo_resuelto or nombre_modelo

            if not self._voz_inicializada:
                self._inicializar_modulos_voz()
                self._voz_inicializada = True

            etiqueta_motor: str = (
                f"Gemini ({self.modelo_activo})"
                if proveedor == PROVEEDOR_GEMINI
                else f"Ollama ({self.modelo_activo})"
            )
            self._cola_ui.put(("motor_listo", etiqueta_motor))

        except ValueError as exc:
            self._cola_ui.put(("fallo_motor", str(exc)))
        except Exception as exc:
            self._cola_ui.put(
                ("fallo_motor", f"Error al inicializar el motor LLM: {exc}")
            )

    def _inicializar_modulos_voz(self) -> None:
        """Inicializa los componentes de entrada y salida de voz una sola vez."""
        try:
            from voice.output import VoiceOutput

            self._salida_voz = VoiceOutput()
        except Exception:
            self._salida_voz = None

        try:
            from voice.input import VoiceInput

            self._entrada_voz = VoiceInput()
        except Exception:
            self._entrada_voz = None
            self._cola_ui.put(("microfono_deshabilitado",))

    # ─── Cola de Eventos Hilo-Seguros ─────────────────────────────────────────

    def _procesar_cola_ui(self) -> None:
        """Despacha eventos acumulados desde los hilos de trabajo hacia la UI."""
        try:
            while True:
                evento = self._cola_ui.get_nowait()
                self._manejar_evento_ui(evento)
        except queue.Empty:
            pass
        self.after(60, self._procesar_cola_ui)

    def _manejar_evento_ui(self, evento: Tuple[Any, ...]) -> None:
        """Procesa un evento individual en el hilo principal de CustomTkinter."""
        tipo_evento: str = evento[0]

        if tipo_evento == "estado":
            _, texto, color = evento
            self._establecer_estado(texto, color)

        elif tipo_evento == "solo_estado":
            self._establecer_estado(evento[1], COLOR_ETIQUETA_TOOL)

        elif tipo_evento == "sistema":
            self._agregar_nota_sistema(evento[1])

        elif tipo_evento == "motor_listo":
            etiqueta_motor: str = evento[1]
            self._inicializando_motor = False
            self.boton_conectar.configure(
                state="normal",
                text="Cambiar Motor",
            )
            self.boton_enviar.configure(state="normal")
            self._establecer_estado(
                f"Conectado — {etiqueta_motor}",
                COLOR_EXITO,
            )
            self._agregar_nota_sistema(
                f"✓ Agente AMADEUS activo con {etiqueta_motor}."
            )
            if self._mensaje_en_espera is not None:
                mensaje_pendiente: str = self._mensaje_en_espera
                self._mensaje_en_espera = None
                self._despachar_mensaje_usuario(mensaje_pendiente)

        elif tipo_evento == "fallo_motor":
            detalle_error: str = evento[1]
            self._inicializando_motor = False
            self.boton_conectar.configure(
                state="normal",
                text="Iniciar Motor",
            )
            self.boton_enviar.configure(state="normal")
            self._mensaje_en_espera = None
            self._establecer_estado("Error de configuración", COLOR_ERROR)
            self._agregar_nota_sistema(
                f"❌ {detalle_error}",
                color=COLOR_ERROR,
            )

        elif tipo_evento == "microfono_deshabilitado":
            self.boton_microfono.configure(state="disabled")

        elif tipo_evento == "resultado_microfono":
            texto_voz: Optional[str] = evento[1]
            self._escuchando = False
            self.boton_microfono.configure(
                fg_color=COLOR_FONDO_ENTRADA,
                text="🎤",
            )
            if texto_voz:
                self._despachar_mensaje_usuario(texto_voz)

        elif tipo_evento == "iniciar_burbuja":
            self._burbuja_actual = BurbujaMensaje(
                self.marco_chat,
                rol="assistant",
            )
            self._burbuja_actual.grid(
                row=self._fila,
                column=0,
                sticky="ew",
                pady=2,
            )
            self._fila += 1
            self._desplazar_al_final()

        elif tipo_evento == "token":
            if self._burbuja_actual:
                self._burbuja_actual.agregar_texto(evento[1])
                self._desplazar_al_final()

        elif tipo_evento == "herramienta":
            if self._burbuja_actual:
                self._burbuja_actual.agregar_etiqueta_herramienta(evento[1])
                self._desplazar_al_final()

        elif tipo_evento == "texto_final":
            if not self._burbuja_actual:
                self._burbuja_actual = BurbujaMensaje(
                    self.marco_chat,
                    rol="assistant",
                )
                self._burbuja_actual.grid(
                    row=self._fila,
                    column=0,
                    sticky="ew",
                    pady=2,
                )
                self._fila += 1
            self._burbuja_actual.agregar_texto(evento[1])
            self._desplazar_al_final()

        elif tipo_evento == "error":
            self._agregar_nota_sistema(f"❌ {evento[1]}", color=COLOR_ERROR)
            self._establecer_listo()

        elif tipo_evento == "pendiente":
            self._mostrar_barra_confirmacion(evento[1])
            self._establecer_listo()

        elif tipo_evento == "finalizado":
            self._establecer_listo()

    # ─── Envío de Mensajes ────────────────────────────────────────────────────

    def _al_enviar_click(self) -> None:
        """Maneja el evento de envío de mensaje por botón o tecla Enter."""
        texto: str = self.campo_entrada.get().strip()
        if not texto and self._imagen_pendiente is None:
            return

        # Si el usuario aún no pulsó 'Iniciar Motor', instanciar con la opción actual
        if self.agente is None:
            if texto:
                self._mensaje_en_espera = texto
                self.campo_entrada.delete(0, "end")
            self._al_conectar_motor_click()
            return

        self.campo_entrada.delete(0, "end")

        if self._imagen_pendiente is not None:
            imagen = self._imagen_pendiente
            etiqueta = self._etiqueta_imagen_pendiente
            self._limpiar_adjunto()
            self._despachar_mensaje_con_imagen(texto, imagen, etiqueta)
        else:
            self._despachar_mensaje_usuario(texto)

    def _despachar_mensaje_usuario(self, texto: str) -> None:
        """Agrega la burbuja del usuario y ejecuta el comando o agente."""
        burbuja = BurbujaMensaje(self.marco_chat, rol="user", texto=texto)
        burbuja.grid(row=self._fila, column=0, sticky="ew", pady=2)
        self._fila += 1
        self._desplazar_al_final()

        texto_limpio: str = texto.strip().lower()

        accion_pendiente: Optional[PendingAction] = get_pending_action()
        if accion_pendiente:
            self._resolver_confirmacion(accion_pendiente, texto_limpio)
            return

        if self._intentar_comando_directo(texto):
            return

        self._establecer_ocupado("AMADEUS está pensando…")
        threading.Thread(
            target=self._trabajador_ejecutar_agente,
            args=(texto,),
            daemon=True,
        ).start()

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
        elif objetivo in PALABRAS_CLAVE_PANEL_SISTEMA:
            resultado = open_system_settings.invoke({"setting": objetivo})
        elif objetivo in {"explorador", "explorer", "file explorer"}:
            resultado = open_file_explorer.invoke({"path": ""})
        else:
            resultado = open_application.invoke({"app_name": objetivo})

        exito: bool = (
            resultado.startswith("✓")
            or resultado.startswith("Opened")
            or resultado.startswith("Abierto")
        )
        mensaje: str = (
            f"He abierto {objetivo}."
            if exito
            else f"No pude abrir {objetivo}."
        )
        self._cola_ui.put(("iniciar_burbuja",))
        self._cola_ui.put(("token", mensaje))
        self._hablar_asincrono(mensaje)
        return True

    # ─── Adjuntos (Imagen / Captura de Pantalla) ──────────────────────────────

    def _al_adjuntar_click(self) -> None:
        """Abre un cuadro de diálogo para adjuntar una imagen desde el disco."""
        ruta: str = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not ruta:
            return
        try:
            imagen = Image.open(ruta).convert("RGB")
        except Exception as exc:
            self._agregar_nota_sistema(
                f"❌ No se pudo abrir la imagen: {exc}",
                color=COLOR_ERROR,
            )
            return
        nombre: str = os.path.basename(ruta)
        self._establecer_imagen_pendiente(imagen, f"📎 {nombre}")

    def _al_capturar_click(self) -> None:
        """Minimiza brevemente la ventana y captura la pantalla actual."""
        self.iconify()
        self.after(280, self._realizar_captura)

    def _realizar_captura(self) -> None:
        """Ejecuta la captura de pantalla y restaura la ventana."""
        try:
            imagen = grab_screenshot()
        except Exception as exc:
            self.deiconify()
            self._agregar_nota_sistema(
                f"❌ No se pudo capturar la pantalla: {exc}",
                color=COLOR_ERROR,
            )
            return
        self.deiconify()
        self._establecer_imagen_pendiente(imagen, "📷 Captura de pantalla")

    def _establecer_imagen_pendiente(
        self,
        imagen: Image.Image,
        etiqueta: str,
    ) -> None:
        """Configura la imagen adjunta pendiente de envío en la barra inferior."""
        self._imagen_pendiente = imagen
        self._etiqueta_imagen_pendiente = etiqueta

        miniatura = imagen.copy()
        miniatura.thumbnail((72, 48))
        miniatura_ctk = ctk.CTkImage(
            light_image=miniatura,
            dark_image=miniatura,
            size=miniatura.size,
        )
        self.etiqueta_miniatura_adjunto.configure(image=miniatura_ctk)
        self.etiqueta_miniatura_adjunto.image = miniatura_ctk  # type: ignore[attr-defined]
        self.etiqueta_nombre_adjunto.configure(
            text=f"{etiqueta} — se enviará junto a tu mensaje"
        )

        self.barra_adjuntos.pack(
            side="bottom",
            fill="x",
            padx=14,
            pady=(0, 6),
            before=self.barra_entrada,
        )
        self.campo_entrada.focus_set()

    def _limpiar_adjunto(self) -> None:
        """Elimina la imagen adjunta pendiente y oculta la barra de adjuntos."""
        self._imagen_pendiente = None
        self._etiqueta_imagen_pendiente = ""
        self.barra_adjuntos.pack_forget()

    def _despachar_mensaje_con_imagen(
        self,
        texto: str,
        imagen: Image.Image,
        _etiqueta: str,
    ) -> None:
        """Muestra el mensaje con miniatura y lanza el análisis multimodal."""
        miniatura = imagen.copy()
        miniatura.thumbnail((220, 160))
        miniatura_ctk = ctk.CTkImage(
            light_image=miniatura,
            dark_image=miniatura,
            size=miniatura.size,
        )

        burbuja = BurbujaMensaje(
            self.marco_chat,
            rol="user",
            texto=texto or "(analiza esta imagen)",
            miniatura=miniatura_ctk,
        )
        burbuja.grid(row=self._fila, column=0, sticky="ew", pady=2)
        self._fila += 1
        self._desplazar_al_final()

        self._establecer_ocupado("Analizando imagen…")
        threading.Thread(
            target=self._trabajador_agente_con_imagen,
            args=(texto, imagen),
            daemon=True,
        ).start()

    def _trabajador_agente_con_imagen(
        self,
        texto: str,
        imagen: Image.Image,
    ) -> None:
        """Analiza la imagen adjunta y envía el contexto combinado al agente."""
        resultado_vision: str = analyze_image(imagen, question=texto)

        if texto.strip():
            mensaje_combinado: str = (
                f"El usuario adjuntó una imagen y preguntó: «{texto.strip()}»\n\n"
                f"[Análisis del modelo de visión sobre la imagen]:\n{resultado_vision}"
            )
        else:
            mensaje_combinado = (
                "El usuario adjuntó una imagen sin pregunta específica. "
                "Analízala y responde de forma útil.\n\n"
                f"[Análisis del modelo de visión sobre la imagen]:\n{resultado_vision}"
            )

        self._cola_ui.put(("solo_estado", "AMADEUS está pensando…"))
        self._trabajador_ejecutar_agente(mensaje_combinado)

    # ─── Extracción e Invocación del Agente ───────────────────────────────────

    @staticmethod
    def _extraer_texto_contenido(contenido: Any) -> str:
        """
        Extrae el texto útil del atributo `content` de un mensaje AI.
        Soporta cadenas y listas de cadenas o diccionarios (formato Gemini/Ollama).
        """
        if not contenido:
            return ""
        if isinstance(contenido, str):
            return contenido
        if isinstance(contenido, list):
            fragmentos: List[str] = []
            for bloque in contenido:
                if isinstance(bloque, str):
                    fragmentos.append(bloque)
                elif isinstance(bloque, dict):
                    if (
                        bloque.get("type") == "thinking"
                        or bloque.get("thought") is True
                    ):
                        continue
                    texto_bloque: Any = bloque.get(
                        "text",
                        bloque.get("content", ""),
                    )
                    if isinstance(texto_bloque, str) and texto_bloque:
                        fragmentos.append(texto_bloque)
            return "".join(fragmentos)
        return ""

    def _trabajador_ejecutar_agente(self, mensaje_usuario: str) -> None:
        """Invoca el agente LangGraph en un hilo secundario y transmite tokens y voz."""
        configuracion: Dict[str, Any] = {
            "configurable": {"thread_id": self.id_hilo}
        }
        carga_entrada: Dict[str, Any] = {
            "messages": [{"role": "user", "content": mensaje_usuario}]
        }

        partes_respuesta: List[str] = []
        buffer_oracion: str = ""
        oraciones_habladas: bool = False
        herramienta_actual: Optional[str] = None
        burbuja_iniciada: bool = False

        try:
            for fragmento in self.agente.stream(
                carga_entrada,
                config=configuracion,
                stream_mode="messages",
            ):
                mensaje, metadatos = fragmento
                nodo: str = metadatos.get("langgraph_node", "")

                if nodo == "tools":
                    nombre_tool: Optional[str] = getattr(mensaje, "name", None)
                    if nombre_tool and nombre_tool != herramienta_actual:
                        herramienta_actual = nombre_tool
                        if not burbuja_iniciada:
                            self._cola_ui.put(("iniciar_burbuja",))
                            burbuja_iniciada = True
                        self._cola_ui.put(("herramienta", herramienta_actual))
                    if not "".join(partes_respuesta).strip():
                        partes_respuesta.clear()
                    continue

                tipo_mensaje: str = getattr(mensaje, "type", "")
                if tipo_mensaje not in {"ai", "AIMessage", "AIMessageChunk"}:
                    continue

                texto_fragmento: str = self._extraer_texto_contenido(
                    getattr(mensaje, "content", None)
                )
                if not texto_fragmento:
                    continue

                if not burbuja_iniciada:
                    self._cola_ui.put(("iniciar_burbuja",))
                    burbuja_iniciada = True

                self._cola_ui.put(("token", texto_fragmento))
                partes_respuesta.append(texto_fragmento)

                # Síntesis de voz iterativa por oración terminada en punto
                buffer_oracion += texto_fragmento
                while True:
                    coincidencia_fin = PATRON_FIN_ORACION.search(buffer_oracion)
                    if not coincidencia_fin:
                        break
                    indice_corte: int = coincidencia_fin.end()
                    oracion_completa: str = buffer_oracion[
                        :indice_corte
                    ].strip()
                    buffer_oracion = buffer_oracion[indice_corte:]
                    if oracion_completa:
                        self._hablar_asincrono(oracion_completa)
                        oraciones_habladas = True

        except Exception as exc:
            self._cola_ui.put(("error", f"Error del agente: {exc}"))
            self._hablar_asincrono(
                "Ha ocurrido un error. Por favor, inténtalo de nuevo."
            )
            return

        partes_con_texto: List[str] = [
            parte for parte in partes_respuesta if parte.strip()
        ]
        if not partes_con_texto:
            partes_respuesta.clear()

        # Forzar recuperación del último mensaje AI si partes_respuesta quedó vacío
        if not partes_respuesta:
            respuesta_estado: Optional[str] = self._obtener_ultimo_mensaje_ia()
            if respuesta_estado:
                self._cola_ui.put(("texto_final", respuesta_estado))
                partes_respuesta.append(respuesta_estado)

        respuesta_final: str = "".join(partes_respuesta).strip()
        if not respuesta_final:
            respuesta_final = (self._obtener_ultimo_mensaje_ia() or "").strip()

        accion_pendiente: Optional[PendingAction] = get_pending_action()
        if accion_pendiente:
            self._cola_ui.put(("pendiente", accion_pendiente))
            texto_pendiente_voz: str = (
                f"{buffer_oracion.strip() if oraciones_habladas else respuesta_final}. "
                f"{accion_pendiente.description}. "
                "Confirma para continuar o cancela para abortar."
            ).strip()
            self._hablar_asincrono(texto_pendiente_voz)
        else:
            if oraciones_habladas:
                remanente: str = buffer_oracion.strip()
                if remanente:
                    self._hablar_asincrono(remanente)
            elif respuesta_final:
                self._hablar_asincrono(respuesta_final)
            reset_all()
            self._cola_ui.put(("finalizado",))

    def _obtener_ultimo_mensaje_ia(self) -> Optional[str]:
        """Recupera el último mensaje AI desde el estado del grafo como respaldo."""
        try:
            configuracion: Dict[str, Any] = {
                "configurable": {"thread_id": self.id_hilo}
            }
            estado = self.agente.get_state(configuracion)
            mensajes: List[Any] = estado.values.get("messages", [])
            for msg in reversed(mensajes):
                tipo_msg: str = getattr(msg, "type", "")
                if tipo_msg not in {"ai", "AIMessage", "AIMessageChunk"}:
                    continue
                texto_extraido: str = self._extraer_texto_contenido(
                    getattr(msg, "content", None)
                ).strip()
                if texto_extraido:
                    return texto_extraido
        except Exception:
            pass
        return None

    # ─── Confirmaciones HITL ──────────────────────────────────────────────────

    def _mostrar_barra_confirmacion(
        self,
        accion_pendiente: PendingAction,
    ) -> None:
        """Muestra la barra visual de confirmación para acciones destructivas."""
        self.etiqueta_confirmacion.configure(
            text=f"⚠️  Confirmación requerida: {accion_pendiente.description}"
        )
        self.barra_confirmacion.pack(
            side="bottom",
            fill="x",
            padx=14,
            pady=(0, 6),
            before=self.barra_entrada,
        )

    def _resolver_confirmacion(
        self,
        accion_pendiente: PendingAction,
        texto_minusculas: str,
    ) -> None:
        """Resuelve una confirmación pendiente mediante entrada de texto."""
        if texto_minusculas in {
            "confirmar",
            "confirm",
            "sí",
            "si",
            "yes",
            "adelante",
            "procede",
        }:
            self._al_confirmar_click()
        elif texto_minusculas in {"cancelar", "cancel", "no", "cancela"}:
            self._al_cancelar_click()
        else:
            self._agregar_nota_sistema(
                f"Tengo pendiente tu confirmación: «{accion_pendiente.description}». "
                "Usa los botones o escribe 'confirmar' / 'cancelar'.",
                color=COLOR_ETIQUETA_TOOL,
            )

    def _al_confirmar_click(self) -> None:
        """Aprueba la acción pendiente y reanuda la ejecución del agente."""
        accion_pendiente: Optional[PendingAction] = get_pending_action()
        if not accion_pendiente:
            return
        confirm(accion_pendiente.key)
        self.barra_confirmacion.pack_forget()
        self._agregar_nota_sistema("Entendido, procediendo con la acción…")
        self._establecer_ocupado("AMADEUS está procesando…")
        threading.Thread(
            target=self._trabajador_ejecutar_agente,
            args=(
                "El usuario ha confirmado la acción. Por favor, procede exactamente "
                "con la misma llamada a la herramienta que hiciste antes.",
            ),
            daemon=True,
        ).start()

    def _al_cancelar_click(self) -> None:
        """Cancela la acción pendiente y limpia el estado HITL."""
        accion_pendiente: Optional[PendingAction] = get_pending_action()
        if accion_pendiente:
            cancel(accion_pendiente.key)
        reset_all()
        self.barra_confirmacion.pack_forget()
        self._agregar_nota_sistema(
            "Acción cancelada. No se ha modificado ningún archivo."
        )
        self._hablar_asincrono(
            "Acción cancelada. No se ha modificado ningún archivo."
        )

    # ─── Voz ──────────────────────────────────────────────────────────────────

    def _al_microfono_click(self) -> None:
        """Inicia la captura de voz por micrófono en un hilo secundario."""
        if self._entrada_voz is None or self._escuchando:
            return
        self._escuchando = True
        self.boton_microfono.configure(fg_color=COLOR_ACENTO, text="●")
        threading.Thread(
            target=self._trabajador_escuchar_microfono,
            daemon=True,
        ).start()

    def _trabajador_escuchar_microfono(self) -> None:
        """Escucha una frase del micrófono y envía el resultado a la cola UI."""
        texto: Optional[str] = self._entrada_voz.listen()
        self._cola_ui.put(("resultado_microfono", texto))

    def _hablar_asincrono(self, texto: str) -> None:
        """Sintetiza voz en un hilo secundario sin bloquear la interfaz gráfica."""
        if os.getenv("NO_TTS", "false").lower() == "true":
            return
        texto_limpio: str = texto.strip()
        if texto_limpio and self._salida_voz is not None:
            threading.Thread(
                target=self._salida_voz.speak,
                args=(texto_limpio,),
                daemon=True,
            ).start()

    # ─── Utilidades de UI ─────────────────────────────────────────────────────

    def _agregar_nota_sistema(
        self,
        texto: str,
        color: str = COLOR_TEXTO_ATENUADO,
    ) -> None:
        """Inserta una nota informativa del sistema en el área de chat."""
        nota = ctk.CTkLabel(
            self.marco_chat,
            text=texto,
            text_color=color,
            font=ctk.CTkFont(size=11, slant="italic"),
            wraplength=680,
            justify="left",
        )
        nota.grid(row=self._fila, column=0, sticky="w", padx=14, pady=4)
        self._fila += 1
        self._desplazar_al_final()

    def _desplazar_al_final(self) -> None:
        """Desplaza automáticamente el área de chat hasta el último mensaje."""
        self.marco_chat._parent_canvas.yview_moveto(1.0)

    def _establecer_estado(self, texto: str, color: str) -> None:
        """Actualiza el indicador y texto de estado en el encabezado."""
        self.etiqueta_estado.configure(text=texto, text_color=color)
        self.punto_estado.configure(text_color=color)

    def _establecer_ocupado(self, texto: str) -> None:
        """Deshabilita controles mientras el agente procesa una solicitud."""
        self.boton_enviar.configure(state="disabled")
        self.boton_conectar.configure(state="disabled")
        self._establecer_estado(texto, COLOR_ETIQUETA_TOOL)

    def _establecer_listo(self) -> None:
        """Restaura el estado interactivo una vez finalizada la respuesta."""
        self.boton_enviar.configure(state="normal")
        self.boton_conectar.configure(state="normal", text="Cambiar Motor")
        self._burbuja_actual = None
        self._herramienta_actual = None
        self._escuchando = False
        self.boton_microfono.configure(fg_color=COLOR_FONDO_ENTRADA, text="🎤")
        etiqueta_motor: str = (
            f"Gemini ({self.modelo_activo})"
            if self.proveedor_activo == PROVEEDOR_GEMINI
            else f"Ollama ({self.modelo_activo})"
        )
        self._establecer_estado(
            f"Conectado — {etiqueta_motor}",
            COLOR_EXITO,
        )


# Alias de compatibilidad
AmadeusGUI = InterfazAmadeus


def analizar_argumentos_gui() -> argparse.Namespace:
    """Analiza los argumentos opcionales de arranque para la interfaz gráfica."""
    analizador = argparse.ArgumentParser(
        description="AMADEUS — Interfaz Gráfica (Local Ollama / Google Gemini)"
    )
    analizador.add_argument(
        "--proveedor",
        "-p",
        choices=[PROVEEDOR_LOCAL, PROVEEDOR_GEMINI],
        default=os.getenv("AMADEUS_PROVIDER", "").strip().lower() or None,
        help="Proveedor LLM inicial ('local' o 'gemini').",
    )
    analizador.add_argument(
        "--auto-iniciar",
        action="store_true",
        help="Conectar automáticamente el motor seleccionado al abrir la ventana.",
    )
    return analizador.parse_args()


def main() -> None:
    """Punto de entrada principal de la interfaz gráfica."""
    argumentos = analizar_argumentos_gui()
    aplicacion = InterfazAmadeus(
        proveedor_inicial=argumentos.proveedor,
        auto_iniciar=argumentos.auto_iniciar,
    )
    aplicacion.mainloop()


if __name__ == "__main__":
    main()
