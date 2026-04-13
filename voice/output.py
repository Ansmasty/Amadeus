"""
voice/output.py

Síntesis de voz para AMADEUS.
Pipeline: Texto → TTS local (pyttsx3) → Reproducción
"""
import os
import re
import threading

import pyttsx3


class VoiceOutput:
    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()
        self._init_tts()

    def _init_tts(self) -> None:
        """Inicializa el motor TTS base con pyttsx3."""
        try:
            self._engine = pyttsx3.init()
            self._configure_pyttsx3()
        except Exception as exc:
            print(f"[TTS] Error inicializando pyttsx3: {exc}")
            self._engine = None

    def _configure_pyttsx3(self) -> None:
        """Configura velocidad, volumen e idioma de pyttsx3."""
        if not self._engine:
            return

        rate = int(os.getenv("TTS_RATE", "165"))
        volume = float(os.getenv("TTS_VOLUME", "1.0"))
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)

        voices = self._engine.getProperty("voices")
        for voice in voices:
            if any(k in voice.name.lower() for k in ["sabina", "helena", "zira", "spanish", "español"]):
                self._engine.setProperty("voice", voice.id)
                print(f"[TTS] Voz base pyttsx3: {voice.name}")
                break

    def speak(self, text: str) -> None:
        """Convierte texto a voz con el motor local."""
        clean = self._clean_for_speech(text)
        if not clean.strip():
            return

        if not self._engine:
            print(f"[SIN VOZ] {clean}")
            return

        with self._lock:
            try:
                self._engine.say(clean)
                self._engine.runAndWait()
            except Exception as exc:
                print(f"[TTS] Error: {exc}")

    def stop(self) -> None:
        """Detiene la síntesis en curso."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Elimina markdown y URLs para una lectura natural."""
        text = re.sub(r"```[\s\S]*?```", " código omitido ", text)
        text = re.sub(r"[*_`#>|]", "", text)
        text = re.sub(r"https?://\S+", "el enlace", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
