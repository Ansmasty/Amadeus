"""
voice/output.py

Síntesis de voz para AMADEUS.
Pipeline: Texto → TTS (pyttsx3/edge-tts) → RVC (voz Kurisu) → Reproducción
"""
import os
import threading
import tempfile
import subprocess
from pathlib import Path

import pyttsx3
import numpy as np
import soundfile as sf


class VoiceOutput:
    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()
        self._rvc = None
        self._use_rvc = os.getenv("USE_RVC", "true").lower() == "true"
        self._use_edge = os.getenv("USE_EDGE_TTS", "false").lower() == "true"

        self._init_tts()
        if self._use_rvc:
            self._init_rvc()

    # ─── Inicialización ───────────────────────────────────────────────────────

    def _init_tts(self) -> None:
        """Inicializa el motor TTS base (pyttsx3 o edge-tts)."""
        if self._use_edge:
            try:
                import edge_tts  # noqa: F401
                print("[TTS] Usando edge-tts como voz base para RVC.")
            except ImportError:
                print("[TTS] edge-tts no disponible, usando pyttsx3.")
                self._use_edge = False

        if not self._use_edge:
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
            if any(k in voice.name.lower()
                   for k in ["sabina", "helena", "zira", "spanish", "español"]):
                self._engine.setProperty("voice", voice.id)
                print(f"[TTS] Voz base pyttsx3: {voice.name}")
                break

    def _init_rvc(self) -> None:
        """Inicializa el conversor RVC."""
        self.rvc_converter = None
        self.rvc_enabled = False
        self._rvc = None

        if not self._use_rvc:
            return

        try:
            from voice.rvc_converter import RVCConverter
            self.rvc_converter = RVCConverter()
            self.rvc_converter._load_rvc()  # si falla, lanza excepción

            self.rvc_enabled = True
            self._rvc = self.rvc_converter  # <- clave
            model_ref = getattr(self.rvc_converter, "model_path", "")
            model_name = Path(str(model_ref)).name if model_ref else "N/A"
            print(f"[RVC] Modelo activo: {model_name}")
        except Exception as e:
            self.rvc_enabled = False
            print(f"[RVC] No disponible: {e}. Usando TTS sin conversión.")
            self._rvc = None
            self._use_rvc = False

    # ─── Síntesis principal ───────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Pipeline completo: Texto → TTS → RVC → Reproducción."""
        if not self._engine and not self._use_edge:
            print(f"[SIN VOZ] {text}")
            return

        clean = self._clean_for_speech(text)
        if not clean.strip():
            return

        with self._lock:
            if self._use_rvc and self.rvc_enabled and self._rvc:
                try:
                    self._speak_with_rvc(clean)
                    return
                except Exception as e:
                    print(f"[RVC] Error en turno, fallback TTS: {e}")
                    # NO apagar RVC globalmente aquí
            self._speak_direct(clean)

    def _speak_with_rvc(self, text: str) -> None:
        """TTS → WAV temporal → RVC → Reproducir."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tts_wav = os.path.join(tmpdir, "tts_raw.wav")
                kurisu_wav = os.path.join(tmpdir, "kurisu.wav")

                ok_tts = self._generate_tts_wav(text, tts_wav)
                if not ok_tts:
                    self._speak_direct(text)
                    return

                if hasattr(self._rvc, "convert"):
                    ok_rvc = self._rvc.convert(tts_wav, kurisu_wav)
                else:
                    self._rvc.convert_file(tts_wav, kurisu_wav)
                    ok_rvc = Path(kurisu_wav).exists() and Path(kurisu_wav).stat().st_size > 0

                if not ok_rvc:
                    print("[RVC] Conversión fallida, usando TTS sin conversión.")
                    self._play_wav(tts_wav)
                    return

                # después de convertir a kurisu_wav:
                self._boost_if_low(kurisu_wav)
                self._play_wav(kurisu_wav)

        except Exception as exc:
            print(f"[RVC] Error en pipeline: {exc}")
            self._speak_direct(text)

    def _generate_tts_wav(self, text: str, output_path: str) -> bool:
        """Genera un archivo WAV con pyttsx3 o edge-tts."""
        if self._use_edge:
            return self._generate_edge_wav(text, output_path)
        return self._generate_pyttsx3_wav(text, output_path)

    def _generate_pyttsx3_wav(self, text: str, output_path: str) -> bool:
        """Genera WAV usando pyttsx3."""
        if not self._engine:
            return False
        try:
            self._engine.save_to_file(text, output_path)
            self._engine.runAndWait()
            return Path(output_path).exists() and Path(output_path).stat().st_size > 0
        except Exception as exc:
            print(f"[TTS] Error generando WAV pyttsx3: {exc}")
            return False

    def _generate_edge_wav(self, text: str, output_path: str) -> bool:
        """Genera WAV usando edge-tts (voz neural de mejor calidad)."""
        try:
            import asyncio
            import edge_tts

            voice = os.getenv("TTS_VOICE", "es-ES-ElviraNeural")
            rate  = os.getenv("TTS_EDGE_RATE", "+0%")

            # edge-tts genera MP3, necesitamos convertir a WAV para RVC
            mp3_path = output_path.replace(".wav", ".mp3")

            async def _gen():
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                await communicate.save(mp3_path)

            asyncio.run(_gen())

            # Convertir MP3 → WAV con ffmpeg (si está disponible)
            if self._mp3_to_wav(mp3_path, output_path):
                return True

            # Fallback: intentar con pydub
            return self._pydub_convert(mp3_path, output_path)

        except Exception as exc:
            print(f"[TTS] Error edge-tts: {exc}")
            return False

    def _mp3_to_wav(self, mp3_path: str, wav_path: str) -> bool:
        """Convierte MP3 a WAV usando ffmpeg."""
        import shutil
        if not shutil.which("ffmpeg"):
            return False
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path,
                 "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False

    def _pydub_convert(self, mp3_path: str, wav_path: str) -> bool:
        """Convierte MP3 a WAV usando pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")
            return True
        except Exception as exc:
            print(f"[TTS] pydub no disponible: {exc}")
            return False

    def _speak_direct(self, text: str) -> None:
        """TTS directo sin RVC (fallback)."""
        if self._use_edge:
            try:
                import asyncio, edge_tts, tempfile as tf
                async def _play():
                    voice = os.getenv("TTS_VOICE", "es-ES-ElviraNeural")
                    with tf.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        path = tmp.name
                    await edge_tts.Communicate(text, voice).save(path)
                    self._play_audio_file(path)
                    os.unlink(path)
                asyncio.run(_play())
                return
            except Exception:
                pass

        if self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                print(f"[TTS] Error: {exc}")

    # ─── Reproducción de audio ────────────────────────────────────────────────

    def _play_wav(self, path: str) -> None:
        """Reproduce un archivo WAV."""
        self._play_audio_file(path)

    def _play_audio_file(self, path: str) -> None:
        """Reproduce cualquier archivo de audio (WAV o MP3)."""
        # Opción 1: pygame (más confiable)
        try:
            import pygame
            pygame.mixer.init(frequency=44100)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
            return
        except ImportError:
            pass
        
        # Opción 3: Windows PowerShell nativo
        if os.name == "nt":
            try:
                subprocess.run(
                    ["powershell", "-c",
                     f'Add-Type -AssemblyName PresentationCore;'
                     f'$p=New-Object System.Windows.Media.MediaPlayer;'
                     f'$p.Open("{path}");$p.Play();Start-Sleep -s 5'],
                    capture_output=True, timeout=30
                )
            except Exception as exc:
                print(f"[Audio] No se pudo reproducir: {exc}")

    def stop(self) -> None:
        """Detiene la síntesis en curso."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Elimina markdown para lectura natural."""
        import re
        text = re.sub(r"```[\s\S]*?```", " código omitido ", text)
        text = re.sub(r"[*_`#>|]", "", text)
        text = re.sub(r"https?://\S+", "el enlace", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _boost_if_low(self, wav_path: str, target_rms: float = 0.03):
        """Boosts the audio level if it's below the target RMS."""
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        if rms > 0 and rms < target_rms:
            gain = min(8.0, target_rms / rms)
            audio = np.clip(audio * gain, -1.0, 1.0)
            sf.write(wav_path, audio, sr)