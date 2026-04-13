"""
voice/input.py

Captura de voz para AMADEUS.
Usa SpeechRecognition con Google (online) o faster-whisper (offline).
"""
import os
import speech_recognition as sr


def list_microphones() -> list[tuple[int, str]]:
    """
    Retorna lista de (índice, nombre) SOLO de dispositivos de ENTRADA (micrófonos).
    Filtra altavoces, outputs y dispositivos que no son micrófonos.
    """
    import pyaudio

    pa = pyaudio.PyAudio()
    mics = []

    try:
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                # Solo incluir dispositivos con canales de entrada disponibles
                if info.get("maxInputChannels", 0) > 0:
                    name = info.get("name", f"Dispositivo {i}")
                    mics.append((i, name))
            except Exception:
                continue
    finally:
        pa.terminate()

    return mics


def pick_microphone() -> int | None:
    """Retorna el índice configurado en MICROPHONE_INDEX o None para el predeterminado."""
    saved = os.getenv("MICROPHONE_INDEX", "").strip()
    if saved.lstrip("-").isdigit():
        idx = int(saved)
        if idx >= 0:
            return idx
        return None

    return None


def _test_microphone(device_index: int | None) -> bool:
    """
    Verifica que el micrófono se puede abrir sin lanzar excepción.
    Retorna True si el dispositivo es funcional.
    """
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
        )
        stream.close()
        return True
    except Exception as exc:
        print(f"[VOZ] Test de micrófono fallido (índice {device_index}): {exc}")
        return False
    finally:
        pa.terminate()


class VoiceInput:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 1.0
        self.recognizer.dynamic_energy_threshold = True
        self.use_whisper = os.getenv("USE_WHISPER", "false").lower() == "true"
        self._whisper_model = None
        self._mic_index: int | None = None

        # Selección y validación del micrófono sin menú interactivo
        self._mic_index = self._resolve_microphone()

        if self.use_whisper:
            self._load_whisper()

    def _resolve_microphone(self) -> int | None:
        """
        Usa el micrófono configurado por entorno si existe.
        En caso contrario, usa el dispositivo por defecto sin mostrar menú.
        """
        saved = os.getenv("MICROPHONE_INDEX", "").strip()
        if saved.lstrip("-").isdigit():
            chosen = int(saved)
            if chosen >= 0:
                print(f"[VOZ] Usando micrófono configurado por entorno: [{chosen}]")
                return chosen
            print("[VOZ] Usando dispositivo por defecto del sistema.")
            return None

        print("[VOZ] Usando dispositivo por defecto del sistema.")
        if _test_microphone(None):
            print("[VOZ] ✓ Dispositivo por defecto funcional.")
        else:
            print("[VOZ] ⚠ No se pudo verificar el micrófono predeterminado. Se intentará igualmente.")
        return None

    def _load_whisper(self):
        """Carga el modelo Whisper (offline) la primera vez."""
        try:
            from faster_whisper import WhisperModel
            whisper_model_size = os.getenv("WHISPER_MODEL", "base")
            print(f"[VOZ] Cargando modelo Whisper '{whisper_model_size}'...")
            self._whisper_model = WhisperModel(
                whisper_model_size,
                device="cpu",
                compute_type="int8"
            )
            print("[VOZ] Modelo Whisper listo.")
        except ImportError:
            print("[VOZ] faster-whisper no disponible. Usando Google Speech Recognition.")
            self.use_whisper = False

    def _get_microphone(self) -> sr.Microphone:
        """Retorna el objeto Microphone con el índice guardado, o el por defecto."""
        if self._mic_index is not None and self._mic_index >= 0:
            return sr.Microphone(device_index=self._mic_index)
        return sr.Microphone()

    def listen(self, timeout: int = 10, phrase_limit: int = 15) -> str | None:
        """
        Escucha el micrófono y retorna el texto reconocido.
        Retorna None si no se detecta audio o hay un error.
        """
        # Construir el micrófono ANTES del with para capturar errores de apertura
        try:
            mic = self._get_microphone()
        except Exception as exc:
            print(f"[VOZ] No se pudo crear el micrófono: {exc}")
            return None

        try:
            with mic as source:
                print("\n🎤 Escuchando... (habla ahora)")
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_limit
                    )
                except sr.WaitTimeoutError:
                    print("[VOZ] No se detectó audio.")
                    return None
                except Exception as exc:
                    print(f"[VOZ] Error al capturar audio: {exc}")
                    return None
        except AttributeError:
            # Stream nunca se inicializó — el dispositivo no es de entrada
            print(
                f"[VOZ] ❌ El dispositivo [{self._mic_index}] no es un micrófono de entrada.\n"
                f"      Reinicia AMADEUS y elige un dispositivo de la lista de micrófonos."
            )
            # Limpiar la selección guardada para forzar nueva elección
            os.environ.pop("MICROPHONE_INDEX", None)
            return None
        except Exception as exc:
            print(f"[VOZ] Error inesperado con el micrófono: {exc}")
            return None

        print("🔄 Procesando audio...")
        return self._transcribe(audio)

    def _transcribe(self, audio: sr.AudioData) -> str | None:
        """Transcribe el audio usando Whisper o Google."""
        if self.use_whisper and self._whisper_model:
            return self._transcribe_whisper(audio)
        return self._transcribe_google(audio)

    def _transcribe_whisper(self, audio: sr.AudioData) -> str | None:
        """Transcripción offline con faster-whisper."""
        import numpy as np
        try:
            wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
            audio_array = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            lang = os.getenv("VOICE_LANGUAGE", "es")
            segments, _ = self._whisper_model.transcribe(
                audio_array,
                language=lang,
                beam_size=5
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception as exc:
            print(f"[VOZ] Error Whisper: {exc}")
            return None

    def _transcribe_google(self, audio: sr.AudioData) -> str | None:
        """Transcripción online con Google Speech Recognition (gratis, requiere internet)."""
        lang = os.getenv("VOICE_LANGUAGE", "es-ES")
        try:
            text = self.recognizer.recognize_google(audio, language=lang)
            return text.strip() if text else None
        except sr.UnknownValueError:
            print("[VOZ] No se entendió el audio.")
            return None
        except sr.RequestError as exc:
            print(f"[VOZ] Error de conexión con Google: {exc}")
            return None