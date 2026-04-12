"""
voice/rvc_converter.py
Conversión de voz usando rvc-python (compatible con Python 3.13).
Transforma audio TTS genérico a la voz de Makise Kurisu.
"""
import os
import sys
import inspect
from pathlib import Path
import torch
from rvc_python.infer import RVCInference

# ─── Configuración ────────────────────────────────────────────────────────────
RVC_MODEL_PATH   = os.getenv("RVC_MODEL_PATH",  "models/KurisuRVCv147.pth")
RVC_INDEX_PATH   = os.getenv("RVC_INDEX_PATH",  "models/KurisuRVCv147.index")
PITCH_SHIFT      = int(os.getenv("RVC_PITCH",         "0"))
INDEX_RATE       = float(os.getenv("RVC_INDEX_RATE",  "0.75"))
FILTER_RADIUS    = int(os.getenv("RVC_FILTER_RADIUS", "3"))
PROTECT          = float(os.getenv("RVC_PROTECT",     "0.33"))
F0_METHOD        = os.getenv("RVC_F0_METHOD",         "rmvpe")


class RVCConverter:
    """Convierte audio WAV a la voz de Kurisu usando rvc-python."""

    def __init__(self, *args, **kwargs):
        self._rvc = None
        self._loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rvc_version = (os.getenv("RVC_VERSION", "v2") or "v2").strip()

        model_env = (os.getenv("RVC_MODEL_PATH", "") or "").strip()
        index_env = (os.getenv("RVC_INDEX_PATH", "") or "").strip()

        self.model_path = Path(model_env) if model_env else None
        self.index_path = Path(index_env) if index_env else None

        self.rvc = RVCInference(device=self.device)

        # opcional: rutas absolutas para evitar problemas de cwd
        if self.model_path and not os.path.isabs(self.model_path):
            self.model_path = os.path.abspath(self.model_path)
        if self.index_path and not os.path.isabs(self.index_path):
            self.index_path = os.path.abspath(self.index_path)

        self._load_rvc()

    def _safe_name(self, p):
        return Path(str(p)).name if p else "N/A"

    def _load_rvc(self) -> None:
        if self.rvc is None:
            self.rvc = RVCInference(device=self.device)

        if not self.model_path:
            raise RuntimeError("RVC_MODEL_PATH no configurado")

        model_path = getattr(self, "model_path", "")
        index_path = getattr(self, "index_path", "")

        print(f"[RVC] Cargando modelo {self._safe_name(model_path)} en {self.device}...")

        self.rvc.load_model(str(self.model_path), version=self.rvc_version, index_path=str(self.index_path) if self.index_path else "")
        print(f"[RVC] Modelo {Path(str(self.model_path)).name} cargado.")

        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def convert_file(self, input_path: str, output_path: str, f0_up_key: int = 0, index_rate: float = 0.75, protect: float = 0.33):
        sig = inspect.signature(self.rvc.infer_file)
        p = sig.parameters
        kwargs = {}

        # input/output
        if "input_path" in p:
            kwargs["input_path"] = input_path
        elif "audio_input_path" in p:
            kwargs["audio_input_path"] = input_path

        if "output_path" in p:
            kwargs["output_path"] = output_path
        elif "audio_output_path" in p:
            kwargs["audio_output_path"] = output_path

        # pitch/key
        if "f0_up_key" in p:
            kwargs["f0_up_key"] = f0_up_key
        elif "key" in p:
            kwargs["key"] = f0_up_key
        elif "transpose" in p:
            kwargs["transpose"] = f0_up_key
        elif "pitch" in p:
            kwargs["pitch"] = f0_up_key

        # index/protect
        if "index_rate" in p:
            kwargs["index_rate"] = index_rate
        elif "index_ratio" in p:
            kwargs["index_ratio"] = index_rate

        if "protect" in p:
            kwargs["protect"] = protect

        return self.rvc.infer_file(**kwargs)