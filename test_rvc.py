from pathlib import Path
import inspect
import torch
import soundfile as sf
import numpy as np
from rvc_python.infer import RVCInference

model = "models/KurisuRVCv147.pth"
index = "models/KurisuRVCv147.index"
input_wav = "test_input.wav"
output_wav = "kurisu_output.wav"

if not Path(input_wav).exists():
    raise FileNotFoundError(f"No existe {input_wav}. Usa un WAV con voz real (3-10s).")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

rvc = RVCInference(device=device)
rvc.load_model(model, version="v2", index_path=index)
print("Modelo RVC cargado correctamente.")

sig = inspect.signature(rvc.infer_file)
p = sig.parameters
kwargs = {}

# Mapeo compatible entre versiones
if "input_path" in p:
    kwargs["input_path"] = input_wav
elif "input_audio_path" in p:
    kwargs["input_audio_path"] = input_wav

if "output_path" in p:
    kwargs["output_path"] = output_wav
elif "audio_output_path" in p:
    kwargs["audio_output_path"] = output_wav

if "f0_up_key" in p:
    kwargs["f0_up_key"] = 0
elif "key" in p:
    kwargs["key"] = 0
elif "transpose" in p:
    kwargs["transpose"] = 0

if "index_rate" in p:
    kwargs["index_rate"] = 0.75
elif "index_ratio" in p:
    kwargs["index_ratio"] = 0.75

if "protect" in p:
    kwargs["protect"] = 0.33

def normalize_wav(in_path: str, out_path: str, target_peak: float = 0.95):
    audio, sr = sf.read(in_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 0:
        audio = audio * (target_peak / peak)
    sf.write(out_path, audio, sr)
    return out_path

norm_input = "test_input_norm.wav"
normalize_wav(input_wav, norm_input)
kwargs["input_path"] = norm_input

print("infer_file signature:", sig)
print("kwargs usados:", kwargs)

result = rvc.infer_file(**kwargs)
print("result:", result)

if not Path(output_wav).exists():
    raise RuntimeError("No se generó kurisu_output.wav")

audio, sr = sf.read(output_wav)
rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
dur = len(audio) / sr if sr else 0.0
print(f"Salida: sr={sr}, dur={dur:.2f}s, rms={rms:.6f}")

if dur < 1.5 or rms < 0.001:
    raise RuntimeError("Salida casi silenciosa. Regraba test_input.wav con voz clara y sin ruido de fondo.")

print("Conversion OK - revisa kurisu_output.wav")
