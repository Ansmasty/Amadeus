"""
agent/tools/interpreter.py

Herramienta de ejecución autónoma de código Python (Computer Use / Code Interpreter)
para AMADEUS. Permite procesar archivos de formatos arbitrarios, automatizar
navegadores web con Playwright e interactuar físicamente con el escritorio mediante
PyAutoGUI y Pyperclip, bajo supervisión obligatoria Human-in-the-Loop (HITL).
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Final, List, Optional

from langchain_core.tools import tool

from agent.hitl import CONFIRMATION_SENTINEL, is_confirmed, register_pending

TIEMPO_LIMITE_SEGUNDOS: Final[int] = 120
MAX_CARACTERES_SALIDA: Final[int] = 8000
MAX_CARACTERES_VISTA_PREVIA: Final[int] = 200
MAX_CARACTERES_PARAMETRO_CODIGO: Final[int] = 600


def _normalizar_codigo(codigo: str) -> str:
    """
    Limpia espacios externos y elimina delimitadores Markdown si el LLM los incluyó.

    Args:
        codigo (str): Código fuente en Python generado por el agente.

    Returns:
        str: Código fuente limpio listo para escribirse en un archivo `.py`.
    """
    texto_limpio: str = codigo.strip()
    if texto_limpio.startswith("```") and texto_limpio.endswith("```"):
        lineas: List[str] = texto_limpio.splitlines()
        if len(lineas) >= 2:
            texto_limpio = "\n".join(lineas[1:-1]).strip()
    return texto_limpio


def _truncar_salida(texto: str) -> str:
    """
    Trunca salidas extensas para proteger la ventana de contexto del modelo LLM.

    Args:
        texto (str): Salida capturada desde `stdout` o `stderr`.

    Returns:
        str: Texto original o truncado con indicador informativo.
    """
    if len(texto) > MAX_CARACTERES_SALIDA:
        return (
            f"{texto[:MAX_CARACTERES_SALIDA]}\n\n"
            "[...salida truncada por límite de longitud...]"
        )
    return texto


def _generar_clave_confirmacion(codigo_limpio: str) -> str:
    """
    Genera un identificador determinista único para registrar la acción en HITL.

    Args:
        codigo_limpio (str): Código Python normalizado.

    Returns:
        str: Clave hash SHA-256 prefijada para el registro de confirmación.
    """
    resumen_hash: str = hashlib.sha256(
        codigo_limpio.encode("utf-8")
    ).hexdigest()[:16]
    return f"run_python:{resumen_hash}"


@tool
def ejecutar_codigo_python(codigo: str) -> str:
    """
    Ejecuta un script de Python 3 en un archivo temporal aislado y retorna su salida.

    Úsala como herramienta principal de autonomía (Computer Use y Code Interpreter):
      - Procesar o modificar archivos desconocidos (.pdf, .docx, .zip, imágenes, etc.).
      - Instalar librerías faltantes mediante `subprocess` y `sys.executable -m pip`.
      - Automatizar sitios web interactivos con `playwright` (modo síncrono).
      - Automatizar aplicaciones de escritorio, ratón, teclado y portapapeles con
        `pyautogui` y `pyperclip`.

    Asegúrate de usar `print(...)` en el script para emitir cualquier resultado que
    necesites leer desde `stdout`. Requiere confirmación explícita del usuario (HITL)
    antes de cada ejecución.

    Args:
        codigo (str): Código fuente completo y válido en Python 3 a ejecutar.

    Returns:
        str: Contenido de `stdout` si la ejecución es exitosa, detalle de `stderr` si
            ocurre un fallo (para permitir auto-corrección), o el sentinel HITL si
            requiere confirmación del usuario.
    """
    codigo_limpio: str = _normalizar_codigo(codigo)
    if not codigo_limpio:
        return "Error: el código Python proporcionado está vacío."

    clave_accion: str = _generar_clave_confirmacion(codigo_limpio)
    if not is_confirmed(clave_accion):
        total_lineas: int = len(codigo_limpio.splitlines())
        vista_previa: str = (
            f"{codigo_limpio[:MAX_CARACTERES_VISTA_PREVIA]}..."
            if len(codigo_limpio) > MAX_CARACTERES_VISTA_PREVIA
            else codigo_limpio
        )
        register_pending(
            key=clave_accion,
            description=(
                f"¿Ejecutar script de Python ({total_lineas} línea(s))?\n"
                f"Vista previa:\n{vista_previa}"
            ),
            action_type="run_python",
            params={
                "lineas": total_lineas,
                "codigo_preview": codigo_limpio[:MAX_CARACTERES_PARAMETRO_CODIGO],
            },
        )
        return CONFIRMATION_SENTINEL

    ruta_temporal: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="amadeus_exec_",
            encoding="utf-8",
            delete=False,
        ) as archivo_temporal:
            archivo_temporal.write(codigo_limpio)
            ruta_temporal = Path(archivo_temporal.name)

        entorno_ejecucion: Dict[str, str] = os.environ.copy()
        entorno_ejecucion["PYTHONIOENCODING"] = "utf-8"
        entorno_ejecucion["PYTHONUTF8"] = "1"

        resultado: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, str(ruta_temporal)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIEMPO_LIMITE_SEGUNDOS,
            env=entorno_ejecucion,
        )

        if resultado.returncode == 0:
            salida_estandar: str = (resultado.stdout or "").strip()
            if salida_estandar:
                return _truncar_salida(salida_estandar)
            return "Ejecución completada exitosamente (sin salida en stdout)."

        salida_error: str = (resultado.stderr or "").strip()
        if salida_error:
            return _truncar_salida(salida_error)

        salida_respaldo: str = (resultado.stdout or "").strip()
        if salida_respaldo:
            return _truncar_salida(
                f"El script finalizó con código {resultado.returncode}:\n{salida_respaldo}"
            )
        return (
            f"Error: el script finalizó con código de salida "
            f"{resultado.returncode} sin mensajes en stderr."
        )

    except subprocess.TimeoutExpired as exc:
        error_parcial: str = ""
        if exc.stderr:
            texto_stderr: str = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            ).strip()
            if texto_stderr:
                error_parcial = f"\nDetalle parcial (stderr):\n{texto_stderr}"
        return _truncar_salida(
            f"Error: la ejecución del script excedió el tiempo límite de "
            f"{TIEMPO_LIMITE_SEGUNDOS} segundos.{error_parcial}"
        )
    except Exception as exc:
        return f"Error inesperado al ejecutar el script de Python: {exc}"
    finally:
        if ruta_temporal is not None and ruta_temporal.exists():
            try:
                ruta_temporal.unlink()
            except OSError:
                pass
