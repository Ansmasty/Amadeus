"""
agent/tools/data_analysis.py

Herramientas de análisis de datos para AMADEUS.
Lee archivos Excel y CSV utilizando pandas y devuelve resúmenes estructurados.
Todas las rutas se validan previamente a través de la capa de seguridad.
"""
import io
from pathlib import Path
from typing import Any, Set, Union

from langchain_core.tools import tool

from security.validators import PathSecurityError, validate_path

EXTENSIONES_EXCEL_SOPORTADAS: Set[str] = {".xlsx", ".xls", ".xlsm", ".xlsb"}


@tool
def read_excel(path: str, sheet_name: str = "0") -> str:
    """
    Lee un archivo de Excel (.xlsx, .xls, .xlsm, .xlsb) y devuelve un resumen estructurado.

    El resumen incluye dimensiones (filas y columnas), nombres de columnas con sus
    tipos de datos, conteo de valores nulos, las primeras 5 filas y estadísticas
    descriptivas básicas.

    Args:
        path (str): Ruta completa del archivo Excel en el sistema de archivos
            (por ejemplo, "C:/Users/Usuario/Documents/datos.xlsx").
        sheet_name (str): Nombre de la hoja o índice numérico basado en cero
            como cadena de texto. Por defecto es "0" (la primera hoja).

    Returns:
        str: Resumen analítico del archivo Excel o descripción del error ocurrido.
    """
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas is not installed. Run: pip install pandas openpyxl"

    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not ruta_validada.exists():
        return f"Error: File does not exist: {ruta_validada}"
    if not ruta_validada.is_file():
        return f"Error: '{ruta_validada}' is a directory, not a file."
    if ruta_validada.suffix.lower() not in EXTENSIONES_EXCEL_SOPORTADAS:
        return (
            f"Error: '{ruta_validada.name}' does not appear to be an Excel file. "
            f"Supported extensions: .xlsx, .xls, .xlsm, .xlsb"
        )

    try:
        hoja_objetivo: Union[str, int] = (
            int(sheet_name)
            if sheet_name.strip().lstrip("-").isdigit()
            else sheet_name
        )
        dataframe = pd.read_excel(
            ruta_validada,
            sheet_name=hoja_objetivo,
            engine="openpyxl",
        )
    except Exception as exc:
        return f"Error reading Excel file: {exc}"

    return _resumir_dataframe(
        dataframe,
        str(ruta_validada),
        informacion_extra=f"Sheet: {sheet_name}",
    )


@tool
def read_csv(path: str, separator: str = ",") -> str:
    """
    Lee un archivo CSV y devuelve un resumen analítico estructurado de su contenido.

    El resumen incluye dimensiones del conjunto de datos, nombres de columnas,
    tipos de datos, valores nulos, las primeras 5 filas y estadísticas descriptivas.

    Args:
        path (str): Ruta completa del archivo CSV a analizar
            (por ejemplo, "C:/Users/Usuario/Documents/ventas.csv").
        separator (str): Carácter delimitador de columnas del archivo CSV.
            Por defecto es "," (coma).

    Returns:
        str: Resumen estructurado del archivo CSV o mensaje de error.
    """
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas is not installed. Run: pip install pandas"

    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not ruta_validada.exists():
        return f"Error: File does not exist: {ruta_validada}"
    if not ruta_validada.is_file():
        return f"Error: '{ruta_validada}' is a directory, not a file."

    try:
        dataframe = pd.read_csv(
            ruta_validada,
            sep=separator,
            encoding="utf-8",
            on_bad_lines="warn",
        )
    except UnicodeDecodeError:
        try:
            dataframe = pd.read_csv(
                ruta_validada,
                sep=separator,
                encoding="latin-1",
                on_bad_lines="warn",
            )
        except Exception as exc:
            return f"Error reading CSV file: {exc}"
    except Exception as exc:
        return f"Error reading CSV file: {exc}"

    return _resumir_dataframe(
        dataframe,
        str(ruta_validada),
        informacion_extra=f"Separator: '{separator}'",
    )


# ─── Función auxiliar interna ─────────────────────────────────────────────────


def _resumir_dataframe(
    dataframe: Any,
    ruta_archivo: str,
    informacion_extra: str = "",
) -> str:
    """
    Formatea un DataFrame de pandas en una cadena de resumen legible.

    Args:
        dataframe (Any): Objeto pandas.DataFrame con los datos cargados.
        ruta_archivo (str): Ruta del archivo analizado.
        informacion_extra (str): Metadatos adicionales (hoja o separador).

    Returns:
        str: Texto estructurado con el análisis del DataFrame.
    """
    buffer_texto = io.StringIO()

    buffer_texto.write("=== File Analysis ===\n")
    buffer_texto.write(f"Path:  {ruta_archivo}\n")
    if informacion_extra:
        buffer_texto.write(f"Info:  {informacion_extra}\n")
    buffer_texto.write(
        f"Shape: {dataframe.shape[0]:,} rows x {dataframe.shape[1]} columns\n\n"
    )

    buffer_texto.write("--- Column Overview ---\n")
    for columna in dataframe.columns:
        no_nulos: int = int(dataframe[columna].notna().sum())
        conteo_nulos: int = int(dataframe[columna].isna().sum())
        tipo_dato: Any = dataframe[columna].dtype
        info_nulos: str = f"  [{conteo_nulos} nulls]" if conteo_nulos > 0 else ""
        buffer_texto.write(
            f"  {columna!r} ({tipo_dato}): {no_nulos:,} values{info_nulos}\n"
        )

    buffer_texto.write("\n--- First 5 Rows ---\n")
    buffer_texto.write(dataframe.head(5).to_string())
    buffer_texto.write("\n")

    buffer_texto.write("\n--- Descriptive Statistics ---\n")
    try:
        buffer_texto.write(dataframe.describe(include="all").to_string())
    except Exception:
        buffer_texto.write("(statistics unavailable)")
    buffer_texto.write("\n")

    return buffer_texto.getvalue()
