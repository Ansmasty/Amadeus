"""
agent/tools/data_analysis.py

Data analysis tools for AMADEUS.
Reads Excel and CSV files using pandas and returns structured summaries.
All paths are validated through the security layer before any I/O.
"""
import io

from langchain_core.tools import tool

from security.validators import validate_path, PathSecurityError


@tool
def read_excel(path: str, sheet_name: str = "0") -> str:
    """
    Read an Excel file (.xlsx or .xls) and return a structured summary.
    Includes: file shape, column names with data types, null value counts,
    the first 5 rows, and basic descriptive statistics.
    Optionally specify a sheet by name or zero-based index (default: first sheet).
    Example: read_excel("C:/Users/me/data.xlsx", sheet_name="Sheet1")
    """
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas is not installed. Run: pip install pandas openpyxl"

    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not validated.exists():
        return f"Error: File does not exist: {validated}"
    if not validated.is_file():
        return f"Error: '{validated}' is a directory, not a file."
    if validated.suffix.lower() not in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        return (
            f"Error: '{validated.name}' does not appear to be an Excel file. "
            f"Supported extensions: .xlsx, .xls, .xlsm, .xlsb"
        )

    try:
        # Convert numeric string to int for positional sheet access
        sheet: str | int = int(sheet_name) if sheet_name.strip().lstrip("-").isdigit() else sheet_name
        df = pd.read_excel(validated, sheet_name=sheet, engine="openpyxl")
    except Exception as exc:
        return f"Error reading Excel file: {exc}"

    return _summarize_dataframe(df, str(validated), extra=f"Sheet: {sheet_name}")


@tool
def read_csv(path: str, separator: str = ",") -> str:
    """
    Read a CSV file and return a structured summary.
    Includes: file shape, column names with data types, null value counts,
    the first 5 rows, and basic descriptive statistics.
    Optionally specify a separator character (default: comma).
    Example: read_csv("C:/Users/me/data.csv", separator=";")
    """
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas is not installed. Run: pip install pandas"

    try:
        validated = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not validated.exists():
        return f"Error: File does not exist: {validated}"
    if not validated.is_file():
        return f"Error: '{validated}' is a directory, not a file."

    try:
        df = pd.read_csv(validated, sep=separator, encoding="utf-8", on_bad_lines="warn")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(validated, sep=separator, encoding="latin-1", on_bad_lines="warn")
        except Exception as exc:
            return f"Error reading CSV file: {exc}"
    except Exception as exc:
        return f"Error reading CSV file: {exc}"

    return _summarize_dataframe(df, str(validated), extra=f"Separator: '{separator}'")


# ─── Internal helper ──────────────────────────────────────────────────────────


def _summarize_dataframe(df, path: str, extra: str = "") -> str:
    """Format a pandas DataFrame into a readable summary string."""
    buf = io.StringIO()

    buf.write(f"=== File Analysis ===\n")
    buf.write(f"Path:  {path}\n")
    if extra:
        buf.write(f"Info:  {extra}\n")
    buf.write(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n\n")

    buf.write("--- Column Overview ---\n")
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        dtype = df[col].dtype
        null_info = f"  [{null_count} nulls]" if null_count > 0 else ""
        buf.write(f"  {col!r} ({dtype}): {non_null:,} values{null_info}\n")

    buf.write("\n--- First 5 Rows ---\n")
    buf.write(df.head(5).to_string())
    buf.write("\n")

    buf.write("\n--- Descriptive Statistics ---\n")
    try:
        buf.write(df.describe(include="all").to_string())
    except Exception:
        buf.write("(statistics unavailable)")
    buf.write("\n")

    return buf.getvalue()
