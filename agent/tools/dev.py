"""
agent/tools/dev.py

Herramientas de asistencia en programación para AMADEUS:
  - git_status / git_diff / git_log: inspección de repositorios Git (solo lectura).
  - search_in_project: búsqueda de texto/código dentro de un proyecto.
  - run_terminal_command: ejecución de comandos de terminal, con validación
    de ruta y confirmación HITL (es una operación potencialmente destructiva).

Toda ruta pasa por security.validators.validate_path antes de tocar disco,
igual que en agent/tools/filesystem.py.
"""
import re
import subprocess

from langchain_core.tools import tool

from agent.hitl import CONFIRMATION_SENTINEL, is_confirmed, register_pending
from security.validators import validate_path, PathSecurityError

_MAX_OUTPUT_CHARS = 6000
_COMMAND_TIMEOUT_SECONDS = 30

# Patrones de comandos bloqueados incluso tras confirmación del usuario.
# No es una lista exhaustiva; es una barrera adicional, no la única defensa.
_BLOCKED_COMMAND_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bdel\s+/s\s+/q\s+[a-zA-Z]:\\?\s*$",
    r"\bformat\s+[a-zA-Z]:",
    r"\bshutdown\b",
    r"\bmkfs\b",
    r"\bdiskpart\b",
    r":\(\)\s*\{.*\}\s*;",  # fork bomb
    r"\breg\s+delete\b",
    r"\bnet\s+user\b.*\/add",
]


def _run(cmd: list[str], cwd) -> tuple[bool, str]:
    """Ejecuta un comando y devuelve (éxito, salida_combinada)."""
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return False, f"Error: no se encontró el ejecutable '{cmd[0]}' en el PATH."
    except subprocess.TimeoutExpired:
        return False, f"Error: el comando excedió {_COMMAND_TIMEOUT_SECONDS}s y fue cancelado."
    except Exception as exc:
        return False, f"Error al ejecutar el comando: {exc}"

    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip() or "(sin salida)"
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n\n[...salida truncada...]"
    return result.returncode == 0, output


def _validated_repo_path(path: str):
    validated = validate_path(path)
    if not validated.exists() or not validated.is_dir():
        raise PathSecurityError(f"'{validated}' no es un directorio válido.")
    return validated


# ─── Git (solo lectura) ────────────────────────────────────────────────────────


@tool
def git_status(path: str) -> str:
    """
    Muestra el estado de un repositorio git (archivos modificados, nuevos,
    staged) en la ruta indicada. Úsala cuando el usuario pregunte qué ha
    cambiado en su proyecto o quiera revisar el estado antes de commitear.
    Example: git_status("C:/Users/me/Proyectos/mi-app")
    """
    try:
        repo = _validated_repo_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    ok, output = _run(["git", "status", "--short", "--branch"], repo)
    if not ok:
        return f"Error al obtener el estado de git:\n{output}"
    return f"Estado de git en '{repo}':\n\n{output}"


@tool
def git_diff(path: str, file: str = "", staged: bool = False) -> str:
    """
    Muestra los cambios (diff) de un repositorio git, opcionalmente de un
    archivo específico o solo de los cambios en staging.
    Úsala cuando el usuario pida ver qué se modificó exactamente en el código.
    Example: git_diff("C:/Users/me/Proyectos/mi-app", file="main.py")
    """
    try:
        repo = _validated_repo_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    if file:
        cmd.append("--")
        cmd.append(file)

    ok, output = _run(cmd, repo)
    if not ok:
        return f"Error al obtener el diff de git:\n{output}"
    if output == "(sin salida)":
        return "No hay cambios que mostrar."
    return f"Diff de git en '{repo}':\n\n{output}"


@tool
def git_log(path: str, count: int = 10) -> str:
    """
    Muestra el historial de commits recientes de un repositorio git
    (hash corto, autor, fecha y mensaje).
    Example: git_log("C:/Users/me/Proyectos/mi-app", count=5)
    """
    try:
        repo = _validated_repo_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    count = max(1, min(count, 50))
    cmd = ["git", "log", f"-{count}", "--pretty=format:%h | %ad | %an | %s", "--date=short"]

    ok, output = _run(cmd, repo)
    if not ok:
        return f"Error al obtener el historial de git:\n{output}"
    return f"Últimos {count} commits en '{repo}':\n\n{output}"


# ─── Búsqueda en proyecto (solo lectura) ───────────────────────────────────────

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".idea", ".vscode", "target", ".next", ".mypy_cache",
}
_SEARCHABLE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css", ".scss", ".json",
    ".yaml", ".yml", ".md", ".txt", ".sql", ".sh", ".bat", ".xml", ".toml",
}
_MAX_SEARCH_RESULTS = 40
_MAX_FILE_SIZE_FOR_SEARCH = 2 * 1024 * 1024  # 2 MB


@tool
def search_in_project(path: str, query: str, file_extension: str = "") -> str:
    """
    Busca un texto o patrón dentro de todos los archivos de código de un
    proyecto (como un 'grep' recursivo). Ignora carpetas como .git,
    node_modules, __pycache__, venv, dist, build.
    Úsala cuando el usuario pida encontrar dónde se usa una función, variable,
    texto de error, o cualquier fragmento de código dentro de su proyecto.
    Example: search_in_project("C:/Proyectos/mi-app", "def calculate_total")
    """
    try:
        base = _validated_repo_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not query.strip():
        return "Error: la búsqueda no puede estar vacía."

    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    except re.error as exc:
        return f"Error en el patrón de búsqueda: {exc}"

    ext_filter = file_extension.strip().lower()
    if ext_filter and not ext_filter.startswith("."):
        ext_filter = "." + ext_filter

    matches: list[str] = []

    for root, dirs, files in _walk_skip_hidden(base):
        for fname in files:
            fpath = root / fname
            ext = fpath.suffix.lower()
            if ext_filter:
                if ext != ext_filter:
                    continue
            elif ext not in _SEARCHABLE_EXTENSIONS:
                continue

            try:
                if fpath.stat().st_size > _MAX_FILE_SIZE_FOR_SEARCH:
                    continue
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for i, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    rel = fpath.relative_to(base)
                    matches.append(f"{rel}:{i}: {line.strip()[:180]}")
                    if len(matches) >= _MAX_SEARCH_RESULTS:
                        break
            if len(matches) >= _MAX_SEARCH_RESULTS:
                break
        if len(matches) >= _MAX_SEARCH_RESULTS:
            break

    if not matches:
        return f"No se encontraron coincidencias para '{query}' en '{base}'."

    header = f"Coincidencias para '{query}' en '{base}'"
    if len(matches) >= _MAX_SEARCH_RESULTS:
        header += f" (mostrando las primeras {_MAX_SEARCH_RESULTS})"
    return header + ":\n\n" + "\n".join(matches)


def _walk_skip_hidden(base):
    """os.walk envuelto en Path, ignorando carpetas irrelevantes."""
    import os as _os
    from pathlib import Path
    for root, dirs, files in _os.walk(base):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        yield Path(root), dirs, files


# ─── Terminal (destructivo — requiere confirmación HITL) ──────────────────────


@tool
def run_terminal_command(command: str, path: str) -> str:
    """
    Ejecuta un comando de terminal dentro de un directorio permitido
    (por ejemplo, correr tests, instalar dependencias, formatear código).
    Esta es una operación potencialmente destructiva: SIEMPRE requiere
    confirmación explícita del usuario antes de ejecutarse.
    No la uses para operaciones que ya tienen su propia herramienta
    (borrar/mover archivos, git) — úsala solo para comandos de build/test/
    dependencias u otros que el usuario pida explícitamente.
    Example: run_terminal_command("npm install", "C:/Proyectos/mi-app")
    """
    try:
        cwd = _validated_repo_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not command.strip():
        return "Error: el comando no puede estar vacío."

    lowered = command.lower()
    for blocked_pattern in _BLOCKED_COMMAND_PATTERNS:
        if re.search(blocked_pattern, lowered):
            return (
                "Error: este comando está bloqueado por seguridad y no se "
                "ejecutará bajo ninguna circunstancia."
            )

    action_key = f"terminal:{cwd}:{command.strip()}"

    if not is_confirmed(action_key):
        register_pending(
            key=action_key,
            description=f"Ejecutar el comando `{command.strip()}` en '{cwd}'?",
            action_type="run_terminal_command",
            params={"command": command.strip(), "cwd": str(cwd)},
        )
        return CONFIRMATION_SENTINEL

    ok, output = _run(["cmd", "/c", command] if _is_windows() else ["/bin/sh", "-c", command], cwd)
    status = "✓ Comando ejecutado" if ok else "✗ El comando terminó con error"
    return f"{status} en '{cwd}':\n\n{output}"


def _is_windows() -> bool:
    import platform
    return platform.system().lower() == "windows"
