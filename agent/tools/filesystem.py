"""
agent/tools/filesystem.py

Herramientas de sistema de archivos para AMADEUS.
Todas las herramientas validan las rutas mediante `security.validators` antes de
realizar operaciones de entrada/salida. Las operaciones destructivas (eliminar,
mover, sobrescribir, editar) utilizan el registro de confirmación HITL y devuelven
`CONFIRMATION_SENTINEL` hasta que el usuario apruebe la operación.
"""
import shutil
from pathlib import Path
from typing import List

from langchain_core.tools import tool

from agent.hitl import (
    CONFIRMATION_SENTINEL,
    is_confirmed,
    register_pending,
)
from security.validators import (
    PathSecurityError,
    get_max_file_size_bytes,
    validate_path,
)


# ─── Herramientas de Solo Lectura ─────────────────────────────────────────────


@tool
def list_directory(path: str) -> str:
    """
    Lista el contenido de un directorio mostrando nombres, tipos (archivo o carpeta) y tamaños.

    Úsala para explorar qué archivos y subcarpetas existen dentro de una ruta permitida.

    Args:
        path (str): Ruta del directorio que se desea explorar
            (por ejemplo, "C:/Users/Usuario/Documents").

    Returns:
        str: Listado detallado de los elementos del directorio o mensaje de error.
    """
    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not ruta_validada.exists():
        return f"Error: Path does not exist: {ruta_validada}"
    if not ruta_validada.is_dir():
        return (
            f"Error: '{ruta_validada}' is a file, not a directory. "
            "Use read_file to read it."
        )

    try:
        entradas: List[Path] = sorted(
            ruta_validada.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        return f"Error: Permission denied reading directory: {ruta_validada}"

    if not entradas:
        return f"Directory is empty: {ruta_validada}"

    lineas: List[str] = [
        f"Contents of '{ruta_validada}' ({len(entradas)} items):"
    ]
    for elemento in entradas:
        tipo_elemento: str = "FILE" if elemento.is_file() else "DIR "
        texto_tamano: str = ""
        if elemento.is_file():
            try:
                texto_tamano = f"  ({elemento.stat().st_size:,} bytes)"
            except OSError:
                texto_tamano = "  (size unknown)"
        lineas.append(f"  [{tipo_elemento}] {elemento.name}{texto_tamano}")

    return "\n".join(lineas)


@tool
def read_file(path: str) -> str:
    """
    Lee y devuelve el contenido de texto de un archivo.

    Funciona con archivos de texto plano (como .txt, .py, .json, .md, .csv, .log).
    Para hojas de cálculo Excel, utiliza `read_excel`. Respeta el límite máximo
    de tamaño configurado en el entorno y trunca salidas mayores a 500 líneas.

    Args:
        path (str): Ruta completa del archivo de texto a leer
            (por ejemplo, "C:/Users/Usuario/Desktop/notas.txt").

    Returns:
        str: Contenido del archivo con encabezado informativo o mensaje de error.
    """
    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not ruta_validada.exists():
        return f"Error: File does not exist: {ruta_validada}"
    if not ruta_validada.is_file():
        return (
            f"Error: '{ruta_validada}' is a directory. "
            "Use list_directory to explore it."
        )

    maximo_bytes: int = get_max_file_size_bytes()
    try:
        tamano_bytes: int = ruta_validada.stat().st_size
    except OSError as exc:
        return f"Error accessing file: {exc}"

    if tamano_bytes > maximo_bytes:
        return (
            f"Error: File is too large to read directly ({tamano_bytes:,} bytes). "
            f"Maximum allowed: {maximo_bytes:,} bytes ({maximo_bytes // (1024 * 1024)} MB). "
            "Consider using a data analysis tool if this is a CSV/Excel file."
        )

    try:
        contenido: str = ruta_validada.read_text(
            encoding="utf-8",
            errors="replace",
        )
        lineas: List[str] = contenido.splitlines()
        total_lineas: int = len(lineas)
        aviso_truncado: str = ""
        if total_lineas > 500:
            contenido = "\n".join(lineas[:500])
            aviso_truncado = (
                f"\n\n[Output truncated: showing first 500 of {total_lineas} lines]"
            )
        return (
            f"=== {ruta_validada.name} ({tamano_bytes:,} bytes, {total_lineas} lines) ===\n"
            f"{contenido}{aviso_truncado}"
        )
    except Exception as exc:
        return f"Error reading file: {exc}"


# ─── Herramientas de Creación y Modificación ─────────────────────────────────


@tool
def create_directory(path: str) -> str:
    """
    Crea un nuevo directorio (incluyendo carpetas padre faltantes) en la ruta indicada.

    Es seguro llamarla aunque el directorio ya exista. No requiere confirmación HITL.

    Args:
        path (str): Ruta completa del directorio a crear
            (por ejemplo, "C:/Users/Usuario/Documents/nuevo_proyecto").

    Returns:
        str: Confirmación de creación del directorio o mensaje de error.
    """
    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    try:
        ruta_validada.mkdir(parents=True, exist_ok=True)
        return f"Directory created (or already exists): {ruta_validada}"
    except PermissionError:
        return f"Error: Permission denied creating directory: {ruta_validada}"
    except Exception as exc:
        return f"Error creating directory: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Crea un archivo nuevo con el contenido dado o sobrescribe uno existente.

    Úsala para crear archivos desde cero o cuando el usuario pida explícitamente
    reemplazar todo el contenido de un archivo. Para cambios puntuales en archivos
    existentes, prefiere siempre `edit_file`. Sobrescribir un archivo existente
    es una operación destructiva y requiere confirmación del usuario.

    Args:
        path (str): Ruta completa del archivo donde se escribirá el contenido.
        content (str): Contenido completo de texto que se guardará en el archivo.

    Returns:
        str: Confirmación de escritura, sentinel de confirmación HITL o mensaje de error.
    """
    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if ruta_validada.is_dir():
        return f"Error: '{ruta_validada}' is a directory, not a file."

    archivo_existe: bool = ruta_validada.exists()

    if archivo_existe:
        clave_accion: str = f"write:{ruta_validada}:{hash(content)}"
        if not is_confirmed(clave_accion):
            try:
                vista_previa: str = ruta_validada.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                vista_previa = "(no se pudo leer el contenido actual)"
            register_pending(
                key=clave_accion,
                description=(
                    f"Sobrescribir el archivo '{ruta_validada.name}' "
                    f"({len(vista_previa.splitlines())} líneas actuales → "
                    f"{len(content.splitlines())} líneas nuevas)?"
                ),
                action_type="write_file",
                params={
                    "path": str(ruta_validada),
                    "content_preview": content[:500],
                },
            )
            return CONFIRMATION_SENTINEL

    try:
        ruta_validada.parent.mkdir(parents=True, exist_ok=True)
        ruta_validada.write_text(content, encoding="utf-8")
        accion_verbo: str = "sobrescrito" if archivo_existe else "creado"
        return (
            f"Archivo {accion_verbo} correctamente: {ruta_validada} "
            f"({len(content.splitlines())} líneas)"
        )
    except PermissionError:
        return f"Error: Permission denied writing to: {ruta_validada}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool
def edit_file(path: str, find: str, replace: str, count: int = 1) -> str:
    """
    Realiza una edición puntual en un archivo existente reemplazando un fragmento exacto.

    Busca el texto `find` (debe coincidir exactamente, incluyendo espacios e indentación)
    y lo sustituye por `replace`. Lee primero el archivo con `read_file` para asegurarte
    de copiar el fragmento exacto. Requiere confirmación del usuario antes de aplicar cambios.

    Args:
        path (str): Ruta completa del archivo existente que se desea editar.
        find (str): Fragmento de texto exacto que se buscará en el archivo.
        replace (str): Nuevo texto que reemplazará al fragmento buscado.
        count (int): Número máximo de ocurrencias a reemplazar (por defecto 1;
            usa -1 para reemplazar todas las apariciones).

    Returns:
        str: Confirmación de edición, sentinel de confirmación HITL o mensaje de error.
    """
    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not ruta_validada.exists() or not ruta_validada.is_file():
        return f"Error: File does not exist: {ruta_validada}"

    maximo_bytes: int = get_max_file_size_bytes()
    try:
        if ruta_validada.stat().st_size > maximo_bytes:
            return (
                "Error: File too large to edit safely "
                f"({ruta_validada.stat().st_size:,} bytes)."
            )
        contenido_original: str = ruta_validada.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return f"Error reading file: {exc}"

    ocurrencias: int = contenido_original.count(find)
    if ocurrencias == 0:
        return (
            "Error: no se encontró el texto indicado en el archivo. "
            "Vuelve a leer el archivo con read_file y copia el fragmento exacto."
        )

    cantidad_reemplazos: int = (
        ocurrencias if count == -1 else min(count, ocurrencias)
    )
    nuevo_contenido: str = (
        contenido_original.replace(find, replace, cantidad_reemplazos)
        if count != -1
        else contenido_original.replace(find, replace)
    )

    clave_accion: str = f"edit:{ruta_validada}:{hash(find)}:{hash(replace)}"
    if not is_confirmed(clave_accion):
        register_pending(
            key=clave_accion,
            description=(
                f"Editar '{ruta_validada.name}': reemplazar "
                f"{cantidad_reemplazos} ocurrencia(s) de:\n  «{find[:120]}»\n"
                f"por:\n  «{replace[:120]}»"
            ),
            action_type="edit_file",
            params={
                "path": str(ruta_validada),
                "find": find[:300],
                "replace": replace[:300],
            },
        )
        return CONFIRMATION_SENTINEL

    try:
        ruta_validada.write_text(nuevo_contenido, encoding="utf-8")
        return (
            f"Archivo editado correctamente: {ruta_validada} "
            f"({cantidad_reemplazos} reemplazo(s))"
        )
    except PermissionError:
        return f"Error: Permission denied writing to: {ruta_validada}"
    except Exception as exc:
        return f"Error writing file: {exc}"


# ─── Herramientas Destructivas (Requieren Confirmación HITL) ─────────────────


@tool
def delete_file(path: str) -> str:
    """
    Elimina permanentemente un archivo o un directorio (con todo su contenido).

    ADVERTENCIA: Operación destructiva e irreversible. Siempre solicita confirmación
    explícita del usuario mediante el flujo HITL antes de ejecutar el borrado.

    Args:
        path (str): Ruta completa del archivo o directorio que se desea eliminar.

    Returns:
        str: Confirmación de eliminación, sentinel de confirmación HITL o mensaje de error.
    """
    try:
        ruta_validada: Path = validate_path(path)
    except PathSecurityError as exc:
        return f"Security error: {exc}"

    if not ruta_validada.exists():
        return f"Error: Nothing to delete - path does not exist: {ruta_validada}"

    clave_accion: str = f"delete:{ruta_validada}"

    if not is_confirmed(clave_accion):
        tipo_elemento: str = (
            "directory and all its contents"
            if ruta_validada.is_dir()
            else "file"
        )
        register_pending(
            key=clave_accion,
            description=f"Permanently delete {tipo_elemento} '{ruta_validada.name}'?",
            action_type="delete_file",
            params={
                "path": str(ruta_validada),
                "type": "directory" if ruta_validada.is_dir() else "file",
            },
        )
        return CONFIRMATION_SENTINEL

    try:
        if ruta_validada.is_dir():
            shutil.rmtree(ruta_validada)
        else:
            ruta_validada.unlink()
        return f"Successfully deleted: {ruta_validada}"
    except PermissionError:
        return f"Error: Permission denied deleting: {ruta_validada}"
    except Exception as exc:
        return f"Error deleting '{ruta_validada}': {exc}"


@tool
def move_file(source: str, destination: str) -> str:
    """
    Mueve o renombra un archivo o directorio desde una ruta de origen a una de destino.

    ADVERTENCIA: Operación destructiva que requiere confirmación del usuario. Si la
    ruta de destino ya existe, será sobrescrita tras la confirmación.

    Args:
        source (str): Ruta completa actual del archivo o directorio de origen.
        destination (str): Ruta completa nueva donde se moverá el archivo o directorio.

    Returns:
        str: Confirmación del movimiento, sentinel de confirmación HITL o mensaje de error.
    """
    try:
        origen_validado: Path = validate_path(source)
    except PathSecurityError as exc:
        return f"Security error (source): {exc}"

    destino_crudo: Path = Path(destination)
    try:
        padre_destino: Path = validate_path(str(destino_crudo.parent))
    except PathSecurityError as exc:
        return f"Security error (destination): {exc}"

    if not origen_validado.exists():
        return f"Error: Source does not exist: {origen_validado}"

    destino_validado: Path = padre_destino / destino_crudo.name
    clave_accion: str = f"move:{origen_validado}:{destino_validado}"

    if not is_confirmed(clave_accion):
        register_pending(
            key=clave_accion,
            description=f"Move '{origen_validado.name}' to '{destino_validado}'?",
            action_type="move_file",
            params={
                "source": str(origen_validado),
                "destination": str(destino_validado),
            },
        )
        return CONFIRMATION_SENTINEL

    try:
        shutil.move(str(origen_validado), str(destino_validado))
        return f"Moved:\n  From: {origen_validado}\n  To:   {destino_validado}"
    except PermissionError:
        return "Error: Permission denied during move."
    except Exception as exc:
        return f"Error moving file: {exc}"


@tool
def copy_file(source: str, destination: str) -> str:
    """
    Copia un archivo o directorio desde una ruta de origen hacia una ruta de destino.

    Si la ruta de destino no existe, realiza la copia directamente. Si el destino ya
    existe, solicita confirmación del usuario antes de sobrescribirlo.

    Args:
        source (str): Ruta completa del archivo o directorio original a copiar.
        destination (str): Ruta completa donde se creará la copia.

    Returns:
        str: Confirmación de la copia, sentinel de confirmación HITL o mensaje de error.
    """
    try:
        origen_validado: Path = validate_path(source)
    except PathSecurityError as exc:
        return f"Security error (source): {exc}"

    destino_crudo: Path = Path(destination)
    try:
        padre_destino: Path = validate_path(str(destino_crudo.parent))
    except PathSecurityError as exc:
        return f"Security error (destination): {exc}"

    if not origen_validado.exists():
        return f"Error: Source does not exist: {origen_validado}"

    destino_validado: Path = padre_destino / destino_crudo.name

    if destino_validado.exists():
        clave_accion: str = f"copy_overwrite:{origen_validado}:{destino_validado}"
        if not is_confirmed(clave_accion):
            register_pending(
                key=clave_accion,
                description=(
                    f"Overwrite existing '{destino_validado.name}' at "
                    f"'{destino_validado.parent}' with a copy of '{origen_validado.name}'?"
                ),
                action_type="copy_file",
                params={
                    "source": str(origen_validado),
                    "destination": str(destino_validado),
                    "overwrite": "yes",
                },
            )
            return CONFIRMATION_SENTINEL

    try:
        if origen_validado.is_dir():
            shutil.copytree(
                str(origen_validado),
                str(destino_validado),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(str(origen_validado), str(destino_validado))
        return f"Copied:\n  From: {origen_validado}\n  To:   {destino_validado}"
    except PermissionError:
        return "Error: Permission denied during copy."
    except Exception as exc:
        return f"Error copying: {exc}"
