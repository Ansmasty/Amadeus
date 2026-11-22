"""
agent/tools/system_apps.py

Herramientas para lanzar aplicaciones nativas y paneles de configuración del sistema operativo.
Utiliza `subprocess` para abrir aplicaciones locales sin necesidad de rutas completas.
"""
import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple

from langchain_core.tools import tool

# ─── Mapa de aplicaciones conocidas ──────────────────────────────────────────
# nombre_clave → (ejecutable_windows, ejecutable_unix)

_MAPA_APLICACIONES: Dict[str, Tuple[str, str]] = {
    # Sistema
    "calculadora": ("calc.exe", "gnome-calculator"),
    "calculator": ("calc.exe", "gnome-calculator"),
    "paint": ("mspaint.exe", "pinta"),
    "bloc de notas": ("notepad.exe", "gedit"),
    "notepad": ("notepad.exe", "gedit"),
    "explorador": ("explorer.exe", "nautilus"),
    "explorer": ("explorer.exe", "nautilus"),
    "task manager": ("taskmgr.exe", "gnome-system-monitor"),
    "administrador de tareas": ("taskmgr.exe", "gnome-system-monitor"),
    "panel de control": ("control.exe", "gnome-control-center"),
    "control panel": ("control.exe", "gnome-control-center"),
    "configuración": ("ms-settings:", "gnome-control-center"),
    "settings": ("ms-settings:", "gnome-control-center"),
    "cmd": ("cmd.exe", "bash"),
    "terminal": ("cmd.exe", "bash"),
    "powershell": ("powershell.exe", "bash"),
    "registro": ("regedit.exe", ""),
    "regedit": ("regedit.exe", ""),
    # Multimedia
    "reproductor": ("wmplayer.exe", "vlc"),
    "media player": ("wmplayer.exe", "vlc"),
    "vlc": ("vlc.exe", "vlc"),
    "grabadora de voz": ("SoundRecorder.exe", "arecord"),
    "camara": ("microsoft.windows.camera:", "cheese"),
    "cámara": ("microsoft.windows.camera:", "cheese"),
    # Ofimática
    "word": ("winword.exe", "libreoffice --writer"),
    "excel": ("excel.exe", "libreoffice --calc"),
    "powerpoint": ("powerpnt.exe", "libreoffice --impress"),
    "outlook": ("outlook.exe", "thunderbird"),
    "onenote": ("onenote.exe", ""),
    # Utilidades
    "snipping tool": ("SnippingTool.exe", "gnome-screenshot"),
    "recortes": ("SnippingTool.exe", "gnome-screenshot"),
    "reloj": ("ms-clock:", "gnome-clocks"),
    "clock": ("ms-clock:", "gnome-clocks"),
    "mapas": ("bingmaps:", ""),
    "maps": ("bingmaps:", ""),
    "tiempo": ("bingweather:", ""),
    "weather": ("bingweather:", ""),
    "tienda": ("ms-windows-store:", ""),
    "store": ("ms-windows-store:", ""),
    # Desarrollo
    "visual studio code": ("code.exe", "code"),
    "vscode": ("code.exe", "code"),
    "vs code": ("code.exe", "code"),
    # Navegadores
    "chrome": ("chrome.exe", "google-chrome"),
    "firefox": ("firefox.exe", "firefox"),
    "edge": ("msedge.exe", "microsoft-edge"),
    "opera": ("opera.exe", "opera"),
    # Comunicación
    "discord": ("discord.exe", "discord"),
    "teams": ("ms-teams.exe", "teams"),
    "slack": ("slack.exe", "slack"),
    "telegram": ("telegram.exe", "telegram-desktop"),
    "whatsapp": ("whatsapp.exe", ""),
    "zoom": ("zoom.exe", "zoom"),
    # Entretenimiento
    "spotify": ("spotify.exe", "spotify"),
    "steam": ("steam.exe", "steam"),
}

# URIs de ms-settings para abrir paneles específicos de Windows
_AJUSTES_WINDOWS: Dict[str, str] = {
    "wifi": "ms-settings:network-wifi",
    "bluetooth": "ms-settings:bluetooth",
    "pantalla": "ms-settings:display",
    "display": "ms-settings:display",
    "sonido": "ms-settings:sound",
    "sound": "ms-settings:sound",
    "privacidad": "ms-settings:privacy",
    "actualizaciones": "ms-settings:windowsupdate",
    "updates": "ms-settings:windowsupdate",
    "aplicaciones": "ms-settings:appsfeatures",
    "apps": "ms-settings:appsfeatures",
}


def _es_windows() -> bool:
    """Indica si el sistema operativo actual es Windows."""
    return os.name == "nt"


def _lanzar_ejecutable(ejecutable: str) -> Tuple[bool, str]:
    """
    Lanza un ejecutable del sistema o una URI de protocolo (ms-settings:, etc.).

    Args:
        ejecutable (str): Nombre del binario o URI de protocolo a ejecutar.

    Returns:
        Tuple[bool, str]: Tupla con (exito, mensaje_descriptivo).
    """
    if not ejecutable:
        return False, "No hay ejecutable definido para este sistema operativo."

    if ":" in ejecutable and not ejecutable.endswith(".exe"):
        try:
            os.startfile(ejecutable)  # type: ignore[attr-defined]
            return True, f"Abierto: {ejecutable}"
        except Exception as exc:
            return False, f"Error abriendo URI '{ejecutable}': {exc}"

    ruta_encontrada: Optional[str] = shutil.which(ejecutable)
    if ruta_encontrada:
        try:
            subprocess.Popen(
                [ruta_encontrada],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=(
                    subprocess.DETACHED_PROCESS if _es_windows() else 0
                ),
            )
            return True, f"Lanzado: {ejecutable}"
        except Exception as exc:
            return False, f"Error lanzando '{ejecutable}': {exc}"

    if _es_windows():
        try:
            subprocess.Popen(
                ejecutable,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"Lanzado (shell): {ejecutable}"
        except Exception as exc:
            return False, f"Error lanzando '{ejecutable}' vía shell: {exc}"

    return False, f"No se encontró '{ejecutable}' en el PATH del sistema."


# ─── Herramientas Expuestas al Agente ─────────────────────────────────────────


@tool
def open_application(app_name: str) -> str:
    """
    Abre una aplicación nativa instalada en la computadora del usuario.

    Funciona con aplicaciones integradas de Windows (Calculadora, Paint, Bloc de notas,
    Explorador, CMD, etc.) y programas instalados (Chrome, Discord, Spotify, VS Code, etc.).
    Prefiere siempre esta herramienta sobre `open_url` cuando el usuario pida abrir
    una aplicación de escritorio.

    Args:
        app_name (str): Nombre de la aplicación que se desea abrir
            (por ejemplo, "calculadora", "paint", "spotify", "notepad").

    Returns:
        str: Mensaje indicando si la aplicación fue abierta con éxito o el error ocurrido.
    """
    clave: str = app_name.strip().lower()

    entrada: Optional[Tuple[str, str]] = _MAPA_APLICACIONES.get(clave)
    if entrada:
        ejecutable: str = entrada[0] if _es_windows() else entrada[1]
        exito, mensaje = _lanzar_ejecutable(ejecutable)
        if exito:
            return f"✓ Aplicación abierta: {app_name}"
        exito_alternativo, mensaje_alternativo = _lanzar_ejecutable(app_name)
        if exito_alternativo:
            return f"✓ Aplicación abierta: {app_name}"
        return (
            f"✗ No se pudo abrir '{app_name}'.\n"
            f"  Intento 1: {mensaje}\n"
            f"  Intento 2: {mensaje_alternativo}"
        )

    exito, mensaje = _lanzar_ejecutable(app_name)
    if exito:
        return f"✓ Aplicación abierta: {app_name}"

    if _es_windows() and not app_name.endswith(".exe"):
        exito_exe, _ = _lanzar_ejecutable(app_name + ".exe")
        if exito_exe:
            return f"✓ Aplicación abierta: {app_name}.exe"

    return (
        f"✗ No se encontró la aplicación '{app_name}'.\n"
        f"  Asegúrate de que está instalada y accesible desde el PATH.\n"
        f"  Error: {mensaje}"
    )


@tool
def open_system_settings(setting: str) -> str:
    """
    Abre directamente un panel específico de la Configuración del sistema operativo.

    Úsala cuando el usuario pida abrir paneles como WiFi, Bluetooth, Pantalla,
    Sonido, Privacidad, Actualizaciones (Windows Update) o Aplicaciones.

    Args:
        setting (str): Nombre del panel de configuración a abrir
            (por ejemplo, "wifi", "bluetooth", "pantalla", "sonido", "actualizaciones").

    Returns:
        str: Confirmación de apertura del panel de configuración o mensaje de error.
    """
    clave: str = setting.strip().lower()

    uri_ajuste: Optional[str] = _AJUSTES_WINDOWS.get(clave)
    if not uri_ajuste:
        uri_ajuste = f"ms-settings:{clave}"

    if not _es_windows():
        exito, mensaje = _lanzar_ejecutable("gnome-control-center")
        return "✓ Configuración abierta" if exito else f"✗ {mensaje}"

    exito, mensaje = _lanzar_ejecutable(uri_ajuste)
    return (
        f"✓ Configuración '{setting}' abierta"
        if exito
        else f"✗ No se pudo abrir '{setting}': {mensaje}"
    )


@tool
def open_file_explorer(path: str = "") -> str:
    """
    Abre el explorador de archivos del sistema operativo en una carpeta específica.

    Si no se proporciona ninguna ruta (cadena vacía), abre el explorador de archivos
    en la carpeta predeterminada del usuario.

    Args:
        path (str): Ruta opcional del directorio que se desea abrir en el explorador
            (por ejemplo, "C:/Users/Usuario/Desktop"). Por defecto es "".

    Returns:
        str: Confirmación de apertura del explorador de archivos o mensaje de error.
    """
    if _es_windows():
        ruta_objetivo: str = path.strip() if path.strip() else ""
        try:
            if ruta_objetivo:
                subprocess.Popen(
                    ["explorer.exe", ruta_objetivo],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return f"✓ Explorador abierto en: {ruta_objetivo}"
            subprocess.Popen(
                ["explorer.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "✓ Explorador de archivos abierto"
        except Exception as exc:
            return f"✗ Error abriendo el explorador: {exc}"

    exito, mensaje = _lanzar_ejecutable(f"nautilus {path}".strip())
    return "✓ Explorador abierto" if exito else f"✗ {mensaje}"
