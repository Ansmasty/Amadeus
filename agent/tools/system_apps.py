"""
agent/tools/system_apps.py

Herramientas para lanzar aplicaciones nativas del sistema operativo.
Usa subprocess para abrir apps de Windows sin necesidad de rutas completas.
"""
import os
import subprocess
import shutil
from langchain_core.tools import tool


# ─── Mapa de aplicaciones conocidas ──────────────────────────────────────────
# nombre_clave → (ejecutable_windows, ejecutable_unix)

_APP_MAP: dict[str, tuple[str, str]] = {
    # Sistema
    "calculadora":      ("calc.exe",                    "gnome-calculator"),
    "calculator":       ("calc.exe",                    "gnome-calculator"),
    "paint":            ("mspaint.exe",                 "pinta"),
    "bloc de notas":    ("notepad.exe",                 "gedit"),
    "notepad":          ("notepad.exe",                 "gedit"),
    "explorador":       ("explorer.exe",                "nautilus"),
    "explorer":         ("explorer.exe",                "nautilus"),
    "task manager":     ("taskmgr.exe",                 "gnome-system-monitor"),
    "administrador de tareas": ("taskmgr.exe",          "gnome-system-monitor"),
    "panel de control": ("control.exe",                 "gnome-control-center"),
    "control panel":    ("control.exe",                 "gnome-control-center"),
    "configuración":    ("ms-settings:",                "gnome-control-center"),
    "settings":         ("ms-settings:",                "gnome-control-center"),
    "cmd":              ("cmd.exe",                     "bash"),
    "terminal":         ("cmd.exe",                     "bash"),
    "powershell":       ("powershell.exe",              "bash"),
    "registro":         ("regedit.exe",                 ""),
    "regedit":          ("regedit.exe",                 ""),
    # Multimedia
    "reproductor":      ("wmplayer.exe",                "vlc"),
    "media player":     ("wmplayer.exe",                "vlc"),
    "vlc":              ("vlc.exe",                     "vlc"),
    "grabadora de voz": ("SoundRecorder.exe",           "arecord"),
    "camara":           ("microsoft.windows.camera:",   "cheese"),
    "cámara":           ("microsoft.windows.camera:",   "cheese"),
    # Ofimática
    "word":             ("winword.exe",                 "libreoffice --writer"),
    "excel":            ("excel.exe",                   "libreoffice --calc"),
    "powerpoint":       ("powerpnt.exe",                "libreoffice --impress"),
    "outlook":          ("outlook.exe",                 "thunderbird"),
    "onenote":          ("onenote.exe",                 ""),
    # Utilidades
    "snipping tool":    ("SnippingTool.exe",            "gnome-screenshot"),
    "recortes":         ("SnippingTool.exe",            "gnome-screenshot"),
    "reloj":            ("ms-clock:",                   "gnome-clocks"),
    "clock":            ("ms-clock:",                   "gnome-clocks"),
    "mapas":            ("bingmaps:",                   ""),
    "maps":             ("bingmaps:",                   ""),
    "tiempo":           ("bingweather:",                ""),
    "weather":          ("bingweather:",                ""),
    "tienda":           ("ms-windows-store:",           ""),
    "store":            ("ms-windows-store:",           ""),
    # Desarrollo
    "visual studio code": ("code.exe",                  "code"),
    "vscode":           ("code.exe",                    "code"),
    "vs code":          ("code.exe",                    "code"),
    # Navegadores
    "chrome":           ("chrome.exe",                  "google-chrome"),
    "firefox":          ("firefox.exe",                 "firefox"),
    "edge":             ("msedge.exe",                  "microsoft-edge"),
    "opera":            ("opera.exe",                   "opera"),
    # Comunicación
    "discord":          ("discord.exe",                 "discord"),
    "teams":            ("ms-teams.exe",                "teams"),
    "slack":            ("slack.exe",                   "slack"),
    "telegram":         ("telegram.exe",                "telegram-desktop"),
    "whatsapp":         ("whatsapp.exe",                ""),
    "zoom":             ("zoom.exe",                    "zoom"),
    # Redes sociales / entretenimiento (abren en navegador solo si no hay app)
    "spotify":          ("spotify.exe",                 "spotify"),
    "steam":            ("steam.exe",                   "steam"),
}

# URIs de ms-settings para abrir paneles específicos de Windows
_MS_SETTINGS = {
    "wifi":             "ms-settings:network-wifi",
    "bluetooth":        "ms-settings:bluetooth",
    "pantalla":         "ms-settings:display",
    "display":          "ms-settings:display",
    "sonido":           "ms-settings:sound",
    "sound":            "ms-settings:sound",
    "privacidad":       "ms-settings:privacy",
    "actualizaciones":  "ms-settings:windowsupdate",
    "updates":          "ms-settings:windowsupdate",
    "aplicaciones":     "ms-settings:appsfeatures",
    "apps":             "ms-settings:appsfeatures",
}


def _is_windows() -> bool:
    return os.name == "nt"


def _launch(executable: str) -> tuple[bool, str]:
    """
    Lanza un ejecutable o URI de protocolo (ms-settings:, ms-clock:, etc.).
    Retorna (éxito, mensaje).
    """
    if not executable:
        return False, "No hay ejecutable definido para este sistema operativo."

    # URIs de protocolo Windows (ms-settings:, ms-clock:, etc.)
    if ":" in executable and not executable.endswith(".exe"):
        try:
            os.startfile(executable)  # type: ignore[attr-defined]
            return True, f"Abierto: {executable}"
        except Exception as exc:
            return False, f"Error abriendo URI '{executable}': {exc}"

    # Buscar el ejecutable en PATH
    found = shutil.which(executable)
    if found:
        try:
            subprocess.Popen(
                [found],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS if _is_windows() else 0,
            )
            return True, f"Lanzado: {executable}"
        except Exception as exc:
            return False, f"Error lanzando '{executable}': {exc}"

    # En Windows, intentar con shell=True como último recurso (soporta .exe del sistema)
    if _is_windows():
        try:
            subprocess.Popen(
                executable,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"Lanzado (shell): {executable}"
        except Exception as exc:
            return False, f"Error lanzando '{executable}' vía shell: {exc}"

    return False, f"No se encontró '{executable}' en el PATH del sistema."


# ─── Tools ────────────────────────────────────────────────────────────────────


@tool
def open_application(app_name: str) -> str:
    """
    Open a native application installed on the user's computer.
    Works with built-in Windows apps (Calculator, Paint, Notepad, etc.)
    and installed apps (Chrome, Discord, Spotify, VS Code, etc.).
    Always prefer this tool over open_url when the user asks to open a desktop app.
    Examples:
      open_application("calculator")
      open_application("paint")
      open_application("spotify")
      open_application("discord")
      open_application("notepad")
    """
    key = app_name.strip().lower()

    # Buscar en el mapa de apps conocidas
    entry = _APP_MAP.get(key)
    if entry:
        executable = entry[0] if _is_windows() else entry[1]
        ok, msg = _launch(executable)
        if ok:
            return f"✓ Aplicación abierta: {app_name}"
        # Si falla el ejecutable del mapa, intentar directamente con el nombre
        ok2, msg2 = _launch(app_name)
        if ok2:
            return f"✓ Aplicación abierta: {app_name}"
        return f"✗ No se pudo abrir '{app_name}'.\n  Intento 1: {msg}\n  Intento 2: {msg2}"

    # No está en el mapa → intentar directamente (puede estar instalada)
    ok, msg = _launch(app_name)
    if ok:
        return f"✓ Aplicación abierta: {app_name}"

    # Último intento: añadir .exe en Windows
    if _is_windows() and not app_name.endswith(".exe"):
        ok2, msg2 = _launch(app_name + ".exe")
        if ok2:
            return f"✓ Aplicación abierta: {app_name}.exe"

    return (
        f"✗ No se encontró la aplicación '{app_name}'.\n"
        f"  Asegúrate de que está instalada y accesible desde el PATH.\n"
        f"  Error: {msg}"
    )


@tool
def open_system_settings(setting: str) -> str:
    """
    Open a specific Windows Settings panel directly.
    Use this when the user asks to open WiFi, Bluetooth, Display, Sound,
    Privacy, Updates, or any system configuration panel.
    Examples:
      open_system_settings("wifi")
      open_system_settings("bluetooth")
      open_system_settings("display")
      open_system_settings("sound")
    """
    key = setting.strip().lower()

    uri = _MS_SETTINGS.get(key)
    if not uri:
        # Intentar con ms-settings: directamente
        uri = f"ms-settings:{key}"

    if not _is_windows():
        ok, msg = _launch("gnome-control-center")
        return f"✓ Configuración abierta" if ok else f"✗ {msg}"

    ok, msg = _launch(uri)
    return f"✓ Configuración '{setting}' abierta" if ok else f"✗ No se pudo abrir '{setting}': {msg}"


@tool
def open_file_explorer(path: str = "") -> str:
    """
    Open the file explorer (Windows Explorer / Finder) at a specific folder path.
    If no path is given, opens the default home folder.
    Examples:
      open_file_explorer()
      open_file_explorer("C:/Users/me/Desktop")
      open_file_explorer("C:/Users/me/Downloads")
    """
    if _is_windows():
        target = path.strip() if path.strip() else ""
        try:
            if target:
                subprocess.Popen(
                    ["explorer.exe", target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return f"✓ Explorador abierto en: {target}"
            else:
                subprocess.Popen(
                    ["explorer.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return "✓ Explorador de archivos abierto"
        except Exception as exc:
            return f"✗ Error abriendo el explorador: {exc}"
    else:
        ok, msg = _launch(f"nautilus {path}".strip())
        return f"✓ Explorador abierto" if ok else f"✗ {msg}"