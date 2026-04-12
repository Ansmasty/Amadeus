"""
agent/tools/system_info.py

Herramientas de información del sistema y consultas generales para AMADEUS.
- Hora y fecha locales (sin internet)
- Información del sistema (CPU, RAM, batería, disco)
- Clima y temperatura (wttr.in — sin API key)
- Cultura general (responde el LLM directamente, no necesita tool)
"""
import datetime
import os
import platform

from langchain_core.tools import tool


# ─── Hora y Fecha ─────────────────────────────────────────────────────────────

@tool
def get_current_time() -> str:
    """
    Get the current local time and date.
    Use this when the user asks: what time is it, what's today's date,
    what day is it, what year are we in, etc.
    Returns time, full date, day of week, and week number.
    """
    now = datetime.datetime.now()

    days_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado",
        "Sunday": "Domingo"
    }
    months_es = {
        1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
        5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
        9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
    }

    day_name = days_es.get(now.strftime("%A"), now.strftime("%A"))
    month_name = months_es[now.month]
    week_num = now.isocalendar()[1]

    return (
        f"🕐 Hora actual:  {now.strftime('%H:%M:%S')}\n"
        f"📅 Fecha:        {day_name}, {now.day} de {month_name} de {now.year}\n"
        f"📆 Semana:       Semana {week_num} del año\n"
        f"🗓️  Día del año:  {now.timetuple().tm_yday}/365"
    )


@tool
def get_time_info(query: str) -> str:
    """
    Answer specific time-related questions: days until a date, day of week
    for a specific date, how many days ago, time zones, etc.
    Examples:
      get_time_info("cuántos días faltan para navidad")
      get_time_info("qué día de la semana es el 25 de diciembre")
      get_time_info("cuántos días han pasado desde el 1 de enero")
    """
    now = datetime.datetime.now()
    q = query.lower().strip()

    results = []

    # Navidad
    if "navidad" in q or "25 de diciembre" in q:
        christmas = datetime.datetime(now.year, 12, 25)
        if now > christmas:
            christmas = datetime.datetime(now.year + 1, 12, 25)
        delta = (christmas - now).days
        results.append(f"🎄 Faltan {delta} días para Navidad ({christmas.strftime('%d/%m/%Y')})")

    # Año nuevo
    if "año nuevo" in q or "1 de enero" in q:
        new_year = datetime.datetime(now.year + 1, 1, 1)
        delta = (new_year - now).days
        results.append(f"🎆 Faltan {delta} días para Año Nuevo ({new_year.strftime('%d/%m/%Y')})")

    # Fin de semana
    if "fin de semana" in q or "finde" in q:
        days_until_saturday = (5 - now.weekday()) % 7
        if days_until_saturday == 0:
            results.append("🎉 ¡Hoy es sábado! Ya es fin de semana.")
        elif days_until_saturday == 6 and now.weekday() == 6:
            results.append("🎉 Hoy es domingo, último día del fin de semana.")
        else:
            results.append(f"📅 Faltan {days_until_saturday} días para el próximo fin de semana.")

    if results:
        return "\n".join(results)

    # Respuesta genérica con la fecha actual
    return (
        f"📅 Fecha y hora actual: {now.strftime('%A %d/%m/%Y %H:%M')}\n"
        f"Consulta específica: '{query}'\n"
        f"No pude calcular esa fecha específica, pero puedo ayudarte con más detalle si me das las fechas exactas."
    )


# ─── Información del Sistema ──────────────────────────────────────────────────

@tool
def get_system_info() -> str:
    """
    Get system information: OS, CPU usage, RAM usage, disk space, battery level,
    computer name, and uptime.
    Use this when the user asks about their computer's performance, battery,
    available storage, or system details.
    """
    try:
        import psutil
    except ImportError:
        return "Error: psutil no está instalado. Ejecuta: pip install psutil"

    lines = []

    # SO y máquina
    lines.append(f"💻 Sistema Operativo: {platform.system()} {platform.release()} ({platform.machine()})")
    lines.append(f"🖥️  Equipo:            {platform.node()}")
    lines.append(f"🐍 Python:            {platform.python_version()}")

    # CPU
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_phys = psutil.cpu_count(logical=False)
        lines.append(f"⚙️  CPU:               {cpu_percent}% usado  ({cpu_phys} núcleos físicos, {cpu_count} lógicos)")
    except Exception:
        lines.append("⚙️  CPU:               No disponible")

    # RAM
    try:
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024 ** 3)
        ram_used_gb  = ram.used  / (1024 ** 3)
        ram_free_gb  = ram.available / (1024 ** 3)
        lines.append(
            f"🧠 RAM:               {ram_used_gb:.1f} GB usados / {ram_total_gb:.1f} GB total "
            f"({ram.percent}% — {ram_free_gb:.1f} GB libres)"
        )
    except Exception:
        lines.append("🧠 RAM:               No disponible")

    # Disco (unidad C: en Windows o / en Unix)
    try:
        disk_path = "C:\\" if os.name == "nt" else "/"
        disk = psutil.disk_usage(disk_path)
        disk_total_gb = disk.total / (1024 ** 3)
        disk_used_gb  = disk.used  / (1024 ** 3)
        disk_free_gb  = disk.free  / (1024 ** 3)
        lines.append(
            f"💾 Disco ({disk_path}):      {disk_used_gb:.1f} GB usados / {disk_total_gb:.1f} GB total "
            f"({disk.percent}% — {disk_free_gb:.1f} GB libres)"
        )
    except Exception:
        lines.append("💾 Disco:             No disponible")

    # Batería
    try:
        battery = psutil.sensors_battery()
        if battery:
            charging = "🔌 Cargando" if battery.power_plugged else "🔋 Batería"
            time_left = ""
            if not battery.power_plugged and battery.secsleft > 0:
                mins = battery.secsleft // 60
                hrs  = mins // 60
                mins = mins % 60
                time_left = f" — aprox. {hrs}h {mins}m restantes"
            lines.append(f"🔋 Batería:           {battery.percent:.0f}% ({charging}{time_left})")
        else:
            lines.append("🔋 Batería:           No disponible (equipo de escritorio)")
    except Exception:
        lines.append("🔋 Batería:           No disponible")

    # Tiempo encendido
    try:
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.datetime.now() - boot_time
        hours   = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        lines.append(f"⏱️  Encendido desde:   {boot_time.strftime('%d/%m/%Y %H:%M')} ({hours}h {minutes}m)")
    except Exception:
        pass

    return "\n".join(lines)


@tool
def get_battery_status() -> str:
    """
    Get the current battery level and charging status.
    Use this when the user asks specifically about battery percentage,
    if the laptop is charging, or how much battery time is left.
    """
    try:
        import psutil
        battery = psutil.sensors_battery()
        if not battery:
            return "🔋 Este equipo no tiene batería (es un equipo de escritorio o no se detecta)."

        status = "cargando 🔌" if battery.power_plugged else "en batería 🔋"
        time_str = ""
        if not battery.power_plugged and battery.secsleft > 0:
            mins = battery.secsleft // 60
            hrs  = mins // 60
            mins = mins % 60
            time_str = f"\n⏱️  Tiempo restante: aproximadamente {hrs} hora(s) y {mins} minuto(s)"

        level_bar = _battery_bar(battery.percent)
        return (
            f"🔋 Batería: {battery.percent:.0f}%  {level_bar}\n"
            f"📌 Estado: {status}"
            f"{time_str}"
        )
    except ImportError:
        return "Error: psutil no instalado. Ejecuta: pip install psutil"
    except Exception as exc:
        return f"No se pudo obtener el estado de la batería: {exc}"


def _battery_bar(percent: float) -> str:
    """Genera una barra visual de batería."""
    filled = int(percent / 10)
    empty  = 10 - filled
    return f"[{'█' * filled}{'░' * empty}]"


# ─── Clima y Temperatura ──────────────────────────────────────────────────────

@tool
def get_weather(city: str = "") -> str:
    """
    Get current weather conditions and temperature for a city.
    Uses wttr.in (free, no API key required, requires internet).
    If no city is provided, tries to use the user's approximate location.
    Use this when the user asks about weather, temperature, rain, humidity, wind, etc.
    Examples:
      get_weather("Madrid")
      get_weather("Buenos Aires")
      get_weather("New York")
      get_weather("")  → uses auto-detected location
    """
    try:
        import httpx
    except ImportError:
        return "Error: httpx no instalado. Ejecuta: pip install httpx"

    location = city.strip() if city.strip() else ""
    url = f"https://wttr.in/{location}?format=j1&lang=es"

    try:
        response = httpx.get(url, timeout=8.0, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        return "⚠️ El servicio de clima tardó demasiado en responder. Comprueba tu conexión a internet."
    except Exception as exc:
        return f"⚠️ No se pudo obtener el clima: {exc}\nComprueba que tienes conexión a internet."

    try:
        current = data["current_condition"][0]
        nearest = data["nearest_area"][0]

        # Ubicación detectada
        area  = nearest["areaName"][0]["value"]
        country = nearest["country"][0]["value"]

        # Condición actual
        desc = current["weatherDesc"][0]["value"]
        temp_c    = current["temp_C"]
        feels_c   = current["FeelsLikeC"]
        humidity  = current["humidity"]
        wind_kmph = current["windspeedKmph"]
        wind_dir  = current["winddir16Point"]
        visibility = current["visibility"]
        uv_index  = current["uvIndex"]
        pressure  = current["pressure"]

        # Traducción básica de condiciones comunes
        desc_es = _translate_weather(desc)

        # Pronóstico de hoy (máx/mín)
        today = data["weather"][0]
        max_c = today["maxtempC"]
        min_c = today["mintempC"]

        # Pronóstico próximos días
        forecast_lines = []
        days_es = ["Hoy", "Mañana", "Pasado mañana"]
        for i, day in enumerate(data["weather"][:3]):
            label = days_es[i] if i < len(days_es) else f"Día {i+1}"
            d_max = day["maxtempC"]
            d_min = day["mintempC"]
            d_desc = _translate_weather(day["hourly"][4]["weatherDesc"][0]["value"])
            rain_chance = day["hourly"][4].get("chanceofrain", "?")
            forecast_lines.append(
                f"  {label:<15} {d_desc:<25} 🌡️ {d_min}°C – {d_max}°C  🌧️ {rain_chance}%"
            )

        location_str = f"{area}, {country}" if area else (city or "ubicación automática")

        return (
            f"🌍 Clima en {location_str}\n"
            f"{'─' * 45}\n"
            f"🌤️  Condición:    {desc_es}\n"
            f"🌡️  Temperatura:  {temp_c}°C  (sensación térmica: {feels_c}°C)\n"
            f"📊 Hoy:          Mín {min_c}°C — Máx {max_c}°C\n"
            f"💧 Humedad:      {humidity}%\n"
            f"💨 Viento:       {wind_kmph} km/h dirección {wind_dir}\n"
            f"👁️  Visibilidad:  {visibility} km\n"
            f"☀️  Índice UV:    {uv_index}\n"
            f"🔵 Presión:      {pressure} hPa\n"
            f"\n📅 Pronóstico próximos días:\n"
            + "\n".join(forecast_lines)
        )
    except (KeyError, IndexError) as exc:
        return f"⚠️ No se pudo interpretar la respuesta del servicio de clima: {exc}"


def _translate_weather(desc: str) -> str:
    """Traducción básica de condiciones meteorológicas comunes."""
    translations = {
        "Sunny": "Soleado ☀️",
        "Clear": "Despejado 🌙",
        "Partly cloudy": "Parcialmente nublado ⛅",
        "Cloudy": "Nublado ☁️",
        "Overcast": "Cubierto ☁️",
        "Mist": "Niebla 🌫️",
        "Fog": "Niebla densa 🌫️",
        "Light rain": "Lluvia ligera 🌦️",
        "Moderate rain": "Lluvia moderada 🌧️",
        "Heavy rain": "Lluvia intensa 🌧️",
        "Light snow": "Nieve ligera 🌨️",
        "Moderate snow": "Nieve moderada ❄️",
        "Heavy snow": "Nieve intensa ❄️",
        "Thunderstorm": "Tormenta eléctrica ⛈️",
        "Blizzard": "Ventisca 🌨️",
        "Freezing drizzle": "Llovizna helada 🌨️",
        "Drizzle": "Llovizna 🌦️",
        "Patchy rain possible": "Posible lluvia parcial 🌦️",
        "Light drizzle": "Llovizna ligera 🌦️",
        "Torrential rain shower": "Aguacero torrencial 🌧️",
    }
    return translations.get(desc, desc)


# ─── Calculadora ──────────────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely and return the result.
    Supports: +, -, *, /, ** (power), % (modulo), sqrt, abs, round, etc.
    Use this when the user asks to calculate something, convert units,
    or do any arithmetic operation.
    Examples:
      calculate("2 + 2")
      calculate("sqrt(144)")
      calculate("15 * 8 + 200 / 4")
      calculate("2 ** 10")
    """
    import math
    import ast
    import operator

    # Operaciones permitidas (sin eval() directo por seguridad)
    allowed_ops = {
        ast.Add:  operator.add,
        ast.Sub:  operator.sub,
        ast.Mult: operator.mul,
        ast.Div:  operator.truediv,
        ast.Pow:  operator.pow,
        ast.Mod:  operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    allowed_names = {
        "sqrt": math.sqrt,
        "abs":  abs,
        "round": round,
        "floor": math.floor,
        "ceil":  math.ceil,
        "log":   math.log,
        "log10": math.log10,
        "sin":   math.sin,
        "cos":   math.cos,
        "tan":   math.tan,
        "pi":    math.pi,
        "e":     math.e,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Operación no permitida: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Operación no permitida: {type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in allowed_names:
                raise ValueError(f"Función no permitida: {func_name}")
            args = [_eval(a) for a in node.args]
            return allowed_names[func_name](*args)
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Variable no permitida: {node.id}")
        raise ValueError(f"Expresión no soportada: {type(node).__name__}")

    expr_clean = expression.strip().replace("^", "**")

    try:
        tree = ast.parse(expr_clean, mode="eval")
        result = _eval(tree.body)

        # Formatear resultado
        if isinstance(result, float):
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{result:.6g}"
        else:
            result_str = str(result)

        return f"🧮 {expression} = {result_str}"
    except ZeroDivisionError:
        return "❌ Error: División por cero."
    except Exception as exc:
        return f"❌ No se pudo calcular '{expression}': {exc}"