"""
agent/tools/system_info.py

Herramientas de información del sistema y consultas generales para AMADEUS:
- Hora y fecha locales (sin necesidad de internet)
- Información del sistema (CPU, RAM, batería, disco, tiempo de actividad)
- Clima y pronóstico meteorológico (vía wttr.in)
- Calculadora matemática segura basada en AST
"""
import ast
import datetime
import math
import operator
import os
import platform
from typing import Any, Callable, Dict, List, Optional, Type, Union

from langchain_core.tools import tool


# ─── Hora y Fecha ─────────────────────────────────────────────────────────────


@tool
def get_current_time() -> str:
    """
    Obtiene la hora y fecha local actual del sistema con detalle completo.

    Úsala siempre que el usuario pregunte qué hora es, cuál es la fecha de hoy,
    qué día de la semana es o en qué semana/año estamos.

    Returns:
        str: Cadena formateada con hora actual, fecha completa en español,
            número de semana y día del año.
    """
    ahora: datetime.datetime = datetime.datetime.now()

    dias_es: Dict[str, str] = {
        "Monday": "Lunes",
        "Tuesday": "Martes",
        "Wednesday": "Miércoles",
        "Thursday": "Jueves",
        "Friday": "Viernes",
        "Saturday": "Sábado",
        "Sunday": "Domingo",
    }
    meses_es: Dict[int, str] = {
        1: "enero",
        2: "febrero",
        3: "marzo",
        4: "abril",
        5: "mayo",
        6: "junio",
        7: "julio",
        8: "agosto",
        9: "septiembre",
        10: "octubre",
        11: "noviembre",
        12: "diciembre",
    }

    nombre_dia: str = dias_es.get(ahora.strftime("%A"), ahora.strftime("%A"))
    nombre_mes: str = meses_es[ahora.month]
    numero_semana: int = ahora.isocalendar()[1]

    return (
        f"🕐 Hora actual:  {ahora.strftime('%H:%M:%S')}\n"
        f"📅 Fecha:        {nombre_dia}, {ahora.day} de {nombre_mes} de {ahora.year}\n"
        f"📆 Semana:       Semana {numero_semana} del año\n"
        f"🗓️  Día del año:  {ahora.timetuple().tm_yday}/365"
    )


@tool
def get_time_info(query: str) -> str:
    """
    Responde consultas específicas relacionadas con fechas y cálculo de días.

    Permite calcular cuántos días faltan para eventos como Navidad, Año Nuevo
    o el próximo fin de semana a partir de la fecha local actual.

    Args:
        query (str): Pregunta o consulta temporal del usuario
            (por ejemplo, "cuántos días faltan para navidad" o "cuándo es fin de semana").

    Returns:
        str: Respuesta con el cálculo de días correspondiente o la fecha actual de referencia.
    """
    ahora: datetime.datetime = datetime.datetime.now()
    consulta: str = query.lower().strip()

    resultados: List[str] = []

    if "navidad" in consulta or "25 de diciembre" in consulta:
        navidad = datetime.datetime(ahora.year, 12, 25)
        if ahora > navidad:
            navidad = datetime.datetime(ahora.year + 1, 12, 25)
        dias_restantes: int = (navidad - ahora).days
        resultados.append(
            f"🎄 Faltan {dias_restantes} días para Navidad ({navidad.strftime('%d/%m/%Y')})"
        )

    if "año nuevo" in consulta or "1 de enero" in consulta:
        ano_nuevo = datetime.datetime(ahora.year + 1, 1, 1)
        dias_restantes = (ano_nuevo - ahora).days
        resultados.append(
            f"🎆 Faltan {dias_restantes} días para Año Nuevo ({ano_nuevo.strftime('%d/%m/%Y')})"
        )

    if "fin de semana" in consulta or "finde" in consulta:
        dias_hasta_sabado: int = (5 - ahora.weekday()) % 7
        if dias_hasta_sabado == 0:
            resultados.append("🎉 ¡Hoy es sábado! Ya es fin de semana.")
        elif dias_hasta_sabado == 6 and ahora.weekday() == 6:
            resultados.append("🎉 Hoy es domingo, último día del fin de semana.")
        else:
            resultados.append(
                f"📅 Faltan {dias_hasta_sabado} días para el próximo fin de semana."
            )

    if resultados:
        return "\n".join(resultados)

    return (
        f"📅 Fecha y hora actual: {ahora.strftime('%A %d/%m/%Y %H:%M')}\n"
        f"Consulta específica: '{query}'\n"
        "No pude calcular esa fecha específica, pero puedo ayudarte con más detalle "
        "si me das las fechas exactas."
    )


# ─── Información del Sistema ──────────────────────────────────────────────────


@tool
def get_system_info() -> str:
    """
    Obtiene un informe completo del estado del sistema y recursos de hardware.

    Incluye sistema operativo, nombre del equipo, versión de Python, porcentaje de
    uso de CPU, memoria RAM utilizada y libre, espacio en disco, nivel de batería
    y tiempo de actividad (uptime).

    Returns:
        str: Resumen detallado de las métricas del sistema operativo y hardware.
    """
    try:
        import psutil
    except ImportError:
        return "Error: psutil no está instalado. Ejecuta: pip install psutil"

    lineas: List[str] = [
        f"💻 Sistema Operativo: {platform.system()} {platform.release()} ({platform.machine()})",
        f"🖥️  Equipo:            {platform.node()}",
        f"🐍 Python:            {platform.python_version()}",
    ]

    # CPU
    try:
        porcentaje_cpu: float = psutil.cpu_percent(interval=1)
        nucleos_logicos: Optional[int] = psutil.cpu_count(logical=True)
        nucleos_fisicos: Optional[int] = psutil.cpu_count(logical=False)
        lineas.append(
            f"⚙️  CPU:               {porcentaje_cpu}% usado  "
            f"({nucleos_fisicos} núcleos físicos, {nucleos_logicos} lógicos)"
        )
    except Exception:
        lineas.append("⚙️  CPU:               No disponible")

    # RAM
    try:
        memoria_ram = psutil.virtual_memory()
        ram_total_gb: float = memoria_ram.total / (1024 ** 3)
        ram_usada_gb: float = memoria_ram.used / (1024 ** 3)
        ram_libre_gb: float = memoria_ram.available / (1024 ** 3)
        lineas.append(
            f"🧠 RAM:               {ram_usada_gb:.1f} GB usados / {ram_total_gb:.1f} GB total "
            f"({memoria_ram.percent}% — {ram_libre_gb:.1f} GB libres)"
        )
    except Exception:
        lineas.append("🧠 RAM:               No disponible")

    # Disco
    try:
        ruta_disco: str = "C:\\" if os.name == "nt" else "/"
        uso_disco = psutil.disk_usage(ruta_disco)
        disco_total_gb: float = uso_disco.total / (1024 ** 3)
        disco_usado_gb: float = uso_disco.used / (1024 ** 3)
        disco_libre_gb: float = uso_disco.free / (1024 ** 3)
        lineas.append(
            f"💾 Disco ({ruta_disco}):      {disco_usado_gb:.1f} GB usados / {disco_total_gb:.1f} GB total "
            f"({uso_disco.percent}% — {disco_libre_gb:.1f} GB libres)"
        )
    except Exception:
        lineas.append("💾 Disco:             No disponible")

    # Batería
    try:
        bateria = psutil.sensors_battery()
        if bateria:
            estado_carga: str = (
                "🔌 Cargando" if bateria.power_plugged else "🔋 Batería"
            )
            tiempo_restante: str = ""
            if not bateria.power_plugged and bateria.secsleft > 0:
                minutos_totales: int = int(bateria.secsleft // 60)
                horas: int = minutos_totales // 60
                minutos: int = minutos_totales % 60
                tiempo_restante = f" — aprox. {horas}h {minutos}m restantes"
            lineas.append(
                f"🔋 Batería:           {bateria.percent:.0f}% ({estado_carga}{tiempo_restante})"
            )
        else:
            lineas.append(
                "🔋 Batería:           No disponible (equipo de escritorio)"
            )
    except Exception:
        lineas.append("🔋 Batería:           No disponible")

    # Tiempo encendido
    try:
        inicio_sistema = datetime.datetime.fromtimestamp(psutil.boot_time())
        tiempo_activo = datetime.datetime.now() - inicio_sistema
        horas_activo: int = int(tiempo_activo.total_seconds() // 3600)
        minutos_activo: int = int((tiempo_activo.total_seconds() % 3600) // 60)
        lineas.append(
            f"⏱️  Encendido desde:   {inicio_sistema.strftime('%d/%m/%Y %H:%M')} "
            f"({horas_activo}h {minutos_activo}m)"
        )
    except Exception:
        pass

    return "\n".join(lineas)


@tool
def get_battery_status() -> str:
    """
    Obtiene únicamente el porcentaje actual de batería y su estado de carga.

    Úsala cuando el usuario pregunte específicamente cuánta batería queda,
    si el equipo portátil se está cargando o cuánto tiempo de autonomía resta.

    Returns:
        str: Estado detallado de la batería con barra de progreso o aviso si es un PC de escritorio.
    """
    try:
        import psutil

        bateria = psutil.sensors_battery()
        if not bateria:
            return "🔋 Este equipo no tiene batería (es un equipo de escritorio o no se detecta)."

        estado: str = (
            "cargando 🔌" if bateria.power_plugged else "en batería 🔋"
        )
        texto_tiempo: str = ""
        if not bateria.power_plugged and bateria.secsleft > 0:
            minutos_totales: int = int(bateria.secsleft // 60)
            horas: int = minutos_totales // 60
            minutos: int = minutos_totales % 60
            texto_tiempo = (
                f"\n⏱️  Tiempo restante: aproximadamente {horas} hora(s) y {minutos} minuto(s)"
            )

        barra_nivel: str = _generar_barra_bateria(float(bateria.percent))
        return (
            f"🔋 Batería: {bateria.percent:.0f}%  {barra_nivel}\n"
            f"📌 Estado: {estado}"
            f"{texto_tiempo}"
        )
    except ImportError:
        return "Error: psutil no instalado. Ejecuta: pip install psutil"
    except Exception as exc:
        return f"No se pudo obtener el estado de la batería: {exc}"


def _generar_barra_bateria(porcentaje: float) -> str:
    """Genera una representación visual en texto del porcentaje de batería."""
    bloques_llenos: int = int(porcentaje / 10)
    bloques_vacios: int = 10 - bloques_llenos
    return f"[{'█' * bloques_llenos}{'░' * bloques_vacios}]"


# ─── Clima y Temperatura ──────────────────────────────────────────────────────


@tool
def get_weather(city: str = "") -> str:
    """
    Consulta las condiciones meteorológicas actuales y el pronóstico de 3 días para una ciudad.

    Utiliza el servicio público wttr.in (requiere conexión a internet). Si no se especifica
    una ciudad (cadena vacía), detecta automáticamente la ubicación aproximada por IP.

    Args:
        city (str): Nombre de la ciudad a consultar (por ejemplo, "Madrid",
            "Buenos Aires", "New York"). Por defecto es "" (detección automática).

    Returns:
        str: Informe meteorológico con temperatura, humedad, viento y pronóstico de 3 días.
    """
    try:
        import httpx
    except ImportError:
        return "Error: httpx no instalado. Ejecuta: pip install httpx"

    ubicacion: str = city.strip() if city.strip() else ""
    url_clima: str = f"https://wttr.in/{ubicacion}?format=j1&lang=es"

    try:
        respuesta = httpx.get(url_clima, timeout=8.0, follow_redirects=True)
        respuesta.raise_for_status()
        datos: Dict[str, Any] = respuesta.json()
    except httpx.TimeoutException:
        return (
            "⚠️ El servicio de clima tardó demasiado en responder. "
            "Comprueba tu conexión a internet."
        )
    except Exception as exc:
        return (
            f"⚠️ No se pudo obtener el clima: {exc}\n"
            "Comprueba que tienes conexión a internet."
        )

    try:
        condicion_actual: Dict[str, Any] = datos["current_condition"][0]
        area_cercana: Dict[str, Any] = datos["nearest_area"][0]

        area: str = area_cercana["areaName"][0]["value"]
        pais: str = area_cercana["country"][0]["value"]

        descripcion: str = condicion_actual["weatherDesc"][0]["value"]
        temp_c: str = condicion_actual["temp_C"]
        sensacion_c: str = condicion_actual["FeelsLikeC"]
        humedad: str = condicion_actual["humidity"]
        viento_kmph: str = condicion_actual["windspeedKmph"]
        direccion_viento: str = condicion_actual["winddir16Point"]
        visibilidad: str = condicion_actual["visibility"]
        indice_uv: str = condicion_actual["uvIndex"]
        presion: str = condicion_actual["pressure"]

        descripcion_es: str = _traducir_condicion_clima(descripcion)

        clima_hoy: Dict[str, Any] = datos["weather"][0]
        max_c: str = clima_hoy["maxtempC"]
        min_c: str = clima_hoy["mintempC"]

        lineas_pronostico: List[str] = []
        etiquetas_dias: List[str] = ["Hoy", "Mañana", "Pasado mañana"]
        for indice, dia in enumerate(datos["weather"][:3]):
            etiqueta: str = (
                etiquetas_dias[indice]
                if indice < len(etiquetas_dias)
                else f"Día {indice + 1}"
            )
            dia_max: str = dia["maxtempC"]
            dia_min: str = dia["mintempC"]
            dia_desc: str = _traducir_condicion_clima(
                dia["hourly"][4]["weatherDesc"][0]["value"]
            )
            prob_lluvia: str = dia["hourly"][4].get("chanceofrain", "?")
            lineas_pronostico.append(
                f"  {etiqueta:<15} {dia_desc:<25} 🌡️ {dia_min}°C – {dia_max}°C  🌧️ {prob_lluvia}%"
            )

        texto_ubicacion: str = (
            f"{area}, {pais}" if area else (city or "ubicación automática")
        )

        return (
            f"🌍 Clima en {texto_ubicacion}\n"
            f"{'─' * 45}\n"
            f"🌤️  Condición:    {descripcion_es}\n"
            f"🌡️  Temperatura:  {temp_c}°C  (sensación térmica: {sensacion_c}°C)\n"
            f"📊 Hoy:          Mín {min_c}°C — Máx {max_c}°C\n"
            f"💧 Humedad:      {humedad}%\n"
            f"💨 Viento:       {viento_kmph} km/h dirección {direccion_viento}\n"
            f"👁️  Visibilidad:  {visibilidad} km\n"
            f"☀️  Índice UV:    {indice_uv}\n"
            f"🔵 Presión:      {presion} hPa\n"
            f"\n📅 Pronóstico próximos días:\n"
            + "\n".join(lineas_pronostico)
        )
    except (KeyError, IndexError) as exc:
        return f"⚠️ No se pudo interpretar la respuesta del servicio de clima: {exc}"


def _traducir_condicion_clima(descripcion: str) -> str:
    """Traduce descripciones meteorológicas comunes del inglés al español con iconos."""
    traducciones: Dict[str, str] = {
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
    return traducciones.get(descripcion, descripcion)


# ─── Calculadora ──────────────────────────────────────────────────────────────


@tool
def calculate(expression: str) -> str:
    """
    Evalúa de forma segura una expresión matemática y devuelve el resultado numérico.

    Soporta operadores aritméticos (+, -, *, /, **, %), constantes (pi, e) y
    funciones matemáticas (sqrt, abs, round, floor, ceil, log, log10, sin, cos, tan).
    Úsala siempre para cualquier cálculo matemático en lugar de calcular mentalmente.

    Args:
        expression (str): Expresión matemática a evaluar
            (por ejemplo, "15 * 8 + 200 / 4", "sqrt(144)", "2 ** 10").

    Returns:
        str: Resultado formateado de la operación matemática o mensaje de error.
    """
    operadores_permitidos: Dict[Type[ast.AST], Callable[..., Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    nombres_permitidos: Dict[str, Any] = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }

    def _evaluar_nodo(nodo: ast.AST) -> Union[int, float]:
        if isinstance(nodo, ast.Constant) and isinstance(
            nodo.value, (int, float)
        ):
            return nodo.value
        if isinstance(nodo, ast.BinOp):
            operacion = operadores_permitidos.get(type(nodo.op))
            if operacion is None:
                raise ValueError(
                    f"Operación no permitida: {type(nodo.op).__name__}"
                )
            return float(
                operacion(_evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right))
            )
        if isinstance(nodo, ast.UnaryOp):
            operacion = operadores_permitidos.get(type(nodo.op))
            if operacion is None:
                raise ValueError(
                    f"Operación no permitida: {type(nodo.op).__name__}"
                )
            return float(operacion(_evaluar_nodo(nodo.operand)))
        if isinstance(nodo, ast.Call):
            nombre_funcion: Optional[str] = (
                nodo.func.id if isinstance(nodo.func, ast.Name) else None
            )
            if nombre_funcion not in nombres_permitidos:
                raise ValueError(f"Función no permitida: {nombre_funcion}")
            argumentos: List[Union[int, float]] = [
                _evaluar_nodo(arg) for arg in nodo.args
            ]
            return float(nombres_permitidos[nombre_funcion](*argumentos))
        if isinstance(nodo, ast.Name):
            if nodo.id in nombres_permitidos:
                valor = nombres_permitidos[nodo.id]
                if isinstance(valor, (int, float)):
                    return valor
            raise ValueError(f"Variable no permitida: {nodo.id}")
        raise ValueError(f"Expresión no soportada: {type(nodo).__name__}")

    expresion_limpia: str = expression.strip().replace("^", "**")

    try:
        arbol_sintactico: ast.Expression = ast.parse(
            expresion_limpia,
            mode="eval",
        )
        resultado: Union[int, float] = _evaluar_nodo(arbol_sintactico.body)

        if isinstance(resultado, float):
            if resultado == int(resultado):
                texto_resultado: str = str(int(resultado))
            else:
                texto_resultado = f"{resultado:.6g}"
        else:
            texto_resultado = str(resultado)

        return f"🧮 {expression} = {texto_resultado}"
    except ZeroDivisionError:
        return "❌ Error: División por cero."
    except Exception as exc:
        return f"❌ No se pudo calcular '{expression}': {exc}"
