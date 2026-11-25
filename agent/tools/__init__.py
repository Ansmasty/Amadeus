"""
agent/tools/__init__.py

Agrega todas las herramientas de AMADEUS en una única lista (`ALL_TOOLS`)
consumida por el agente ReAct de LangGraph.
"""
from typing import List

from langchain_core.tools import BaseTool

from agent.tools.browser import open_url, search_web, search_youtube
from agent.tools.data_analysis import read_csv, read_excel
from agent.tools.dev import (
    git_diff,
    git_log,
    git_status,
    run_terminal_command,
    search_in_project,
)
from agent.tools.filesystem import (
    copy_file,
    create_directory,
    delete_file,
    edit_file,
    list_directory,
    move_file,
    read_file,
    write_file,
)
from agent.tools.interpreter import ejecutar_codigo_python
from agent.tools.screen import describe_screen, read_screen_text
from agent.tools.system_apps import (
    open_application,
    open_file_explorer,
    open_system_settings,
)
from agent.tools.system_info import (
    calculate,
    get_battery_status,
    get_current_time,
    get_system_info,
    get_time_info,
    get_weather,
)

ALL_TOOLS: List[BaseTool] = [
    # Sistema de archivos — solo lectura
    list_directory,
    read_file,
    # Sistema de archivos — escritura no destructiva
    create_directory,
    # Sistema de archivos — destructivo (requiere HITL)
    write_file,
    edit_file,
    delete_file,
    move_file,
    copy_file,
    # Navegador
    open_url,
    search_youtube,
    search_web,
    # Análisis de datos
    read_excel,
    read_csv,
    # Sistema — aplicaciones nativas
    open_application,
    open_system_settings,
    open_file_explorer,
    # Sistema — información y consultas
    get_current_time,
    get_time_info,
    get_system_info,
    get_battery_status,
    get_weather,
    calculate,
    # Pantalla — percepción visual
    read_screen_text,
    describe_screen,
    # Desarrollo — git, búsqueda en proyecto, terminal (destructivo)
    git_status,
    git_diff,
    git_log,
    search_in_project,
    run_terminal_command,
    # Autonomía e interacción (Computer Use / Code Interpreter — requiere HITL)
    ejecutar_codigo_python,
]

__all__: List[str] = ["ALL_TOOLS", "ejecutar_codigo_python"]
