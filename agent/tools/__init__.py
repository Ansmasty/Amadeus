"""
agent/tools/__init__.py

Aggregates all AMADEUS tools into a single list consumed by the LangGraph agent.
"""
from agent.tools.filesystem import (
    list_directory,
    read_file,
    create_directory,
    delete_file,
    move_file,
    copy_file,
)
from agent.tools.browser import open_url, search_youtube
from agent.tools.data_analysis import read_excel, read_csv
from agent.tools.system_apps import (
    open_application,
    open_system_settings,
    open_file_explorer,
)
from agent.tools.system_info import (
    get_current_time,
    get_time_info,
    get_system_info,
    get_battery_status,
    get_weather,
    calculate,
)

ALL_TOOLS = [
    # Filesystem - read-only
    list_directory,
    read_file,
    # Filesystem - non-destructive write
    create_directory,
    # Filesystem - destructive (HITL required)
    delete_file,
    move_file,
    copy_file,
    # Browser
    open_url,
    search_youtube,
    # Data analysis
    read_excel,
    read_csv,
    # Sistema — apps nativas
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
]

__all__ = ["ALL_TOOLS"]
