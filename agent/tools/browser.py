"""
agent/tools/browser.py

Herramientas de automatización de navegador y búsqueda web para AMADEUS.
Utiliza el módulo estándar `webbrowser` y `httpx` para consultas controladas.
"""
import re
import webbrowser
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from langchain_core.tools import tool


def _limpiar_etiquetas_html(texto: str) -> str:
    """Elimina etiquetas HTML y normaliza espacios en blanco de un fragmento."""
    texto_sin_html: str = re.sub(r"<.*?>", "", texto)
    texto_sin_html = texto_sin_html.replace("&quot;", '"').replace("&#39;", "'")
    texto_sin_html = re.sub(r"\s+", " ", texto_sin_html)
    return texto_sin_html.strip()


@tool
def open_url(url: str) -> str:
    """
    Abre una dirección URL en el navegador web predeterminado del sistema operativo.

    Acepta URLs con esquema `http://` o `https://`. Si no se proporciona esquema,
    se asume automáticamente `https://`. Úsala para navegar a cualquier sitio web.

    Args:
        url (str): Dirección URL completa o dominio del sitio web a abrir
            (por ejemplo, "https://www.google.com" o "github.com").

    Returns:
        str: Mensaje indicando si la URL se abrió correctamente o el detalle del error.
    """
    url_limpia: str = url.strip()
    if not url_limpia.startswith(("http://", "https://")):
        url_limpia = "https://" + url_limpia

    try:
        abierto: bool = webbrowser.open(url_limpia)
        if abierto:
            return f"Opened in browser: {url_limpia}"
        return (
            "Could not open browser (no default browser configured on this system). "
            f"URL was: {url_limpia}"
        )
    except Exception as exc:
        return f"Error opening browser: {exc}"


@tool
def search_youtube(query: str) -> str:
    """
    Busca videos en YouTube que coincidan con la consulta dada y abre los resultados.

    Construye la URL de búsqueda de YouTube y abre la página de resultados en el
    navegador web predeterminado del sistema.

    Args:
        query (str): Términos de búsqueda para encontrar videos en YouTube
            (por ejemplo, "tutorial de python langchain").

    Returns:
        str: Confirmación de apertura de la búsqueda en el navegador o mensaje de error.
    """
    consulta_limpia: str = query.strip()
    if not consulta_limpia:
        return "Error: Search query cannot be empty."

    consulta_codificada: str = quote_plus(consulta_limpia)
    url_busqueda: str = (
        f"https://www.youtube.com/results?search_query={consulta_codificada}"
    )
    return str(open_url.invoke({"url": url_busqueda}))


@tool
def search_web(query: str) -> str:
    """
    Realiza una búsqueda en la web sobre noticias, actualidad, investigación o datos generales.

    Consulta DuckDuckGo HTML y devuelve un resumen estructurado con los primeros
    resultados (título, enlace limpio y fragmento descriptivo). Úsala cuando el usuario
    pregunte sobre eventos recientes o temas que requieran fuentes externas en internet.

    Args:
        query (str): Consulta o términos de búsqueda a investigar en la web.

    Returns:
        str: Lista formateada con los principales resultados web o mensaje de error.
    """
    consulta: str = query.strip()
    if not consulta:
        return "Error: Search query cannot be empty."

    url_buscador: str = "https://html.duckduckgo.com/html/"
    try:
        respuesta = httpx.get(
            url_buscador,
            params={"q": consulta},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
            follow_redirects=True,
        )
        respuesta.raise_for_status()
    except Exception as exc:
        return f"Error searching the web: {exc}"

    contenido_html: str = respuesta.text
    bloques_resultado: List[Tuple[str, str]] = re.findall(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>',
        contenido_html,
        re.S,
    )
    fragmentos: List[str] = re.findall(
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        contenido_html,
        re.S,
    )

    if not bloques_resultado:
        return f"No se encontraron resultados para: {consulta}"

    lineas_salida: List[str] = [f"Resultados web para: {consulta}"]
    for indice, (enlace_crudo, titulo_html) in enumerate(
        bloques_resultado[:5],
        start=1,
    ):
        titulo: str = _limpiar_etiquetas_html(titulo_html)
        enlace_analizado = urlparse(enlace_crudo)
        parametros_consulta: Dict[str, List[str]] = parse_qs(
            enlace_analizado.query
        )
        enlace_limpio: str = unquote(
            parametros_consulta.get("uddg", [enlace_crudo])[0]
        )
        resumen: str = (
            _limpiar_etiquetas_html(fragmentos[indice - 1])
            if indice - 1 < len(fragmentos)
            else ""
        )
        lineas_salida.append(f"{indice}. {titulo}")
        lineas_salida.append(f"   {enlace_limpio}")
        if resumen:
            lineas_salida.append(f"   {resumen}")

    return "\n".join(lineas_salida)
