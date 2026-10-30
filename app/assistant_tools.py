import ast
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from langchain.tools import tool

MAX_EXPRESSION_LENGTH = 256
MAX_ABSOLUTE_VALUE = 1e100


def _evaluate_expression(node):
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only numeric values are allowed.")
        if abs(value) > MAX_ABSOLUTE_VALUE:
            raise ValueError("The number is too large.")
        return value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("Exponents must be between -100 and 100.")
            result = left**right
        else:
            raise ValueError("That math operator is not supported.")

        if not math.isfinite(result) or abs(result) > MAX_ABSOLUTE_VALUE:
            raise ValueError("The result is too large to calculate safely.")
        return result

    raise ValueError("Use numbers, parentheses, and basic math operators only.")


@tool
def calculate(expression: str) -> str:
    """Safely calculate an arithmetic expression using +, -, *, /, //, %, **, and parentheses."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return f"Expression must be {MAX_EXPRESSION_LENGTH} characters or fewer."

    try:
        parsed_expression = ast.parse(expression, mode="eval")
        result = _evaluate_expression(parsed_expression)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Could not calculate that expression: {exc}"

    return f"{expression} = {result}"


@tool
def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time in an IANA timezone such as UTC or America/Denver."""
    try:
        current_time = datetime.now(ZoneInfo(timezone_name))
    except (ZoneInfoNotFoundError, ValueError):
        return (
            f"Unknown timezone '{timezone_name}'. Use an IANA timezone such as "
            "UTC or America/Denver."
        )

    return current_time.isoformat(timespec="seconds")


@tool
def get_weather(city: str) -> str:
    """Get current weather and temperature in Celsius for a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather lookup is unavailable: set OPENWEATHER_API_KEY in the environment."
    if not city.strip() or len(city) > 100:
        return "Enter a city name of 1 to 100 characters."

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        return f"Weather service request failed: {exc}"

    if response.status_code == 404:
        return f"No weather data was found for '{city}'. Check the city name."
    if response.status_code == 401:
        return "Weather lookup failed because the OpenWeather API key was rejected."
    try:
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return f"Could not read the weather service response: {exc}"

    try:
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
    except (IndexError, KeyError, TypeError):
        return "The weather service returned an unexpected response."

    return (
        f"The current weather in {city} is {weather_description} "
        f"with a temperature of {temperature}°C."
    )


@tool
def get_location() -> str:
    """Get the configured default city for weather requests; this is not device geolocation."""
    return os.getenv("DEFAULT_LOCATION", "rome, italy")


@tool
def search_my_pdfs(query: str) -> str:
    """Search indexed PDFs belonging to the configured Django browser session."""
    session_key = os.getenv("CLI_DOCUMENT_SESSION_KEY")
    if not session_key:
        return (
            "PDF search is not configured. Set CLI_DOCUMENT_SESSION_KEY to the "
            "Django session key whose PDFs you own. Search is disabled until then."
        )
    if not query.strip():
        return "Enter a search question or keywords."

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project.settings")
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()

    from .services import find_document_chunks

    chunks = find_document_chunks(query, session_key)
    if not chunks:
        return "No matching passages were found in your PDFs."

    return "\n\n".join(
        f"[{chunk.document.filename}, page {chunk.page_number}]\n{chunk.content}"
        for chunk in chunks
    )
