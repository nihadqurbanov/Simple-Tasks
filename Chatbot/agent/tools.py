from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 2 * 3'."""
    allowed = set("0123456789+-*/(). ")
    if not all(char in allowed for char in expression):
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression))  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"
