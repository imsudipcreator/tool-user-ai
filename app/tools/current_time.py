from datetime import datetime
from langchain.tools import tool

@tool
def current_time() -> str:
    """Returns current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
