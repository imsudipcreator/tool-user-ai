import wikipedia
from langchain.tools import tool

@tool
def wiki_search(query: str) -> str:
    """Use this tool whenever the user asks a factual question about a person, place, or concept. It searches Wikipedia and returns a short summary."""
    print(f"\n[TOOL CALLED 🔧] wiki_search with query: {query}")
    return wikipedia.summary(query, sentences=3)