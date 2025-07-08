import wikipedia
from langchain.tools import tool

@tool
def wiki_search(query: str) -> str:
    """Use this tool whenever the user asks a factual question about a person, place, or concept. It searches Wikipedia and returns a short summary."""
    print(f"\n[TOOL CALLED 🔧] wiki_search with query: {query}")
    try:
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=3)
    except wikipedia.exceptions.PageError as e:
        return f"No Wikipedia page found for '{query}'"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"