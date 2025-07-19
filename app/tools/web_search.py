from langchain.tools import tool
from typing import Literal
from pydantic import BaseModel
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()


class WebSearchInput(BaseModel):
    query : str
    max_results : int = 5
    type : Literal["video", "image", "text", "news"]


@tool
def search_web(input : WebSearchInput):
    """Use this tool to search the web. It is very versatile, Use smartly. 
    This can provide videos, images, texts and even news with relevant info and sources depending on what type you want."""
    print(f"\n[TOOL CALLED 🔧] search_web with query : {input.query}, {input.max_results}, {input.type}")

    match input.type:
        case "image" : 
            results = DDGS().images(query=input.query, max_results=input.max_results)
            formatted = "\n\n".join(
                f"🔹 {r['title']}\n{r['image']}\n{r['source']}"
                for r in results
            )

            return formatted
        case "news":
            results = DDGS().news(query=input.query, max_results=input.max_results)
            formatted = "\n\n".join(
                f"🔹 {r['title']}\n{r['date']}\n{r['body']}\n{r['url']}"
                for r in results
            )

            return formatted
        case "video":
            results = DDGS().videos(query=input.query, max_results=input.max_results)
            formatted = "\n\n".join(
                f"🔹 {r['title']}\n{r['content']}\n{r['publisher']}"
                for r in results
            )

            return formatted
        case "text" | _:
            results = DDGS().text(query=input.query, max_results=input.max_results)
            formatted = "\n\n".join(
                f"🔹 {r['title']}\n{r['href']}\n{r['body']}"
                for r in results
            )

            return formatted
        











# google_search = GoogleSerperAPIWrapper()
# duckduckgo_search = DuckDuckGoSearchResults()



# @tool
# def search_web(input: str) -> str:
#     """Search the web intelligently. Uses Serper for recommendations, DuckDuckGo for general lookups."""
#     print(f"\n[TOOL CALLED 🔧] search_web with query : {input}")
#     try:
#         if any(word in input.lower() for word in ["best", "top", "recommend", "review", "vs", "comparison"]):
#             print("🔎 Using Serper...")
#             return google_search.run(input)
#         else:
#             print("🦆 Using DuckDuckGo...")
#             return duckduckgo_search.invoke(input)
#     except Exception as e:
#         print("❌ Serper failed. Falling back to DuckDuckGo.", e)
#         return duckduckgo_search.invoke(input)
