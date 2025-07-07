from pydantic import SecretStr, BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from fastapi import FastAPI
from dotenv import load_dotenv
from datetime import datetime
import wikipedia
import os


load_dotenv()


class MessageInput(BaseModel):
    input: str


app = FastAPI()


@tool
def wiki_search(query: str) -> str:
    """Use this tool whenever the user asks a factual question about a person, place, or concept. It searches Wikipedia and returns a short summary."""
    print(f"\n[TOOL CALLED 🔧] wiki_search with query: {query}")
    return wikipedia.summary(query, sentences=3)


@tool
def current_time() -> str:
    """Returns current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



system_prompt = """You are Imago Intelligence, a smart assistant created by Imago, a tech brand that builds open, indie-friendly software tools.
You can help users learn, build, and interact with tools like the iStore (an app store), webStore (a site store), and more.

If the user asks about the brand, products, or history, respond with accurate and short answers.
Do NOT make up new products. Stick to what's real.
"""

api_key = os.getenv("GROQ_API_KEY")
model = ChatOpenAI(
    model="qwen-qwq-32b",
    base_url="https://api.groq.com/openai/v1",
    api_key=SecretStr(api_key) if api_key else None,
)

tools = [wiki_search, current_time]
agent_executor = create_react_agent(model, tools)




@app.get("/")
def greet():
    return { "message" : "Welcome to imago intelligence"}


@app.post("/chat")
def chat(msg: MessageInput):
    user_input = msg.input
    response_text = ""

    for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input), SystemMessage(content=system_prompt)]}):
        if "agent" in chunk and "messages" in chunk["agent"]:
            for message in chunk["agent"]["messages"]:
                response_text += message.content

    if not response_text:
        return { "error": "No response generated." }


    return { "assistant" : response_text}
