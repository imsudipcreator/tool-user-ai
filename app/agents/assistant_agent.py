from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
from tools import wiki_search, current_time, generate_image, generate_image_from_image
import os


load_dotenv()

system_prompt = """You are Imago Intelligence, a smart assistant created by Imago, a tech brand that builds open, indie-friendly software tools.
You can help users learn, build, and interact with tools like the iStore (an app store), webStore (a site store), and more.

If the user asks about the brand, products, or history, respond with accurate and short answers.
Do NOT make up new products. Stick to what's real.
"""

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")


model = ChatOpenAI(
    model="qwen-qwq-32b",
    base_url="https://api.groq.com/openai/v1",
    api_key=SecretStr(api_key) if api_key else None,
)

tools = [wiki_search, current_time, generate_image, generate_image_from_image]
agent_executor = create_react_agent(model, tools)