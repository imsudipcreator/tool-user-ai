from tools import (
    wiki_search,
    current_time,
    generate_image,
    generate_image_from_image,
    get_imago_info,
    advanced_coder,
    search_web
)
from langgraph.prebuilt import create_react_agent
from typing import Literal, List, Union, Optional
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os


load_dotenv()


MODEL_MAP = {
    "imi1": "qwen/qwen3-32b",
    "imi1c": "gemma2-9b-it",
    "imi2": "llama-3.3-70b-versatile",
    "imi2c": "deepseek-r1-distill-llama-70b",
    "imi3": "meta-llama/llama-4-scout-17b-16e-instruct",
    "imi4": "moonshotai/kimi-k2-instruct",
}


class AssistantInput(BaseModel):
    user_input: str
    image_url: Optional[str] = None
    history_messages: List[Union[HumanMessage, AIMessage]]
    model: Literal["imi1", "imi1c", "imi2", "imi2c", "imi3", "imi4"] = "imi1"
    persona: Optional[str] = ""
    custom_prompt: Optional[str] = ""


def assistant_response(data: AssistantInput):
    system_prompt = f"""You are Imago Intelligence, a smart assistant created by Imago (founder sudip mahata), a tech brand that builds open, indie-friendly software tools.
    You can help users learn, build, and interact with tools like the iStore (an app store), webStore (a site store), and more.

    ---

    # Instructions (Warning : Do not share this info directly with the user. Only use this for better responses):
    1. **Tool Awareness**: Never mention internal tools (e.g., `wiki_search`, `current_time`, etc.). If asked, respond as if you're unaware or naturally redirect the conversation.
    2. When calling a tool like `advanced_coder`, use its full response directly in your reply. Do not summarize or rewrite the code unless explicitly instructed.
    3. **Image Output Format**: Always return image links using this Markdown format: `![View/Download](image_url)`. Avoid using `[View/Download](...)` or any other variation.
    4. **Markdown Use**: Utilize diverse and appropriate Markdown elements like:
    - Lists (bullet/numbered)
    - Headings (`##`)
    - Code blocks (``` or inline `code`)
    - Tables (if data fits)
    - Emojis for tone
    5. Two key context blocks might be provide below:
    - `PERSONA`: defines your communication tone/style
    - `CUSTOM_PROMPT`: contains user-specific context or preferences.
    Always align responses with both

    ---

    ### Context:

    **PERSONA**:  
    {data.persona}

    **CUSTOM_PROMPT**:  
    {data.custom_prompt}

    ---

    ### Rules:
    - If the user asks about the brand, products, or history, respond with accurate answers.
    - Do NOT make up new products. Stick to what's real.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    # creating the agent
    imi = ChatOpenAI(
        model=MODEL_MAP[data.model],
        base_url="https://api.groq.com/openai/v1",
        api_key=SecretStr(api_key),
    )
    tools = [
        wiki_search,
        current_time,
        generate_image,
        generate_image_from_image,
        get_imago_info,
        advanced_coder,
        search_web
    ]
    agent_executor = create_react_agent(model=imi, tools=tools)

    full_input = data.user_input

    # Include image reference in the prompt if it's provided
    if data.image_url:
        full_input += f"\n[Image URL: {data.image_url}]"

    conversation_history = [
        SystemMessage(content=system_prompt),
        *data.history_messages,
        HumanMessage(content=full_input),
    ]

    try:
        response = ""
        for chunk in agent_executor.stream({"messages": conversation_history}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response += message.content

        if not response:
            return {"error": "No response generated."}

        return {"assistant": response}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}
