from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.assistant_agent import agent_executor, system_prompt
from pydantic import BaseModel
from typing import Optional, List
from fastapi import APIRouter

class ChatHistoryItem(BaseModel):
    sender : str
    content : str

class MessageInput(BaseModel):
    input : str
    image : Optional[str] = None
    history : Optional[List[ChatHistoryItem]] = [] 

router = APIRouter()

@router.get('/')
def greet():
    return {"message": "Welcome to imago intelligence"}


@router.post("/chat")
def chat(msg: MessageInput):
    user_input = msg.input
    image_url=msg.image
    response_text = ""

    history_messages = []
    if msg.history:
        for m in msg.history:
            if m.sender == "user":
                history_messages.append(HumanMessage(content=m.content))
            elif m.sender == "assistant":
                history_messages.append(AIMessage(content=m.content))

    try:
        full_input = user_input

        # Include image reference in the prompt if it's provided
        if image_url:
            full_input += f"\n[Image URL: {image_url}]"
            
        history_messages.append(HumanMessage(content=full_input))

        for chunk in agent_executor.stream(
        {
            "messages": [
                *history_messages,
                SystemMessage(content=system_prompt),
            ]
        }
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response_text += message.content

        if not response_text:
            return {"error": "No response generated."}

        return {"assistant": response_text}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}