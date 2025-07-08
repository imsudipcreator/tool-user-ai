from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from agents.assistant_agent import agent_executor, system_prompt
from fastapi import APIRouter

class MessageInput(BaseModel):
    input : str


router = APIRouter()

@router.get('/')
def greet():
    return {"message": "Welcome to imago intelligence"}


@router.post("/chat")
def chat(msg: MessageInput):
    user_input = msg.input
    response_text = ""

    for chunk in agent_executor.stream(
        {
            "messages": [
                HumanMessage(content=user_input),
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