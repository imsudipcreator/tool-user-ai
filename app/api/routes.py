from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.assistant_agent import assistant_response, AssistantInput
from pydantic import BaseModel
from typing import Optional, List, Literal
from fastapi import APIRouter


class ChatHistoryItem(BaseModel):
    sender: str
    content: str


class MessageInput(BaseModel):
    input: str = "Hi"
    image: Optional[str] = ""
    model: Literal["imi1", "imi1c", "imi2", "imi2c", "imi3", "imi4"] = "imi1"
    history: Optional[List[ChatHistoryItem]] = []
    custom_prompt: Optional[str] = ""
    persona: Optional[str] = ""


router = APIRouter()


@router.get("/")
def greet():
    return {"message": "Welcome to imago intelligence"}


@router.post("/chat")
def chat(msg: MessageInput):

    history_messages = []
    if msg.history:
        for m in msg.history:
            if m.sender == "user":
                history_messages.append(HumanMessage(content=m.content))
            elif m.sender == "assistant":
                history_messages.append(AIMessage(content=m.content))

    assistant_input = AssistantInput(
        user_input=msg.input,
        image_url=msg.image,
        history_messages=history_messages,
        model=msg.model,
    )
    result = assistant_response(data=assistant_input)
    # print(result)
    return result
