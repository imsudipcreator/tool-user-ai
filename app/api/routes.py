from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.assistant_agent import assistant_response, AssistantInput
from agents.image_gen_agent import image_gen_response
from agents.async_image_gen_agent import async_image_gen_response
from typing import Optional, List, Literal
from pydantic import BaseModel
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
    persona: Optional[str] = "Default"



class ImageGenInput(BaseModel):
    prompt: str = "Generate an image of a dog"


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
        custom_prompt=msg.custom_prompt,
        persona=msg.persona
    )
    result = assistant_response(data=assistant_input)
    # print(result)
    return result



@router.post("/generate/image")
def generate_image(input : str):
    result = image_gen_response(prompt=input)

    return result


@router.post("/generate/image/async")
def generate_image_async(input : str):
    result = async_image_gen_response(prompt=input)
    return result

