from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from typing import Optional
from agents.assistant_agent import agent_executor, system_prompt
from fastapi import APIRouter

class MessageInput(BaseModel):
    input : str
    image : Optional[str] = None
    thread_id : str


router = APIRouter()

@router.get('/')
def greet():
    return {"message": "Welcome to imago intelligence"}


@router.post("/chat")
def chat(msg: MessageInput):
    user_input = msg.input
    image_url=msg.image
    response_text = ""

    try:
        full_input = user_input

        # Include image reference in the prompt if it's provided
        if image_url:
            full_input += f"\n[Image URL: {image_url}]"
            

        for chunk in agent_executor.stream(
        {
            "messages": [
                HumanMessage(content=full_input),
                SystemMessage(content=system_prompt),
            ]
        },
            config={"configurable": {"thread_id": msg.thread_id}}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    response_text += message.content

        if not response_text:
            return {"error": "No response generated."}

        return {"assistant": response_text}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}