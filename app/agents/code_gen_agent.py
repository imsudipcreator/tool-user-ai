from gradio_client import Client, handle_file
from pydantic import BaseModel


class CodeAgentInput(BaseModel):
    prompt: str = ""


def code_agent_response(input: CodeAgentInput):
    try:
        client = Client("akhaliq/anycoder")
        result = client.predict(
            query=input.prompt,
            image=None,
            file=None,
            website_url="",
            enable_search=False,
            language="html",
            api_name="/generation_code",
        )

        return {"response": result[0], "error": None}
    except Exception as e:
        return {"response": "Could not generate your code", "error": e}
