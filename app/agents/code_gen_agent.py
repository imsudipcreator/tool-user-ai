from gradio_client import Client, handle_file
from pydantic import BaseModel


class CodeAgentInput(BaseModel):
    prompt: str = ""


def code_agent_response(input: CodeAgentInput):
    try:
        client = Client("akhaliq/anycoder")
        result = client.predict(
            query=f"{input.prompt} and write html, css, javascript in a single html file. Don't add anything like '=== index.html ===' or anything similar. if you want you can add comment only <!-- index.html -->",
                vlm_image=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
                gen_image=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
                file=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf'),
                website_url="",
                enable_search=False,
                language="html",
                enable_image_generation=False,
                enable_image_to_image=False,
                image_to_image_prompt=None,
                text_to_image_prompt=None,
                enable_image_to_video=False,
                image_to_video_prompt=None,
                enable_text_to_video=False,
                text_to_video_prompt=None,
                enable_text_to_music=False,
                text_to_music_prompt=None,
                api_name="/generation_code"
        )

        return {"response": result[0], "error": None}
    except Exception as e:
        return {"response": "Could not generate your code", "error": e}
