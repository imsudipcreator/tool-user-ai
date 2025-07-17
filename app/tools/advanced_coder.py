from langchain.tools import tool
from gradio_client import Client


@tool
def advanced_coder(user_prompt: str):
    """Use this tool for generating adavanced coding stuffs like a webpages, components and more. Use a descriptive prompt for better result."""
    print(f"\n[TOOL CALLED 🔧] advanced_coder with query: {user_prompt}")


    try:
        client = Client("Qwen/Qwen2.5-Coder-Artifacts")
        result = client.predict(
                query=user_prompt,
                api_name="/generation_code"
        )
        response = result[0]
        print(response)
        return response

    except Exception as e1:
        print("[Qwen/Qwen2.5-Coder-Artifacts failed]", e1)

        try:
            client = Client("MiniMaxAI/MiniMax-M1")
            result = client.predict(
                    query=user_prompt,
                    api_name="/generate_code"
            )
            response = result[0]["value"]
            print(response)
            return response
        
        except Exception as e2:
            print("[All coding models failed]", e2)
            return "Error : All advanced coding models are currently unavailable"



