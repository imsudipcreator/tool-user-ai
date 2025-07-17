from langchain.tools import tool
from gradio_client import Client


@tool
def advanced_coder(user_prompt: str):
    """Use this tool for generating advanced coding stuffs like a webpages, components and more. Use a descriptive prompt for better result."""
    print(f"\n[TOOL CALLED 🔧] advanced_coder with query: {user_prompt}")

    try:
        # input validation
        if not user_prompt or not user_prompt.strip():
            return "Error: Empty prompt provided"

        # Sanitize input to prevent injection attacks
        sanitized_prompt = user_prompt.strip()
        client = Client("Qwen/Qwen2.5-Coder-Artifacts")
        result = client.predict(query=sanitized_prompt, api_name="/generation_code")

        # validate result
        if not result or len(result) == 0:
            raise ValueError("[Qwen/Qwen2.5-Coder-Artifacts]: Empty response")

        response = result[0]

        if not response or not isinstance(response, str):
            raise ValueError("[Qwen/Qwen2.5-Coder-Artifacts]: Invaild response format")
        print(response)
        return response

    except Exception as e1:
        print("[Qwen/Qwen2.5-Coder-Artifacts failed]", e1)

        try:
            client = Client("MiniMaxAI/MiniMax-M1")
            result = client.predict(query=user_prompt, api_name="/generate_code")
            # validate result
            if not result or len(result) == 0:
                raise ValueError("[MiniMaxAI/MiniMax-M1]: Empty response")

            if not isinstance(result[0], dict) or "value" not in result[0]:
                raise ValueError("Invalid response format from fallback model")

            response = result[0]["value"]

            if not response or not isinstance(response, str):
                raise ValueError("[MiniMaxAI/MiniMax-M1]: Invaild response format")
            print(response)
            return response

        except Exception as e2:
            print("[All coding models failed]", e2)
            return "Error : All advanced coding models are currently unavailable"
