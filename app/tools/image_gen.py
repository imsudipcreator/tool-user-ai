from gradio_client import Client
from langchain.tools import tool
from services.cloudinary_upload import upload_image


@tool
def generate_image(user_prompt: str) -> str:
    """Use this tool to generate images through text prompts"""
    print(f"\n[TOOL CALLED 🔧] generate_image with query: {user_prompt}")

    try:
        client = Client("black-forest-labs/FLUX.1-dev")
        result = client.predict(
            prompt=user_prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            api_name="/infer",
        )
        print(result)
        image_path = result[0]
    except Exception as e1:
        print("[Flux.1-dev failed]", e1)
        try:
            client = Client("black-forest-labs/FLUX.1-schnell")
            result = client.predict(
                prompt="generate an ghibli art from your imagination",
                seed=0,
                randomize_seed=True,
                width=1024,
                height=1024,
                num_inference_steps=4,
                api_name="/infer",
            )
            print(result)
            image_path = result[0]
        except Exception as e2:
            print("[FLUX.1-schnell failed]", e2)

            client = Client("stabilityai/stable-diffusion")
            result = client.predict(
                prompt=user_prompt, 
                negative="", scale=9, 
                api_name="/infer"
            )
            print(result)
            image_path = result[0]["image"]

    image_url = upload_image(image_path)

    if image_url:
        return image_url
    else:
        return image_path
