from gradio_client import Client
from langchain.tools import tool
from services.cloudinary_upload import upload_image


@tool
def generate_image(user_prompt: str) -> str:
    """Use this tool when the user wants to create a completely new image from scratch based on a text description. Do NOT use this for editing existing images."""
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
                prompt=user_prompt,
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

            try:
                client = Client("stabilityai/stable-diffusion")
                result = client.predict(
                    prompt=user_prompt, 
                    negative="", scale=9, 
                    api_name="/infer"
                )
                print(result)
                image_path = result[0]["image"]
            except Exception as e3:
                print("[stable-diffusion failed]", e3)
                return "Error: All image generation services are currently unavailable."

    if not image_path:
        return "Error : failed to generate images"

    image_url = upload_image(image_path)

    if image_url:
        return f"Image generated successfully: {image_url}"
    else:
       return f"Image generated but upload failed. Local path: {image_path}"
