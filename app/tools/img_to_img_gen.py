from gradio_client import Client, handle_file
from langchain.tools import tool
from services.cloudinary_upload import upload_image


@tool
def generate_image_from_image(prompt: str, image: str) -> str:
    """Use this tool when the user wants to modify, enhance, or apply transformations to an existing image using a text instruction. This includes changing background, color, lighting, objects, or artistic style."""
    print(f"\n[TOOL CALLED 🔧] generate_image_from_image with query: {prompt}")
    try:
        client = Client("black-forest-labs/FLUX.1-Kontext-Dev")
        result = client.predict(
            input_image=handle_file(
                image
            ),
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            guidance_scale=2.5,
            steps=28,
            api_name="/infer",
        )
        image_path = result[0]

    except Exception as e1:
        print(f"[FLUX.1-Kontext-Dev failed]", e1)

        try:
            client = Client("kontext-community/kontext-relight")
            result = client.predict(
                input_image=handle_file(
                    image
                ),
                prompt=prompt,
                illumination_dropdown="sunshine from window",
                direction_dropdown="auto",
                seed=0,
                randomize_seed=True,
                guidance_scale=2.5,
                api_name="/infer",
            )

            image_path = result[0][0]
        except Exception as e2:
            print(f"[kontext-relight failed]", e2)

            try:
                client = Client("hysts/ControlNet-v1-1")
                result = client.predict(
                    image=handle_file(
                        image
                    ),
                    prompt=prompt,
                    additional_prompt="best quality, extremely detailed",
                    negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                    num_images=1,
                    image_resolution=768,
                    num_steps=20,
                    guidance_scale=9,
                    seed=0,
                    low_threshold=100,
                    high_threshold=200,
                    api_name="/canny",
                )
                image_path = result[1]["image"]
            except Exception as e3:
                print("[ControlNet fallback failed]", e3)
                return f"Error : All image enhancing/editing models failed"

    if not image_path:
        return "Error: failed to enhance or edit the image"

    image_url = upload_image(image_path)

    if image_url:
        return f"Image successfully enhanced or edited:  {image_url}"
    else:
        return f"Image successfully enhanced or edited but upload failed. Local path:  {image_path}"


