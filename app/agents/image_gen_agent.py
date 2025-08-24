from services.cloudinary_upload import upload_image
from gradio_client import Client
from typing import TypedDict, Literal
from uuid import uuid4
import requests
import base64
import os


class ImageGenResponse(TypedDict):
    type: Literal["error", "image"]
    image_url: str
    log: str


def image_gen_response(prompt: str) -> ImageGenResponse:
    try:
        client = Client("Qwen/Qwen-Image")
        result = client.predict(
            prompt="Shakespeare's Hamlet: Beautiful Ophelia in ornate princess dress, riding on a modern metro looking out the window at the sunset, with wired ear buds.Style is high-detail, hyper-realistic, cinematic. It cannot be an illustration or a painting. Styled as: 만화경, 絵巻, wimmelbilder, 长卷, चित्रकथा, Polyptych, fully photoreal. Fish-eye lens without vignette or frame. Extreme-45°-angle Dutch tilt. Extreme close-up details blended into wide panoramic continuity. High-speed action.",
            seed=0,
            randomize_seed=True,
            aspect_ratio="9:16",
            guidance_scale=4,
            num_inference_steps=50,
            prompt_enhance=True,
            api_name="/infer",
        )
        print(result)
        image_path = result[0]
    except Exception as e:
        print("[Qwen-Image failed]", e)
        try:
            client = Client("black-forest-labs/FLUX.1-dev")
            result = client.predict(
                prompt,
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
                image_path = cloudflare_image_gen(prompt)
                print(image_path)
            except Exception as e2:
                print("[FLUX.1-schnell/cloudflare failed]", e2)

                try:
                    client = Client("black-forest-labs/FLUX.1-schnell")
                    result = client.predict(
                        prompt,
                        seed=0,
                        randomize_seed=True,
                        width=1024,
                        height=1024,
                        num_inference_steps=20,
                        api_name="/infer",
                    )
                    print(result)
                    image_path = result[0]
                except Exception as e3:
                    print("[FLUX.1-schnell failed]", e3)

                    try:
                        client = Client("NihalGazi/FLUX-Pro-Unlimited")
                        result = client.predict(
                            prompt,
                            width=1280,
                            height=1280,
                            seed=0,
                            randomize=True,
                            server_choice="Google US Server",
                            api_name="/generate_image",
                        )
                        print(result)
                        image_path = result[0]
                    except Exception as e4:
                        print("[NihalGazi/FLUX-Pro-Unlimited failed]", e4)

                        try:
                            client = Client("stabilityai/stable-diffusion")
                            result = client.predict(
                                prompt, negative="", scale=9, api_name="/infer"
                            )
                            print(result)
                            image_path = result[0]["image"]
                        except Exception as e4:
                            print("[stable-diffusion failed]", e4)
                            return {
                                "image_url": "",
                                "type": "error",
                                "log": "All image generation services are currently unavailable.",
                            }

    if not image_path:
        return {
            "type": "error",
            "image_url": "",
            "log": "Image path could not be found",
        }

    image_url = upload_image(image_path)

    if image_url:
        return {
            "type": "image",
            "image_url": image_url,
            "log": "Image generated successfully",
        }
    else:
        return {
            "image_url": image_path,
            "log": "Image generated but upload failed",
            "type": "error",
        }


def cloudflare_image_gen(prompt: str) -> str:
    CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        print("[CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN is unavailable]")
        raise ValueError("Missing required environment variables")

    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell",
        headers={
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"prompt": prompt},
    ).json()

    base64_img = response["result"]["image"]
    if not base64_img:
        raise ValueError("Image generation failed or image not found in response")

    try:
        img_bytes = base64.b64decode(base64_img)
        image_path = f"/tmp/gen_{uuid4().hex}.png"
        with open(image_path, "wb") as f:
            f.write(img_bytes)

        return image_path
    except Exception as e:
        raise RuntimeError(f"Failed to write or save image: {str(e)}")
