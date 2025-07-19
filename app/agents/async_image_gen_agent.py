from services.cloudinary_upload import upload_image
from .image_gen_agent import ImageGenResponse
from dotenv import load_dotenv
import requests
import tempfile
import time
import os

load_dotenv()


def async_image_gen_response(prompt: str) -> ImageGenResponse:
    API_KEY = os.getenv("STABLEHORDE_API_KEY")

    if not API_KEY:
        print("Stable Horde API key is unavailable")
        raise ValueError("Missing required environment variable")

    ENDPOINT = "https://stablehorde.net/api/v2/generate/async"
    headers = {"apikey": API_KEY}

    payload = {
        "prompt": prompt,
        "params": {"n": 1, "width": 512, "height": 512, "steps": 20, "cfg_scale": 7},
        "nsfw": False,
        "censor_nsfw": True,
    }
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()

    task_id = response.json()["id"]
    print("Task ID: ", task_id)

    status_url = f"https://stablehorde.net/api/v2/generate/status/{task_id}"
    while True:
        status = requests.get(status_url).json()
        if status.get("done", False):
            break

        print("Waiting... Queue position:", status.get("queue_position"))
        time.sleep(5)

    result_url = f"https://stablehorde.net/api/v2/generate/status/{task_id}"
    result = requests.get(result_url).json()

    print("result: ", result)

    try:
        temp_image_url = result["generations"][0]["img"]
        image_url = upload_stable_horde_to_cloudinary(temp_image_url)
        if image_url:
            return {
                "type": "image",
                "image_url": image_url,
                "log": "image generated successfully",
            }
        else:
            return {
                "type": "error",
                "image_url": "",
                "log": "Error occured in getting cloudinary url",
            }
    except Exception as e:
        return {
            "type": "error",
            "image_url": "",
            "log": f"Failed to retrieve image: {e}",
        }


def upload_stable_horde_to_cloudinary(image_url: str) -> str | None:
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_file:
            tmp_file.write(response.content)
            temp_path = tmp_file.name

        cloudinary_url = upload_image(temp_path)

        os.remove(temp_path)

        return cloudinary_url

    except Exception as e:
        print(f"[StableHorde to Cloudinary Upload Error] {e}")
        return None
