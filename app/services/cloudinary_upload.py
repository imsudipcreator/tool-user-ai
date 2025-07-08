import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import os

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)


def upload_image(image_path: str, folder="imago") -> str | None:
    """
    Uploads image to Cloudinary and returns the secure URL.
    """
    try:
        result = cloudinary.uploader.upload(
            image_path,
            folder=folder,
            use_filename=True,
            unique_filename=False,
            overwrite=True,
        )
        return result.get("secure_url")

    except Exception as e:
        print(f"[Cloudinary Upload Error] {e}")
        return None
