from dotenv import load_dotenv
import cloudinary.uploader
import cloudinary
import os

load_dotenv()

required_vars = ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {','.join(missing_vars)}")

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
