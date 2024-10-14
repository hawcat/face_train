import base64
import glob
import os
import shutil
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

client = OpenAI(
    base_url="https://aihubmix.com/v1",
    api_key="",
)

os.makedirs("images/sides", exist_ok=True)
os.makedirs("images/front", exist_ok=True)
os.makedirs("images/tongue", exist_ok=True)


def scan_images_in_folder(folder_path, extensions=[".jpg", ".jpeg", ".png"]):
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    return image_files


class params(BaseModel):
    is_face: Literal["front", "sides", "tongue"]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def classify_image(image_path):
    system_prompt = "你是一个图片分类助手，需要读取图片并分析其中内容，如果图片是正脸且不张嘴露出舌头，则返回front，如果图片是正脸且张嘴露出舌头，则返回tongue，如果图片是侧脸，则返回sides。"

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64, {encode_image(image_path=image_path)}"
                        },
                    }
                ],
            },
        ],
        response_format=params,
    )

    result: params = response.choices[0].message.parsed

    result = result.model_dump()

    if result["is_face"] == "front":
        shutil.move(
            image_path, os.path.join("images", "front", os.path.basename(image_path))
        )
    elif result["is_face"] == "sides":
        shutil.move(
            image_path, os.path.join("images", "sides", os.path.basename(image_path))
        )
    elif result["is_face"] == "tongue":
        shutil.move(
            image_path, os.path.join("images", "tongue", os.path.basename(image_path))
        )
    else:
        raise Exception("Invalid result")


if __name__ == "__main__":
    for image_path in tqdm(scan_images_in_folder("images")):
        classify_image(image_path)
