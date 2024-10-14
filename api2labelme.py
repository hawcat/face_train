import base64
import glob
import json
import os
import uuid
from pprint import pprint

import cv2
import numpy as np
import oss2
import requests
from tqdm import tqdm

# 阿里云 OSS 的配置信息

# 阿里云 OSS 初始化



def scan_images_in_folder(folder_path, extensions=[".jpg", ".jpeg", ".png"]):
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    return image_files


def scan_json_in_folder(folder_path, extensions=[".json"]):
    json_files = []
    for ext in extensions:
        json_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
    return json_files


def read_image_write_base64(json_file_path, dataset_dir):
    # 读取现有的JSON文件
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)

    # 将图像转换为Base64并更新到JSON数据
    image_base64 = image_to_base64(os.path.join(dataset_dir, json_data["imagePath"]))
    json_data["imageData"] = image_base64

    # 将更新后的数据写回JSON文件
    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def upload_file_to_oss(file_path):
    file_name = os.path.basename(file_path)
    file_name_source, file_extension = os.path.splitext(file_name)
    custom_file_name = f"{uuid.uuid4()}{file_extension}"
    bucket.put_object(f"uuid/{custom_file_name}", open(file_path, "rb"))

    return f"{public_url}/uuid/{custom_file_name}"


class FaceAnalysis:
    def __init__(self) -> None:
        self.app_id = "5d47d0c26eeb6286"
        self.app_secret = "bf6d005c24b9204d55331f5836d2ebcc"

        train_attributes = [
            ("spot_no", 2),
            ("pockmark_no", 65536),
        ]

        self.no_sum_train = sum([value for name, value in train_attributes])
        # self.no_sum = 34359738368

    def get_face_analysis(self, image_url: str):
        credentials = f"{self.app_id}:{self.app_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        # API的URL
        detect_types = f"{self.no_sum_train}"  # 将此替换为实际的检测类型
        url = f"https://api.yimei.ai/v2/api/face/analysis/{detect_types}"

        # 请求头
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # 请求数据
        data = {"image": image_url, "result_type": 2}

        # 发送POST请求
        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None


def image_to_base64(image_path):
    # 读取图片并将其转换为Base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


name_mapping_spot = {
    "B_QTB": "ban",  # #18FFFF in BGR 斑_其他斑
    "Z_Z": "zhi",  # #4740A5 in BGR 痣
    "B_QB": "ban",  # #FFE57F in BGR 斑_雀斑
    "B_HHB": "ban",  # #40FF81 in BGR 斑_黄褐斑
}
name_mapping_pockmark = {
    "CC_DD": "dou",  # #E040FB in BGR 痘痘
    "CC_DY": "douyin",  # #7C4DFF in BGR 痘印
}


def build_shape(cls, points, type):
    if len(points) > 2:
        print(f"{cls} has more than 2 points: {points}")
    return {
        "label": cls,
        "points": points,
        "shape_type": type,
        "group_id": None,
        "description": "",
        "flags": {},
        "mask": None,
    }


def convert_txt_to_labelme_json(api_data: dict, image_path):
    if api_data["code"] == 21001:
        print(f"{image_path}->{api_data['msg']}")
        return
    labelme_json = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": None,
        "imageWidth": None,
    }

    base_name = os.path.basename(image_path)
    output_dir = os.path.dirname(image_path)
    labelme_json["imagePath"] = base_name
    labelme_json["imageData"] = None

    image = cv2.imread(image_path)
    # 获取图像高度和宽度
    h, w = image.shape[:2]
    labelme_json["imageHeight"] = h
    labelme_json["imageWidth"] = w

    pockmarks: list = api_data["pockmark"]["category"]
    spots: list = api_data["spot"]["category"]
    for pockmark in pockmarks:
        cls = pockmark["cls"]
        label_cls = name_mapping_pockmark[cls]
        rects = pockmark.get("rects", None)
        if rects:
            for rect in rects:
                x1, y1, x2, y2 = (
                    float(rect[0]),
                    float(rect[1]),
                    float(rect[2]),
                    float(rect[3]),
                )
                points = [[x1, y1], [x2, y2]]
                labelme_json["shapes"].append(
                    build_shape(label_cls, points, "rectangle")
                )

    for spot in spots:
        cls = spot["cls"]
        label_cls = name_mapping_spot[cls]
        rects = spot.get("rects", None)
        if rects:
            for rect in rects:
                x1, y1, x2, y2 = (
                    float(rect[0]),
                    float(rect[1]),
                    float(rect[2]),
                    float(rect[3]),
                )
                points = [[x1, y1], [x2, y2]]
                labelme_json["shapes"].append(
                    build_shape(label_cls, points, "rectangle")
                )

    json_name_path = os.path.join(output_dir, base_name.split(".")[0] + ".json")
    # 写入JSON文件
    fd = open(json_name_path, "w")
    json.dump(labelme_json, fd, indent=2)
    fd.close()
    # 输出保存信息
    print("save json={}".format(json_name_path))


if __name__ == "__main__":
    face = FaceAnalysis()

    dataset_dir = input("dataset_dir:")
    for json_path in tqdm(
        scan_json_in_folder(dataset_dir, extensions=[".json"]),
        desc="AutoMatic Processing json",
    ):
        read_image_write_base64(json_path, "datasets")
    # for image_path in tqdm(
    #     scan_images_in_folder(dataset_dir), desc="AutoMatic Processing images"
    # ):
    #     file_path = upload_file_to_oss(image_path)
    #     convert_txt_to_labelme_json(face.get_face_analysis(file_path), image_path)
    # image_path = "https://cdn.hawcat.cn/test_img/42d.jpg"
    # print(face.get_face_analysis(image_path))
