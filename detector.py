"""
CODE FROM business code
"""

import os
import uuid

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont


class Detector:
    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        base_options = python.BaseOptions(
            model_asset_path="models/face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        logger.success("Face Detector loaded")

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in face_landmarks
                ]
            )

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        cv2.imwrite(
            os.path.join(self.output_dir, "img_with_landmarks.jpg"),
            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
        )

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [
            face_blendshapes_category.category_name
            for face_blendshapes_category in face_blendshapes
        ]
        face_blendshapes_scores = [
            face_blendshapes_category.score
            for face_blendshapes_category in face_blendshapes
        ]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(
            face_blendshapes_ranks,
            face_blendshapes_scores,
            label=[str(x) for x in face_blendshapes_ranks],
        )
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(
                patch.get_x() + patch.get_width(),
                patch.get_y(),
                f"{score:.4f}",
                va="top",
            )

        ax.set_xlabel("Score")
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "face_blendshapes_scores.png"))

    def plot_face_threedimension_matrix(self, detection_result):
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
        else:
            logger.warning("No face detected")
            return

        # 创建3D图形并绘制关键点
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # 绘制关键点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="b", marker="o")

        # 设置坐标轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 设置视角
        ax.view_init(elev=20, azim=45)

        # 绘制连接线（这里使用 MediaPipe 的 FACE_CONNECTIONS）
        connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        for connection in connections:
            start = points[connection[0]]
            end = points[connection[1]]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                c="r",
                alpha=0.5,
            )

        plt.show()
        plt.savefig(os.path.join(self.output_dir, "face_3d_matrix.png"))

    @staticmethod
    def crop_and_normalize_face(
        image, detection_result, is_resize=False, target_size=(640, 640), is_show=False
    ):
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            h, w = image.shape[:2]
            landmarks = np.array(
                [(landmark.x * w, landmark.y * h) for landmark in face_landmarks]
            )

            # 提取 landmarks 的 x 和 y 坐标
            x_coords = [int(point[0]) for point in landmarks]
            y_coords = [int(point[1]) for point in landmarks]

            # 找到最小和最大 x, y 值，构成边界框
            x_min = max(min(x_coords), 0)
            y_min = max(min(y_coords), 0)
            x_max = min(max(x_coords), image.shape[1])
            y_max = min(max(y_coords), image.shape[0])

            # 扩展边界框，加大 30%
            padding = 0.3
            width = x_max - x_min
            height = y_max - y_min
            x_min = max(int(x_min - padding * width), 0)
            y_min = max(int(y_min - padding * height), 0)
            x_max = min(int(x_max + padding * width), image.shape[1])
            y_max = min(int(y_max + padding * height), image.shape[0])

            # 裁剪图像
            cropped_image = image[y_min:y_max, x_min:x_max]
            if is_resize:
                # 调整裁剪后的图像大小
                resized_image = cv2.resize(cropped_image, target_size)

                # 将像素值归一化到 [0, 1]
                normalized_image = resized_image.astype(np.float32) / 255.0

                if is_show:
                    fig = plt.figure()
                    a = fig.add_subplot(1, 2, 1)
                    plt.imshow(image)  # 此处的img为上面直接读取的img
                    a.set_title("Before")

                    a = fig.add_subplot(1, 2, 2)
                    a.set_title("After")
                    plt.imshow(normalized_image)

                    plt.show()  # show出两幅图，before为原生图像，after为resize后的图像

                save_image = (normalized_image * 255).astype(np.uint8)

                return save_image
            else:
                return cropped_image

        else:
            logger.warning("No face detected")
            return None

    def get_rgb_color(self, image, mask):
        mean_color = cv2.mean(image, mask=mask)  # Returns (B, G, R, alpha)
        return mean_color[:3]  # 返回 RGB 三个通道的平均值


if __name__ == "__main__":
    import glob

    from tqdm import tqdm

    face = Detector()

    jpg_list = glob.glob("C:/Users/hawcat/Desktop/datasets/face/*.jpg")
    png_list = glob.glob("C:/Users/hawcat/Desktop/datasets/face/*.png")

    out_dir = "C:/Users/hawcat/Desktop/datasets/face_normalize"

    com_list = jpg_list + png_list

    for path in tqdm(com_list, desc="Nomalizing..."):
        basename = os.path.basename(path)
        img_cv = cv2.imread(path)
        try:
            img_mp = mp.Image.create_from_file(path)
            result = face.detector.detect(img_mp)

            cv2.imwrite(
                os.path.join(out_dir, basename),
                face.crop_and_normalize_face(img_cv, result, is_resize=True),
            )

        except Exception as e:
            logger.warning(basename)
            logger.info(e)
