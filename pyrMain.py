import os
import cv2
import numpy as np
from ImagePyramid.pyrEval import PyramidEvaluator


def load_images_from_folder(folder, max_images=24):
    """
    从指定文件夹加载最多 max_images 张灰度图像（按文件顺序）。

    返回:
        images: 灰度图像列表（float32，归一化到 [0, 1]，形状 (H, W, 1)）。
        image_names: 图像文件名列表。
    """
    images = []
    image_names = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = img.astype(np.float32) / 255.0  # 归一化
            img = img[:, :, np.newaxis]  # 转换为 (H, W, 1)
            images.append(img)
            image_names.append(filename)
        if len(images) >= max_images:
            break
    return images, image_names


if __name__ == "__main__":
    image_folder = ".\\Dataset\\KodakLosslessTrue\\"
    images, image_names = load_images_from_folder(image_folder, max_images=24)

    evaluator = PyramidEvaluator(max_level=7)
    result_df = evaluator.evaluate(images, image_names)
    print(result_df)

    os.makedirs(".\\ImagePyramid", exist_ok=True)
    result_df.to_csv(".\\ImagePyramid\\outputs\\pyramid_quality_results.csv", index=False)
    print("\n评估结果已保存至 '.\\ImagePyramid\\outputs\\pyramid_quality_results.csv'")
