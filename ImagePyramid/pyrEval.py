import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ImagePyramid.ImagePyramid import ImagePyramid


class PyramidEvaluator:
    def __init__(self, max_level=5):
        self.max_level = max_level
        self.pyramid = ImagePyramid(max_levels=max_level)

    def evaluate(self, images, image_names, output_base_dir=".\\ImagePyramid\\outputs\\KodakLosslessTrue"):
        """
        主评估函数：评估使用不同金字塔层数的重建质量，并保存分解和重构结果的灰度图。

        参数:
            images: 灰度图像列表，每个图像为 (H, W, 1) 的 float32 数组（归一化到 [0, 1]）。
            image_names: 图像文件名列表，用于保存输出。
            output_base_dir: 输出基础目录，分解和重构结果分别保存到 decompose 和 recompose 子目录。

        返回:
            pandas DataFrame，包含每种层数的平均 PSNR 和 SSIM。
        """
        results = []

        for use_level in range(1, self.max_level + 1):
            psnr_list = []
            ssim_list = []

            for img, img_name in tqdm(zip(images, image_names), total=len(images), desc=f"Processing with {use_level} layers"):
                # 构建拉普拉斯金字塔
                lap_pyr = self.pyramid.build_laplacian_pyramid(img)

                # 保存分解结果的灰度图
                decompose_dir = os.path.join(output_base_dir, "decompose", os.path.splitext(img_name)[0])
                os.makedirs(decompose_dir, exist_ok=True)
                for lvl, lap_layer in enumerate(lap_pyr[:use_level]):
                    lap_layer = np.clip(lap_layer, 0, 1)  # 确保范围在 [0, 1]
                    gray = (lap_layer[:, :, 0] * 255).astype(np.uint8)  # 提取单通道，转换为 uint8
                    cv2.imwrite(os.path.join(decompose_dir, f"{lvl}.png"), gray)

                # 截断金字塔，只保留 use_level 层
                truncated_pyr = lap_pyr[:use_level] + [
                    np.zeros_like(lap) for lap in lap_pyr[use_level:]
                ]

                # 重建图像
                recon = self.pyramid.reconstruct_from_laplacian(truncated_pyr)
                recon = np.clip(recon, 0, 1)

                # 保存重构结果的灰度图
                recompose_dir = os.path.join(output_base_dir, "recompose", os.path.splitext(img_name)[0])
                os.makedirs(recompose_dir, exist_ok=True)
                gray_recon = (recon[:, :, 0] * 255).astype(np.uint8)  # 提取单通道，转换为 uint8
                cv2.imwrite(os.path.join(recompose_dir, f"{use_level}.png"), gray_recon)

                # 计算质量指标
                img_single = img[:, :, 0]  # 提取单通道
                recon_single = recon[:, :, 0]
                psnr = peak_signal_noise_ratio(img_single, recon_single, data_range=1.0)
                ssim = structural_similarity(img_single, recon_single, multichannel=False, data_range=1.0)

                psnr_list.append(psnr)
                ssim_list.append(ssim)

            results.append({
                "Used_Layers": use_level,
                "Avg_PSNR": np.mean(psnr_list),
                "Avg_SSIM": np.mean(ssim_list)
            })

        return pd.DataFrame(results)
