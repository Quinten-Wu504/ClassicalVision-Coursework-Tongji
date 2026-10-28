import cv2
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from ImageSegmentation.ImageSegmenter import ImageSegmentation
from EdgeDetection.canny import CannyEdgeDetector
from EdgeDetection.sobel import SobelEdgeDetector
from EdgeDetection.prewitt import PrewittEdgeDetector
from EdgeDetection.laplacian import LaplacianEdgeDetector


def compute_metrics(pred_mask, gt_mask):
    """
    计算分割指标：Dice 系数、IoU 和精确度

    参数:
        pred_mask (np.ndarray): 预测掩码（0/255）
        gt_mask (np.ndarray): Ground truth 掩码（0/255）

    返回:
        tuple: (dice, iou, accuracy)
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)

    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0
    iou = intersection / union if union > 0 else 0

    tp = intersection
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return dice, iou, accuracy


if __name__ == "__main__":
    """
    测试 ImageSegmentation 类的分水岭法和区域增长法，结合四种边缘检测算法，生成平均指标表格
    使用两个进度条：一个用于分水岭法，一个用于区域增长法
    """
    # 路径设置
    image_dir = "./Dataset/Cats/images/"
    gt_dir = "./Dataset/Cats/groundTruth/"
    output_base_dir = "./ImageSegmentation/outputs/"

    # 确保输出基础目录存在
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 分割算法和边缘检测器
    modes = ["watershed", "region_grow"]
    detectors = {
        "canny": CannyEdgeDetector(kernel_size=5, sigma=1.0),
        "sobel": SobelEdgeDetector(),
        "prewitt": PrewittEdgeDetector(),
        "laplacian": LaplacianEdgeDetector()
    }

    # 存储指标
    metrics = []

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    total_tasks = len(image_files) * len(detectors)  # 每种模式的任务数

    # 为每种分割模式设置一个进度条
    for mode in modes:
        desc = f"Processing {mode.replace('_', ' ').title()}"
        with tqdm(total=total_tasks, desc=desc, leave=True) as pbar:
            for image_name in image_files:
                image_path = os.path.join(image_dir, image_name)
                gt_name = os.path.splitext(image_name)[0] + ".png"
                gt_path = os.path.join(gt_dir, gt_name)

                # 加载图像和 ground truth 掩码
                color_img = cv2.imread(image_path)
                if color_img is None:
                    print(f"无法加载图像: {image_path}")
                    pbar.update(len(detectors))  # 跳过所有检测器
                    continue
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is None:
                    print(f"无法加载 ground truth 掩码: {gt_path}")
                    pbar.update(len(detectors))
                    continue

                # 对每种边缘检测器进行测试
                for detector_name, detector in detectors.items():
                    # 获取边缘图并转换为 uint8
                    edge_map = detector.process_image(image_path)
                    if edge_map.dtype != np.uint8:
                        edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    # 初始化 ImageSegmentation
                    segmenter = ImageSegmentation(image=gray_img, edge_map=edge_map)

                    # 应用分割
                    try:
                        if mode == "watershed":
                            mask = segmenter.watershed_segmentation(kernel_size=3)
                        else:  # region_grow
                            mask = segmenter.region_growing_segmentation(threshold=20)
                    except ValueError as e:
                        print(f"分割失败 ({mode}, {detector_name}, {image_name}): {e}")
                        metrics.append({
                            "Image": image_name,
                            "Mode": mode,
                            "Detector": detector_name,
                            "Dice": 0.0,
                            "IoU": 0.0,
                            "Accuracy": 0.0
                        })
                        pbar.update(1)
                        continue

                    # 保存掩码
                    output_dir = os.path.join(output_base_dir, mode, os.path.splitext(image_name)[0])
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    cv2.imwrite(f"{output_dir}/{detector_name}_mask.png", mask)

                    # 计算指标
                    dice, iou, accuracy = compute_metrics(mask, gt_mask)

                    # 存储指标
                    metrics.append({
                        "Image": image_name,
                        "Mode": mode,
                        "Detector": detector_name,
                        "Dice": dice,
                        "IoU": iou,
                        "Accuracy": accuracy
                    })

                    pbar.update(1)

    # 计算平均指标并生成表格
    metrics_df = pd.DataFrame(metrics)
    for mode in modes:
        mode_metrics = metrics_df[metrics_df["Mode"] == mode]
        avg_metrics = mode_metrics.groupby("Detector")[["Dice", "IoU", "Accuracy"]].mean().reset_index()
        avg_metrics = avg_metrics.round(4)  # 保留4位小数

        # 打印表格
        print(f"\nAverage Metrics for {mode.replace('_', ' ').title()}:")
        print(tabulate(avg_metrics, headers="keys", tablefmt="grid"))

        # 保存为 CSV
        csv_path = os.path.join(output_base_dir, f"{mode}_metrics.csv")
        avg_metrics.to_csv(csv_path, index=False)
        print(f"Saved {mode} metrics to {csv_path}")
