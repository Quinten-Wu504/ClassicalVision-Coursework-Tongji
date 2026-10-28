import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat

from EdgeDetection.canny import CannyEdgeDetector
from EdgeDetection.sobel import SobelEdgeDetector
from EdgeDetection.prewitt import PrewittEdgeDetector
from EdgeDetection.laplacian import LaplacianEdgeDetector
from EdgeDetection.edgeEval import EdgeEvaluator


def load_bsds500_dataset(image_dir, gt_dir, max_images=None):
    image_files = sorted(os.listdir(image_dir))[:max_images]
    image_list = []
    gt_list_per_image = []
    shape_list = []

    for fname in tqdm(image_files, desc='Loading dataset'):
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        image_list.append(img)
        shape_list.append(img.shape)

        name = os.path.splitext(fname)[0]
        mat_path = os.path.join(gt_dir, f"{name}.mat")
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"GT .mat file {mat_path} not found")

        mat = loadmat(mat_path)
        gt_edges = []
        for ann in mat['groundTruth'][0]:
            boundary = ann['Boundaries'][0][0]  # shape: (H, W)
            gt_edges.append((boundary > 0).astype(np.uint8))
        gt_list_per_image.append(gt_edges)

    return image_list, gt_list_per_image, shape_list, image_files


def apply_edge_detection(image, method):
    """
    Apply edge detection to an input image using the specified method.

    Parameters:
        image: Input image (NumPy array, RGB or grayscale)
        method: Edge detection method ('canny', 'sobel', 'laplacian', 'prewitt')

    Returns:
        Edge-detected image (NumPy array, uint8)
    """
    # Convert image to grayscale if it is RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'canny':
        detector = CannyEdgeDetector(kernel_size=5, sigma=1.0)
        return detector.canny_edge_detection(image)
    elif method == 'sobel':
        detector = SobelEdgeDetector(kernel_size=3, sigma=1.0)
        return detector.sobel_edge_detection(image)
    elif method == 'laplacian':
        detector = LaplacianEdgeDetector(kernel_size=3, sigma=1.0)
        return detector.laplacian_edge_detection(image, ksize=1)
    elif method == 'prewitt':
        detector = PrewittEdgeDetector(kernel_size=3, sigma=1.0)
        return detector.prewitt_edge_detection(image)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == '__main__':
    image_dir = '.\\Dataset\\BSDS500\\data\\images\\test'
    gt_dir = '.\\Dataset\\BSDS500\\data\\groundTruth\\test'
    methods = ['canny', 'sobel', 'laplacian', 'prewitt']
    max_images = None

    # 修改 load_bsds500_dataset 以返回 image_files
    images, gt_all, shape_list, image_files = load_bsds500_dataset(image_dir, gt_dir, max_images=max_images)

    all_results = []

    for method in methods:
        print(f'\n--- Evaluating {method.upper()} ---')
        edge_maps = []
        output_dir = f'.\\EdgeDetection\\outputs\\BSDS500\\{method}'
        os.makedirs(output_dir, exist_ok=True)

        for img, fname in tqdm(zip(images, image_files), total=len(images), desc=f'Applying {method}'):
            # 应用边缘检测
            edge_map = apply_edge_detection(img, method)
            edge_maps.append(edge_map)

            # 转换为二值图
            if method == 'canny':
                # Canny 已经是二值图（0 或 255），无需额外处理
                binary_edge = edge_map
            else:
                # 对 Sobel、Laplacian、Prewitt 使用 Otsu 阈值进行二值化
                _, binary_edge = cv2.threshold(edge_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 保存二值化边缘图
            output_path = os.path.join(output_dir, fname)
            cv2.imwrite(output_path, binary_edge)

        # 评估边缘检测结果
        evaluator = EdgeEvaluator(max_dist=0.0075, fast_mode=False)
        results = evaluator.evaluate_dataset(edge_maps, gt_all, shape_list)

        print(f"{method.upper()}:")
        print(f"  F1         => OIS = {results['ois_f1']:.4f}, ODS = {results['ods_f1']:.4f}")
        print(f"  Precision  => OIS = {results['ois_precision']:.4f}, ODS = {results['ods_precision']:.4f}")
        print(f"  Recall     => OIS = {results['ois_recall']:.4f}, ODS = {results['ods_recall']:.4f}")

        all_results.append({
            "Method": method.upper(),
            "OIS_F1": results['ois_f1'],
            "ODS_F1": results['ods_f1'],
            "OIS_Precision": results['ois_precision'],
            "ODS_Precision": results['ods_precision'],
            "OIS_Recall": results['ois_recall'],
            "ODS_Recall": results['ods_recall'],
        })

    # 保存评估结果为 CSV
    df = pd.DataFrame(all_results)
    os.makedirs('.\\EdgeDetection', exist_ok=True)
    df.to_csv('.\\EdgeDetection\\outputs\\edge_extract_quality_results.csv', index=False)
    print("\n评估结果已保存至 '.\\EdgeDetection\\outputs\\edge_extract_quality_results.csv'")
