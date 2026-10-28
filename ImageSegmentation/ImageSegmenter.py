import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from EdgeDetection.canny import CannyEdgeDetector
# 假设其他边缘检测器模块存在
from EdgeDetection.sobel import SobelEdgeDetector
from EdgeDetection.prewitt import PrewittEdgeDetector
from EdgeDetection.laplacian import LaplacianEdgeDetector


class ImageSegmentation:
    """用于边缘图分割的类，集成分水岭法和区域增长法"""

    def __init__(self, image, edge_map):
        """
        初始化，输入边缘图

        参数:
            edge_map: 输入边缘图（numpy数组，灰度或二值化格式，边缘白色255，背景黑色0）
        """
        self.image_gray = image
        if len(image.shape) == 3:  # 如果是RGB格式，转换为灰度
            self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.edge_map = edge_map.copy()
        if len(edge_map.shape) == 3:  # 如果是RGB格式，转换为灰度
            self.edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.edge_map.shape[:2]

    def watershed_segmentation(self, kernel_size=3):
        """
        对边缘图应用分水岭算法分割，返回二值掩码

        参数:
            kernel_size (int): 形态学操作的核大小，默认为3

        返回:
            二值掩码（0 和 255），分割区域为255，背景为0
        """
        # Step 1: 使用 Otsu 阈值处理边缘图（或者换成 self.image_gray）
        _, thresh = cv2.threshold(self.edge_map, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Step 2: 背景区域（dilate 后可能连通）
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Step 3: 前景区域 via 距离变换
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Step 4: 未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Step 5: 连接组件作为前景标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # 背景设为1，前景从2开始
        markers[unknown > 0] = 0  # 未知区域设为0

        # Step 6: 分水岭处理 —— 使用灰度图比边缘图效果更好
        color_img = cv2.cvtColor(self.image_gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color_img, markers)

        # Step 7: 构建掩码 —— 正确理解 marker 值
        # markers == -1 是分水岭线
        # markers > 1 是前景（我们关心的目标区域）
        mask = np.zeros((self.height, self.width), np.uint8)
        mask[markers > 1] = 255

        mask = 255 - mask

        return mask

    def region_growing_segmentation(self, seed_point=None, threshold=10, kernel_size=3):
        """
        对边缘图应用区域增长算法分割，返回二值掩码

        参数:
            seed_point (tuple): 种子点坐标 (x, y)，若为None则选择背景区域的中心
            threshold (int): 区域增长的强度差阈值，默认为10

        返回:
            二值掩码（0 和 255），分割区域为255，背景为0
        """
        # 形态学预处理：开运算（先侵蚀后膨胀）去除噪声
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_edge_map = cv2.morphologyEx(self.edge_map, cv2.MORPH_CLOSE, kernel, iterations=1)

        mask = np.zeros((self.height, self.width), np.uint8)
        visited = np.zeros((self.height, self.width), np.uint8)

        # 如果未提供种子点，寻找背景区域（强度0）的中心
        if seed_point is None:
            background_pixels = np.where(processed_edge_map == 0)
            if len(background_pixels[0]) == 0:
                raise ValueError("边缘图中无背景像素（强度0），请指定种子点")
            y_center = int(np.mean(background_pixels[0]))
            x_center = int(np.mean(background_pixels[1]))
            seed_point = (x_center, y_center)

        # 如果种子点位于边缘（强度255），搜索附近非边缘点
        if processed_edge_map[seed_point[1], seed_point[0]] > 127:
            queue = deque([seed_point])
            search_visited = np.zeros((self.height, self.width), np.uint8)
            search_visited[seed_point[1], seed_point[0]] = 1
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                          (0, 1), (1, -1), (1, 0), (1, 1)]

            found = False
            while queue:
                current = queue.popleft()
                x, y = current

                # 检查邻域像素
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and not search_visited[ny, nx]:
                        search_visited[ny, nx] = 1
                        if processed_edge_map[ny, nx] <= 127:
                            seed_point = (nx, ny)
                            found = True
                            break
                        queue.append((nx, ny))
                if found:
                    break

            if not found:
                raise ValueError(f"种子点{seed_point}附近无非边缘像素（强度0）")

        # 初始化种子点
        queue = [seed_point]
        seed_value = processed_edge_map[seed_point[1], seed_point[0]]
        mask[seed_point[1], seed_point[0]] = 255
        visited[seed_point[1], seed_point[0]] = 1

        # 定义8连通性
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]

        # 区域增长
        while queue:
            current = queue.pop(0)
            x, y = current

            # 检查邻域像素
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not visited[ny, nx]:
                    if processed_edge_map[ny, nx] > 127:
                        continue  # 不跨越边缘
                    intensity_diff = abs(int(processed_edge_map[ny, nx]) - seed_value)
                    if intensity_diff <= threshold:
                        queue.append((nx, ny))
                        mask[ny, nx] = 255
                        visited[ny, nx] = 1

        return mask


if __name__ == '__main__':
    # 初始边缘检测器
    detector = CannyEdgeDetector(kernel_size=5, sigma=1)
    detector = LaplacianEdgeDetector()

    # 加载图像
    # image_path = '..\\Dataset\\lena.png'
    image_path = '..\\Dataset\\Cats\\images\\cat_grumpy.jpg'
    if not cv2.os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    edge_map = detector.process_image(image_path)
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(image_path)

    segmenter = ImageSegmentation(image=gray_img, edge_map=edge_map)

    # 应用分水岭分割
    watershed_result = segmenter.watershed_segmentation(kernel_size=3)

    # 应用区域增长分割
    region_growing_result = segmenter.region_growing_segmentation(threshold=20)

    # 区域增长
    def show_result(title, edge_map, gray_img, result_rg, result_ws):
        plt.figure(figsize=(12, 4))
        plt.suptitle(title)
        plt.subplot(1, 4, 1)
        plt.imshow(edge_map, cmap='gray')
        plt.title("Edge Map")
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(gray_img, cmap='gray')
        plt.title("Gray Image")
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(result_rg, cmap='gray')
        plt.title("Region Growing")
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(result_ws, cmap='gray')
        plt.title("Watershed")
        plt.axis('off')
        plt.show()

    # 可视化对比
    show_result("Segmentation Results", edge_map, gray_img, region_growing_result, watershed_result)
