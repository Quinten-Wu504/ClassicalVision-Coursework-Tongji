import cv2
import torch
import numpy as np
import torch.nn.functional as F


class CannyEdgeDetector:
    """Canny 边缘检测器类，从头实现，与 cv2.Canny 的实现对齐。"""

    def __init__(self, kernel_size=5, sigma=2.0):
        """
        初始化 Canny 边缘检测器。

        参数:
            kernel_size (int): 高斯核大小，默认为 5（与 cv2.Canny 一致）。
            sigma (float): 高斯核标准差，默认为 2.0（增强平滑）。
            L2gradient (bool): 是否使用 L2 范数计算梯度，默认为 False（与 cv2.Canny 一致）。
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _dnorm(self, x, mu, sd):
        """计算高斯分布的概率密度值"""
        return 1 / (np.sqrt(2 * np.pi) * sd) * torch.exp(-((x - mu) / sd) ** 2 / 2)

    def _gaussian_kernel(self):
        """生成 2D 高斯核"""
        size = self.kernel_size
        kernel_1D = torch.linspace(-(size // 2), size // 2, size, device=self.device)
        kernel_1D = self._dnorm(kernel_1D, 0, self.sigma)
        kernel_2D = torch.outer(kernel_1D, kernel_1D)  # 外积生成 2D 核
        kernel_2D /= kernel_2D.sum()  # 归一化到总和 1
        # 转换为卷积核格式 [out_channels, in_channels, height, width]
        kernel_2D = kernel_2D.view(1, 1, size, size)
        return kernel_2D

    def _convolution(self, image, kernel, average=False):
        """对图像应用卷积操作"""
        # 确保输入图像是 4D 张量 [batch, channels, height, width]
        image = image.unsqueeze(0).unsqueeze(0) if len(image.shape) == 2 else image
        # 确保 kernel 是 4D 张量 [out_channels, in_channels, height, width]
        kernel = kernel.view(1, 1, kernel.shape[-2], kernel.shape[-1]).to(self.device, dtype=torch.float32)

        # 计算填充值，分别处理高度和宽度
        padding_height = kernel.shape[-2] // 2
        padding_width = kernel.shape[-1] // 2
        padding = (padding_width, padding_width, padding_height, padding_height)  # (left, right, top, bottom)

        # 使用反射填充（reflect）而不是默认的零填充
        image_padded = F.pad(image, padding, mode='reflect')

        # 应用卷积
        output = F.conv2d(image_padded, kernel)
        return output.squeeze(0).squeeze(0)  # 移除批次和通道维度

    def _non_maximum_suppression(self, grad_magnitude, grad_angle):
        """非极大值抑制，使用 8 个角度区间和双线性插值优化，恢复严格抑制。"""
        suppressed = np.zeros_like(grad_magnitude)
        angle = grad_angle % 180
        rows, cols = grad_magnitude.shape

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if grad_magnitude[i, j] == 0:
                    continue
                ang = angle[i, j]
                if ang < 0:
                    ang += 180

                # 8 个角度区间，均匀分布
                if 0 <= ang < 22.5:
                    q = grad_magnitude[i, j+1]
                    r = grad_magnitude[i, j-1]
                elif 22.5 <= ang < 67.5:
                    weight = (ang - 22.5) / 45.0
                    q = (1 - weight) * grad_magnitude[i+1, j] + weight * grad_magnitude[i+1, j-1]
                    r = (1 - weight) * grad_magnitude[i-1, j] + weight * grad_magnitude[i-1, j+1]
                elif 67.5 <= ang < 112.5:
                    q = grad_magnitude[i+1, j]
                    r = grad_magnitude[i-1, j]
                elif 112.5 <= ang < 157.5:
                    weight = (ang - 112.5) / 45.0
                    q = (1 - weight) * grad_magnitude[i-1, j] + weight * grad_magnitude[i-1, j-1]
                    r = (1 - weight) * grad_magnitude[i+1, j] + weight * grad_magnitude[i+1, j+1]
                else:  # 157.5 <= ang <= 180
                    q = grad_magnitude[i, j+1]
                    r = grad_magnitude[i, j-1]

                if grad_magnitude[i, j] >= max(q, r):
                    suppressed[i, j] = grad_magnitude[i, j]

        return suppressed

    def canny_edge_detection(self, image):
        """
        应用 Canny 边缘检测，与 cv2.Canny 的实现对齐。

        参数:
            image: 输入图像（RGB 或灰度）。

        返回:
            edges: 二值边缘图（0 或 255）。
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将 NumPy 数组转换为 PyTorch 张量，并移动到设备
        image = torch.from_numpy(image).float().to(self.device)

        # 1. 高斯平滑
        gaussian_kernel = self._gaussian_kernel()
        smoothed = self._convolution(image, gaussian_kernel)

        # 2. 梯度计算
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device)
        grad_x = self._convolution(smoothed, sobel_x)
        grad_y = self._convolution(smoothed, sobel_y)

        grad_magnitude = torch.abs(grad_x) + torch.abs(grad_y)

        # 调整梯度幅值
        grad_magnitude = torch.clamp(grad_magnitude, 0, 255).byte() * 2.5  # 降低缩放因子

        grad_angle = torch.atan2(grad_y, grad_x) * 180 / np.pi

        # 3. 非极大值抑制
        suppressed = self._non_maximum_suppression(grad_magnitude.cpu().numpy(), grad_angle.cpu().numpy())

        # 转换为 uint8 格式
        edges = np.clip(suppressed, 0, 255).astype(np.uint8)

        return edges

    def process_image(self, image_path):
        """处理图像文件并返回边缘检测结果。"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return self.canny_edge_detection(image)


if __name__ == '__main__':
    detector = CannyEdgeDetector(kernel_size=5, sigma=1.0)
    edges = detector.process_image('../Dataset/lena.png')
    cv2.imwrite("../Dataset/canny_edges.png", edges)
    img = cv2.imread('../Dataset/lena.png')
    edges_cv2 = cv2.Canny(img, 100, 200)
    cv2.imwrite('../Dataset/canny_edges_cv2.png', edges_cv2)