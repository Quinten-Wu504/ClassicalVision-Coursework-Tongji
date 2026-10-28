import cv2
import torch
import numpy as np
import torch.nn.functional as F


class LaplacianEdgeDetector:
    """Laplacian 边缘检测器类，整合卷积和 Laplacian 边缘检测功能"""

    def __init__(self, kernel_size=3, sigma=1.0):
        """
        初始化 Laplacian 边缘检测器

        参数:
            kernel_size (int): 高斯核大小，默认为 3
            sigma (float): 高斯核标准差，默认为 1.0
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernels = {
            1: torch.tensor([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=torch.float32, device=self.device),
            3: torch.tensor([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]], dtype=torch.float32, device=self.device)
        }

    def _dnorm(self, x, mu, sd):
        """计算高斯分布的概率密度值"""
        return 1 / (np.sqrt(2 * np.pi) * sd) * torch.exp(-((x - mu) / sd) ** 2 / 2)

    def _gaussian_kernel(self):
        """生成 2D 高斯核"""
        size = self.kernel_size
        kernel_1D = torch.linspace(-(size // 2), size // 2, size, device=self.device)
        kernel_1D = self._dnorm(kernel_1D, 0, self.sigma)
        kernel_2D = torch.outer(kernel_1D, kernel_1D)
        kernel_2D /= kernel_2D.sum()
        kernel_2D = kernel_2D.view(1, 1, size, size)
        return kernel_2D

    def _convolution(self, image, kernel):
        """对图像应用卷积操作"""
        image = image.unsqueeze(0).unsqueeze(0) if len(image.shape) == 2 else image
        kernel = kernel.view(1, 1, kernel.shape[-2], kernel.shape[-1]).to(self.device, dtype=torch.float32)
        padding_height = kernel.shape[-2] // 2
        padding_width = kernel.shape[-1] // 2
        padding = (padding_width, padding_width, padding_height, padding_height)
        image_padded = F.pad(image, padding, mode='reflect')
        output = F.conv2d(image_padded, kernel)
        return output.squeeze(0).squeeze(0)

    def _gaussian_blur(self, image):
        """对图像应用高斯平滑"""
        kernel = self._gaussian_kernel()
        return self._convolution(image, kernel)

    def laplacian_edge_detection(self, image, ksize=1):
        """
        应用 Laplacian 边缘检测，输出 uint8 格式边缘图

        参数:
            image: 输入图像（RGB 或灰度）
            ksize: 拉普拉斯核大小，默认为 1

        返回:
            edge_image: 边缘检测结果（uint8，0-255）
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = torch.from_numpy(image).float().to(self.device)
        image = self._gaussian_blur(image)
        kernel = self.kernels[ksize]
        result = self._convolution(image, kernel)

        # 取绝对值并归一化到 0-255，转换为 uint8
        result = torch.abs(result)
        if result.max() > 0:
            edge_image = (result / result.max() * 255).byte()
        else:
            edge_image = torch.zeros_like(result, dtype=torch.uint8)

        return edge_image.detach().cpu().numpy()

    def process_image(self, image_path, ksize=1):
        """处理图像文件并返回边缘检测结果（uint8）"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return self.laplacian_edge_detection(image, ksize)


if __name__ == '__main__':
    detector = LaplacianEdgeDetector(kernel_size=3, sigma=1.0)
    result = detector.process_image('../Dataset/lena.png', ksize=3)

    # 应用 OpenCV 的拉普拉斯边缘检测进行对比
    img = cv2.imread('../Dataset/lena.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2_result = cv2.Laplacian(gray, ddepth=cv2.CV_8U, ksize=3)

    # 保存结果
    cv2.imwrite("../Dataset/custom_laplacian.png", result)
    cv2.imwrite("../Dataset/cv2_laplacian.png", cv2_result)