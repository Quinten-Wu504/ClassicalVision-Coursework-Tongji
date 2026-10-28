import cv2
import torch
import numpy as np
import torch.nn.functional as F


class PrewittEdgeDetector:
    """Prewitt 边缘检测器类，整合高斯平滑、卷积和 Prewitt 边缘检测功能"""

    def __init__(self, kernel_size=3, sigma=1.0, device=None):
        """
        初始化 Prewitt 边缘检测器

        参数:
            kernel_size (int): 高斯核大小，默认为 3
            sigma (float): 高斯核标准差，默认为 1.0
            device (str): 计算设备，默认为 None（自动选择 CUDA 或 CPU）
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

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

    def _gaussian_blur(self, image):
        """对图像应用高斯平滑。"""
        kernel = self._gaussian_kernel()
        return self._convolution(image, kernel)

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

    def prewitt_edge_detection(self, image, filter=torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])):
        """
        应用 Prewitt 边缘检测（PyTorch 实现）。

        参数:
            image: 输入图像（NumPy 数组，RGB 或灰度）
            filter: Prewitt 核，默认为水平核

        返回:
            gradient_magnitude: 灰度梯度图（0-255，NumPy 数组）
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将 NumPy 数组转换为 PyTorch 张量，并移动到设备
        image = torch.from_numpy(image).float().to(self.device)

        # 高斯平滑
        image = self._gaussian_blur(image)

        # 计算水平和垂直梯度
        filter = filter.to(self.device)
        grad_x = self._convolution(image, filter)  # 水平核
        grad_y = self._convolution(image, torch.flip(filter.t(), dims=[0, 1]))  # 垂直核

        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        # 归一化到 0-255
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).byte()

        # 转换回 NumPy 数组
        gradient_magnitude = gradient_magnitude.cpu().numpy()
        return gradient_magnitude

    def process_image(self, image_path):
        """处理图像文件并返回边缘检测结果"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return self.prewitt_edge_detection(image)


if __name__ == '__main__':
    detector = PrewittEdgeDetector(kernel_size=3, sigma=1.0)
    edges = detector.process_image('../Dataset/lena.png')
    cv2.imwrite("../Dataset/prewitt_edges.png", edges)
    img = cv2.imread('../Dataset/lena.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    # prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    # edges = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
    # cv2.imwrite("../Dataset/cv2_prewitt_edges.png", edges)
