import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class ImagePyramid:
    """
    图像金字塔类：实现高斯金字塔和拉普拉斯金字塔的构建与重建。
    支持输入单张灰度图像，格式为 [H,W,1]。
    """

    def __init__(self, max_levels=4, kernel=None, borderType=cv2.BORDER_REFLECT):
        """
        初始化金字塔构建参数。
        :param max_levels: 金字塔层数（包括原图层）。
        :param kernel: 高斯卷积核，默认为5x5 binomial核 [1,4,6,4,1] x [1,4,6,4,1] / 256。
        :param borderType: OpenCV 边界处理类型，默认为反射边界。
        """
        self.max_levels = max_levels
        self.border = borderType
        if kernel is None:
            k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
            kernel = np.outer(k, k)
            kernel = kernel / np.sum(kernel)
        self.kernel = kernel.astype(np.float32)

    def _prepare_image(self, image):
        """
        统一图像格式为 (H, W, 1)，专为灰度图设计。
        支持形状为 (H,W), (H,W,1)。
        返回: image_list，列表中的每个图像为 (H, W, 1)。
        """
        imgs = []
        shape = image.shape
        if image.ndim == 2:
            imgs = [image[:, :, np.newaxis]]  # (H, W) -> (H, W, 1)
        elif image.ndim == 3 and image.shape[2] == 1:
            imgs = [image]  # (H, W, 1)
        else:
            raise ValueError(f"Unsupported image shape: {shape}. Expected (H, W) or (H, W, 1).")
        return imgs

    def _gaussian_blur(self, img):
        """对单张灰度图像进行高斯模糊卷积，保留通道维度。"""
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # 确保输入为 (H, W, 1)
        blurred = cv2.filter2D(img, -1, self.kernel, borderType=self.border)
        if blurred.ndim == 2:
            blurred = blurred[:, :, np.newaxis]  # 确保输出为 (H, W, 1)
        return blurred

    def _pyr_down(self, img):
        """高斯平滑后下采样，输出尺寸为 ceil(H/2) x ceil(W/2)，保留通道维度。"""
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        blurred = self._gaussian_blur(img)
        return blurred[::2, ::2, :]  # (H/2, W/2, 1)

    def _pyr_up(self, img, target_shape):
        """
        上采样至目标形状并模糊，保留通道维度。
        :param img: 输入图像 (H, W, 1) 或 (H, W)。
        :param target_shape: 目标尺寸 (Ht, Wt)。
        :return: 上采样图像 (Ht, Wt, 1)。
        """
        Ht, Wt = target_shape
        h, w = img.shape[:2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        c = img.shape[2]
        up = np.zeros((2*h, 2*w, c), dtype=img.dtype)
        up[::2, ::2, :] = img
        up_blur = cv2.filter2D(up, -1, self.kernel, borderType=self.border)
        if up_blur.ndim == 2:
            up_blur = up_blur[:, :, np.newaxis]  # 确保输出为 (Ht, Wt, 1)
        return up_blur[:Ht, :Wt, :]

    def build_gaussian_pyramid(self, image):
        """
        构建高斯金字塔，返回每层图像列表。
        :param image: 输入灰度图像，格式为 [H,W,1].
        :return: 列表[g0, g1, ..., g_{L-1}]，每个元素为 (H, W, 1)。
        """
        imgs = self._prepare_image(image)
        pyramids = []
        for img in imgs:
            img = img.astype(np.float32)
            levels = [img]
            current = img
            for lvl in range(1, self.max_levels):
                if current.shape[0] < 2 or current.shape[1] < 2:
                    break
                current = self._pyr_down(current)
                levels.append(current)
            pyramids.append(levels)
        return pyramids[0] if len(pyramids) == 1 else pyramids

    def build_laplacian_pyramid(self, image):
        """
        构建拉普拉斯金字塔，返回每层灰度图像列表。
        最后一层为最低分辨率高斯图像。
        :param image: 输入灰度图像，格式为 [H,W,1].
        :return: 列表[l0, l1, ..., l_{L-1}]，每个元素为 (H, W, 1)。
        """
        imgs = self._prepare_image(image)
        lap_pyramids = []
        for img in imgs:
            img = img.astype(np.float32)
            gauss_levels = [img]
            current = img
            for lvl in range(1, self.max_levels):
                if current.shape[0] < 2 or current.shape[1] < 2:
                    break
                current = self._pyr_down(current)
                gauss_levels.append(current)
            lap_levels = []
            for i in range(len(gauss_levels)-1):
                G_i = gauss_levels[i]
                G_next = gauss_levels[i+1]
                expand = self._pyr_up(G_next, target_shape=G_i.shape[:2])
                lap = G_i - expand
                lap_levels.append(lap)
            lap_levels.append(gauss_levels[-1])
            lap_pyramids.append(lap_levels)
        return lap_pyramids[0] if len(lap_pyramids) == 1 else lap_pyramids

    def reconstruct_from_laplacian(self, lap_pyramid):
        """
        从拉普拉斯金字塔重建图像。
        :param lap_pyramid: 拉普拉斯金字塔列表。
        :return: 重建灰度图像 (H, W, 1)。
        """
        if isinstance(lap_pyramid, list) and isinstance(lap_pyramid[0], list):
            recon_images = []
            for lap_levels in lap_pyramid:
                recon = lap_levels[-1]
                for i in range(len(lap_levels)-2, -1, -1):
                    lap = lap_levels[i]
                    recon = self._pyr_up(recon, target_shape=lap.shape[:2])
                    recon = recon + lap
                recon_images.append(recon)
            return recon_images
        else:
            lap_levels = lap_pyramid
            recon = lap_levels[-1]
            for i in range(len(lap_levels)-2, -1, -1):
                lap = lap_levels[i]
                recon = self._pyr_up(recon, target_shape=lap.shape[:2])
                recon = recon + lap
            return recon


if __name__ == "__main__":
    img = np.random.randint(0, 256, (512, 768, 1), dtype=np.uint8)  # 灰度图
    pyramid = ImagePyramid(max_levels=5)
    gauss_pyr = pyramid.build_gaussian_pyramid(img)
    lap_pyr = pyramid.build_laplacian(img)
    lap_recon = pyramid.reconstruct_from_laplacian(lap_pyr)
    psnr_val = peak_signal_noise_ratio(img.astype(np.float32), lap_recon.astype(np.float32), data_range=255)
    ssim_val = structural_similarity(img.astype(np.float32)[:, :, 0], lap_recon.astype(np.float32)[:, :, 0], multichannel=False, data_range=255)
    print(f"Laplacian result: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")
