import cv2
import numpy as np
import matplotlib.pyplot as plt


class Morphology:
    def __init__(self, kernel, iterations=1, pad_mode='edge'):
        self.kernel = kernel.astype(np.uint8)
        self.iterations = iterations
        self.pad_mode = pad_mode  # default is 'edge' to mimic OpenCV

    def _apply_single(self, image, op='erode'):
        kh, kw = self.kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Determine pad kwargs based on pad_mode
        if self.pad_mode == 'constant':
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        else:
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=self.pad_mode)

        h, w = image.shape
        output = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]
                if op == 'erode':
                    output[i, j] = np.min(region[self.kernel == 1])
                elif op == 'dilate':
                    output[i, j] = np.max(region[self.kernel == 1])
        return output

    def erode(self, image):
        result = image.copy()
        for _ in range(self.iterations):
            result = self._apply_single(result, op='erode')
        return result

    def dilate(self, image):
        result = image.copy()
        for _ in range(self.iterations):
            result = self._apply_single(result, op='dilate')
        return result

    def opening(self, image):
        return self.dilate(self.erode(image))

    def closing(self, image):
        return self.erode(self.dilate(image))

    def apply(self, image, mode='erode'):
        if image.ndim == 2:
            return self._dispatch(image, mode)
        elif image.ndim == 3:
            return np.stack([self._dispatch(im, mode) for im in image], axis=0)
        elif image.ndim == 4:
            b, c, h, w = image.shape
            return np.stack([
                np.stack([self._dispatch(image[b_i, c_i], mode) for c_i in range(c)], axis=0)
                for b_i in range(b)
            ], axis=0)
        else:
            raise ValueError(f"Unsupported image dimension: {image.ndim}")

    def _dispatch(self, image, mode):
        if mode == 'erode':
            return self.erode(image)
        elif mode == 'dilate':
            return self.dilate(image)
        elif mode == 'opening':
            return self.opening(image)
        elif mode == 'closing':
            return self.closing(image)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __call__(self, image, mode='erode'):
        return self.apply(image, mode=mode)


if __name__ == "__main__":
    # 1. 生成测试图像（随机整数）
    np.random.seed(42)
    image = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)

    # 2. 定义 kernel 和 iterations
    kernel = np.ones((3, 3), dtype=np.uint8)
    iterations = 1

    # 3. 使用 OpenCV 官方实现
    official_eroded = cv2.erode(image, kernel, iterations=iterations)
    official_dilated = cv2.dilate(image, kernel, iterations=iterations)
    official_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # 4. 使用你的实现
    morph = Morphology(kernel, iterations=iterations)
    custom_eroded = morph(image, mode='erode')
    custom_dilated = morph(image, mode='dilate')
    custom_opened = morph(image, mode='opening')

    # 5. 对比差异（打印最大误差或是否完全一致）
    print("Erosion difference (max):", np.max(np.abs(official_eroded - custom_eroded)))
    print("Dilation difference (max):", np.max(np.abs(official_dilated - custom_dilated)))
    print("Opening difference (max):", np.max(np.abs(official_opened - custom_opened)))

    def show_diff(title, img1, img2):
        plt.figure(figsize=(10, 3))
        plt.suptitle(title)
        plt.subplot(1, 3, 1)
        plt.imshow(img1, cmap='gray')
        plt.title("OpenCV")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img2, cmap='gray')
        plt.title("Custom")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(img1.astype(int) - img2.astype(int)), cmap='hot')
        plt.title("Difference")
        plt.axis('off')
        plt.show()


    # 可视化对比
    show_diff("Erosion", official_eroded, custom_eroded)
    show_diff("Dilation", official_dilated, custom_dilated)
    show_diff("Opening", official_opened, custom_opened)
