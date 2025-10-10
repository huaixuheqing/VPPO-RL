import numpy as np
from PIL import Image

def random_patch_blackening(pil_img, patch_size=14, black_prob=0.5):
    """Randomly blacken square patches in a PIL image."""
    img = np.array(pil_img).astype(np.float32)
    h, w = img.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if np.random.rand() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if img.ndim == 3:
                    img[y:y_end, x:x_end, :] = 0
                else:
                    img[y:y_end, x:x_end] = 0
    return Image.fromarray(img.astype(np.uint8))

def add_gaussian_noise(pil_img, mean=0.0, std=189):
    """
    向 PIL 图像添加高斯噪声。

    参数:
    pil_img (PIL.Image.Image): 输入的 PIL 图像。
    mean (float): 噪声的均值 (通常为 0)。
    std (float): 噪声的标准差。值越大，噪声越明显。

    返回:
    PIL.Image.Image: 添加了噪声的 PIL 图像。
    """
    # 将 PIL 图像转换为 NumPy 数组，使用浮点数以便计算
    img = np.array(pil_img).astype(np.float32)
    
    # 获取图像尺寸和通道数
    if img.ndim == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1 # 灰度图

    # 创建一个与图像尺寸相同的高斯噪声数组
    # 如果是 RGBA 图像，我们只对 RGB 通道添加噪声
    if c == 4:
        # 只为 RGB 通道生成噪声
        noise = np.random.normal(mean, std, (h, w, 3))
        # 创建一个全零的 alpha 通道噪声
        alpha_noise = np.zeros((h, w, 1))
        # 合并噪声
        noise = np.concatenate((noise, alpha_noise), axis=-1)
    else:
        noise = np.random.normal(mean, std, img.shape)
    
    # 将噪声添加到图像上
    noisy_img = img + noise
    
    # 将像素值裁剪到有效的 [0, 255] 范围内
    noisy_img = np.clip(noisy_img, 0, 255)
    
    # 将数组转换回 PIL 图像所需的 uint8 类型
    noisy_img = noisy_img.astype(np.uint8)
    
    # 从数组创建新的 PIL 图像并返回
    return Image.fromarray(noisy_img)

def complete_masking(pil_img: Image.Image, mask_value: Union[int, Tuple] = 128):
    """
    用指定的纯色完全替换（遮蔽）一个 PIL 图像。
    
    参数:
    pil_img (PIL.Image.Image): 输入的 PIL 图像。
    mask_value (Union[int, Tuple]): 用于填充的颜色值。
                                   对于灰度图，应为单个整数 (e.g., 128)。
                                   对于RGB图，可以是单个整数（将被广播为灰色）或一个元组 (e.g., (128, 128, 128))。
    
    返回:
    PIL.Image.Image: 一个与原始图像尺寸和模式相同，但内容被纯色替换的图像。
    """
    # 将 PIL 图像转换为 NumPy 数组以获取其形状和类型
    original_array = np.array(pil_img)

    # 使用 np.full_like 创建一个与原始数组形状和类型都相同的数组，
    # 并用指定的 mask_value 填充它。
    # 这个函数能优雅地处理灰度图和多通道（如RGB）彩色图。
    masked_array = np.full_like(original_array, fill_value=mask_value, dtype=np.uint8)

    # 从新的 NumPy 数组创建并返回 PIL 图像
    return Image.fromarray(masked_array)

def gaussian_blur(pil_img: Image.Image, radius: Union[int, float] = 6.0):
    """
    对一个 PIL 图像应用高斯模糊。

    参数:
    pil_img (PIL.Image.Image): 输入的 PIL 图像。
    radius (Union[int, float]): 模糊半径，对应高斯核的标准差(sigma)。
                                 值越大，图像越模糊。
    
    返回:
    PIL.Image.Image: 应用了高斯模糊效果的新图像。
    """
    # Pillow 的 filter 方法可以直接应用高斯模糊效果
    # ImageFilter.GaussianBlur() 会创建一个模糊滤镜对象
    # .filter() 方法返回一个新的、经过滤镜处理的图像
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

augment_image = random_patch_blackening