import numpy as np
import torch
import pdb
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = 0.5
    def __call__(self, img_tensor):
        '''
            img_tensor: [b, 3, h, w]
        '''
        if np.random.uniform(0, 1.0) < self.p:
            img_tensor = torch.flip(img_tensor, dims=(2,))
        return img_tensor

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_normaled=True):
        self.mean = torch.tensor(mean).reshape(1, 3, 1,1)
        self.std = torch.tensor(std).reshape(1, 3, 1,1)
        self.is_normaled = is_normaled
    def __call__(self, x: torch.Tensor):
        '''
            x: [b, 3, h, w]
        '''
        if not self.is_normaled:
            
            x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return x


class CustomColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, is_RGB=True):
        """
        自定义 ColorJitter 数据增强
        Args:
            brightness: 亮度调整幅度（0~1），对应 [1-brightness, 1+brightness] 范围的随机缩放
            contrast: 对比度调整幅度（0~1）
            saturation: 饱和度调整幅度（0~1）
            is_RGB: if False, img is bgr
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.is_RGB=is_RGB

        # 验证参数范围
        if brightness < 0 or contrast < 0 or saturation < 0:
            raise ValueError("调整幅度不能为负数")

    def _adjust_brightness(self, img_tensor: torch.Tensor):
        """调整亮度：img shape (b, 3, H, W)，uint8 或 float32"""
        if self.brightness == 0:
            return img

        # 生成随机亮度因子（在 [1-b, 1+b] 之间）
        brightness_factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
        img = img_tensor.to(torch.float32) * brightness_factor

        # 确保像素值在 [0, 255] 范围内（避免溢出）
        img = torch.clip(img, 0, 255)
        return img

    def _adjust_contrast(self, img):
        """调整对比度：img shape (b, 3, H, W) float32"""
        if self.contrast == 0:
            return img

        # 生成随机对比度因子（在 [1-c, 1+c] 之间）
        contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
        img = img.to(torch.float32)
        # 对比度调整公式：img = (img - mean) * factor + mean（保持亮度均值不变）
        mean = torch.mean(img, dim=(-2,-1), keepdim=True)
        # mean = np.mean(img, axis=(-2, -1), keepdims=True)  # 每个通道的均值
        img = (img - mean) * contrast_factor + mean
        # 裁剪像素值范围
        img = torch.clip(img, 0.0, 255.0)
        return img

    def _adjust_saturation(self, img:torch.Tensor):
        """调整饱和度：img shape (b, 3, H, W)，uint8 或 float32（RGB 格式）"""
        if self.saturation == 0:
            return img

        # 生成随机饱和度因子（在 [1-s, 1+s] 之间）
        saturation_factor = np.random.uniform(1 - self.saturation, 1 + self.saturation)
        img = img.to(torch.float32)

        # 方法1：RGB 空间直接计算（快速）
        # 灰度图 = 0.299*R + 0.587*G + 0.114*B（人眼对亮度的感知权重）
        gray = img.clone()
        if self.is_RGB:
            gray[:, 0, ...] = gray[:, 0, ...]*0.299
            gray[:, 2, ...] = gray[:, 2, ...]*0.114
        else:
            gray[:, 0, ...] = gray[:, 0, ...]*0.114
            gray[:, 2, ...] = gray[:, 2, ...]*0.299

        gray[:, 1, ...] = gray[:, 1, ...]*0.587
            
        # gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]  # (H,W,1)
        # 饱和度调整：img = gray + (img - gray) * factor（factor=0 则为灰度图，factor>1 饱和度提升）
        img = gray + (img - gray) * saturation_factor

        # 方法2：HSV 空间调整（更精准，需转换通道，稍慢）
        # img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        # img_hsv[..., 1] = np.clip(img_hsv[..., 1] * saturation_factor, 0, 255)
        # img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        # 裁剪像素值范围
        img = torch.clip(img, 0, 255)
        return img

    def __call__(self, img):
        """
        执行颜色抖动（随机顺序应用亮度、对比度、饱和度调整）
        Args:
            img: 输入图像，torch.Tensor, [b, 3, h, w]
        Returns:
            增强后的图像（与输入格式一致）
        """
 
        # 随机调整顺序（模拟 PyTorch 逻辑，提升随机性）
        transforms = []
        if self.brightness > 0:
            transforms.append(self._adjust_brightness)
        if self.contrast > 0:
            transforms.append(self._adjust_contrast)
        if self.saturation > 0:
            transforms.append(self._adjust_saturation)
        
        np.random.shuffle(transforms)  # 随机打乱调整顺序

        # 应用所有变换
        for transform in transforms:
            img = transform(img)
        return img


