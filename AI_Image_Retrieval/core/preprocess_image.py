import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    # 第一步：加载图片
    image = Image.open(img_path).convert("RGB")

    # 第二步：中心裁剪
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # 尺寸变为 (224, 224)

    # 第三步：转换为Numpy数组
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # 第四步：归一化
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] 

def resize_short_side(img_path, target_size=224, patch_size=14):

    # 第一步：加载图片
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # 第二步：计算新尺寸
    scale = target_size / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 确保缩放后的高和宽都是patch_size(14)的倍数。
    new_w = int(round(new_w / patch_size) * patch_size)
    new_h = int(round(new_h / patch_size) * patch_size)
    
    # 防止尺寸缩小到0
    new_w = max(new_w, patch_size)
    new_h = max(new_h, patch_size)

    image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    # 第三步：转换为Numpy数组
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # 第四步：归一化
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] 