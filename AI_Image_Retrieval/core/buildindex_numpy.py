import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side


GALLERY_DIR = 'gallery_images'
INDEX_FILE = 'gallery_features.npz'
NUM_WORKERS = 16  # 这里设置使用CPU核心数，根据电脑自身配置进行修改，数字越大性能更好


def process_one_image_safe(path):
    """
    单个图片预处理函数
    返回: (pixel_values, path)
    """
    try:
        # resize_short_side 返回的是 (1, C, H, W)
        # 正好直接喂给模型，不需要额外的stack操作
        img = resize_short_side(path)
        return img, path
    except Exception as e:
        return None, None

def main():
    # 1. 加载模型
    weights = np.load("vit-dinov2-base.npz")
    model = Dinov2Numpy(weights)

    # 2. 扫描文件
    # 这里支持3种后缀
    image_paths = glob.glob(os.path.join(GALLERY_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(GALLERY_DIR, "*.png")) + \
                  glob.glob(os.path.join(GALLERY_DIR, "*.jpeg"))
    image_paths.sort()
    
    if not image_paths:
        print("错误：无法读取'{GALLERY_DIR}'")
        return
    print(f"发现 {len(image_paths)} 张图片")

    all_features = []
    all_paths = []

    # 3. 并行加载照片，随后进行串行推理
    print(f"⚡ 开始特征提取 (单图模式, 兼容变长输入)...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        # future_to_path字典用于追踪进度
        futures = {executor.submit(process_one_image_safe, p): p for p in image_paths}
        
        # as_completed会在任务完成时立即yield，不用按顺序等待
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="Building Index"):
            try:
                img_tensor, path = future.result()
                
                if img_tensor is None:
                    continue

                # 模型推理
                feat = model(img_tensor)
                
                # 归一化
                feat = feat / np.linalg.norm(feat, axis=-1, keepdims=True)
                
                all_features.append(feat[0])
                all_paths.append(path)

            except Exception as e:
                # 某张图损毁直接跳过
                pass

    # 4. 合并保存
    if all_features:
        final_features = np.array(all_features)
        final_paths = np.array(all_paths)
        np.savez(INDEX_FILE, features=final_features, paths=final_paths)

if __name__ == "__main__":
    main()