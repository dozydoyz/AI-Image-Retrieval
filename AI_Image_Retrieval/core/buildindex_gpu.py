import os
import numpy as np
import cupy as cp
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dinov2_gpu import Dinov2GPU
from preprocess_image import resize_short_side


GALLERY_DIR = 'gallery_images'
INDEX_FILE = 'gallery_features.npz'
NUM_WORKERS = 16  # 这里设置使用CPU线程数，根据电脑自身配置进行修改，数字越大性能更好


def load_and_process(path):
    """CPU的任务是读图+缩放,方便gpu端处理"""
    try:
        return resize_short_side(path), path
    except Exception as e:
        return None, str(e)

def main():
    try:
        weights = np.load("vit-dinov2-base.npz")
        model = Dinov2GPU(weights)
         # 测试模型能否正常运行，这里先预热显卡
        dummy = cp.random.randn(1, 3, 224, 224).astype(cp.float32)
        model(dummy)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 扫描图片
    image_paths = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    if not image_paths:
        print(f"错误：无法读取'{GALLERY_DIR}' ")
        return
    image_paths.sort()
    print(f"发现 {len(image_paths)} 张图片")

    all_features = []
    all_paths = []
    
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有读取任务
        results = executor.map(load_and_process, image_paths)
        
        for img_cpu, info in tqdm(results, total=len(image_paths), unit="img"):
            if img_cpu is None:
                continue # 跳过坏图
            
            try:
                # 1. 提交任务
                img_gpu = cp.asarray(img_cpu)
                
                # 2. 推理
                feat_gpu = model(img_gpu)
                
                # 3. 进行归一化
                feat_gpu = feat_gpu / cp.linalg.norm(feat_gpu, axis=-1, keepdims=True)
                
                # 4. 获得结果
                all_features.append(feat_gpu.get()[0])
                all_paths.append(info)
                
            except Exception as e:
                print(f"GPU Error: {e}")

    # 保存
    if all_features:
        
        abs_path = os.path.abspath(INDEX_FILE)
        np.savez(INDEX_FILE, features=np.array(all_features), paths=np.array(all_paths))
        print(f"文件位于: {abs_path}")

if __name__ == "__main__":
    main()