import os
import numpy as np
from django.conf import settings
from .preprocess_image import resize_short_side

# ==========================================
# 这里是我的一大创新点，提供了CPU使用numpy训练以及GPU使用cupy训练两种模式
# 目前使用numpy版本，也提供了cupy版本以支持GPU加速
# 如果需要GPU训练模式，请注释掉下方的numpy导入，并取消GPU导入的注释
# ==========================================

# 模式1.numpy导入(默认)
from .dinov2_numpy import Dinov2Numpy

# 模式2.GPU导入
# import cupy as cp
# from .dinov2_gpu import Dinov2GPU

class SearchEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("Initializing AI Engine (Loading model and index)...")
            cls._instance = cls()
            print("AI Engine initialized successfully.")
        return cls._instance

    def __init__(self):
        # 1. 检查文件路径
        weights_path = os.path.join(settings.BASE_DIR, 'core', 'vit-dinov2-base.npz')
        index_path = os.path.join(settings.BASE_DIR, 'core', 'gallery_features.npz')

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Gallery index not found at: {index_path}")

        # 2. 加载基础数据
        weights = np.load(weights_path)
        data = np.load(index_path)

        # 初始化模型与特征库
        
        # [模式1]numpy初始化 (默认)
        self.model = Dinov2Numpy(weights)
        self.gallery_features = data['features'] # 保持为 numpy 数组
        self.gallery_paths = data['paths']
        self.device = 'cpu'
        print("Running in CPU mode.")

        # [模式2]GPU初始化
        # self.model = Dinov2GPU(weights)
        # dummy = cp.random.randn(1, 3, 224, 224).astype(cp.float32)
        # self.model(dummy)
        # # 将特征库传输到显存
        # self.gallery_features = cp.asarray(data['features']) 
        # self.gallery_paths = data['paths'] # 路径保持在内存
        # self.device = 'gpu'
        # print("Running in GPU mode.")

    def search(self, query_img_path, top_k=10):
        """
        执行搜索逻辑
        """
        try:
            # 1.预处理
            img = resize_short_side(query_img_path)
            # 推理与相似度计算
            # [模式1]
            # 推理
            query_feat = self.model(img)
            
            # L2归一化
            query_feat = query_feat / np.linalg.norm(query_feat, axis=-1, keepdims=True)
            
            # 相似度计算
            # (1, 768) @ (N, 768).T -> (1, N)
            sims = np.dot(query_feat, self.gallery_features.T)[0]
            
            # 排序
            indices = np.argsort(sims)[-top_k:][::-1]
            
            # 获取结果
            indices_final = indices
            sims_final = sims[indices]
            # ----------------------------------------

            # [模式2]
            # ----------------------------------------
            # # 上传到 GPU
            # img_gpu = cp.asarray(img)
            # 
            # # 推理
            # query_feat = self.model(img_gpu)
            # 
            # # 归一化
            # query_feat = query_feat / cp.linalg.norm(query_feat, axis=-1, keepdims=True)
            # 
            # # 相似度
            # sims = cp.matmul(query_feat, self.gallery_features.T)[0]
            # 
            # # 排序
            # indices = cp.argsort(sims)[-top_k:][::-1]
            # 
            # # 传回CPU
            # indices_final = indices.get()
            # sims_final = sims[indices].get()
            # ----------------------------------------

            # 3. 结果封装
            results = []
            for idx, score in zip(indices_final, sims_final):
                full_path = str(self.gallery_paths[idx])
                # 提取文件名，构建 URL:/static/gallery_images/xxx.jpg
                filename = os.path.basename(full_path)
                web_url = f"/static/gallery_images/{filename}"
                score_percent = int(score * 100)
                results.append((web_url, score_percent))

            return results

        except Exception as e:
            print(f"Error during search: {e}")
            return []