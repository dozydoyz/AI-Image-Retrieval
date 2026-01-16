# core/classifier.py
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

class ImageClassifier:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def predict(self, img_path, topk=5):
        try:
            img = Image.open(img_path).convert('RGB')
            batch = self.preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.model(batch).squeeze(0).softmax(0)
            
            # 我的设置是在Label里面输出所有大于20%的结果，这里先返回5个最高的Label的列表，随后在results.html里过滤
            values, indices = torch.topk(prediction, topk)
            
            results = []
            for i in range(topk):
                score = values[i].item() * 100 # 转成百分比数值
                class_id = indices[i].item()
                name = self.weights.meta["categories"][class_id]
                results.append((name, int(score)))
            
            return results
            
        except Exception as e:
            print(f"分类出错: {e}")
            return []