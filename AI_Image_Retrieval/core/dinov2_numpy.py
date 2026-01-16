import numpy as np
from scipy.ndimage import zoom

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum

class Embeddings:
    def __init__(self, weights):
        self.hidden_size = 768 # D
        self.patch_size  = 14  # ps

        self.cls_token           = weights["embeddings.cls_token"] # (1, 1, D)
        self.position_embeddings = weights["embeddings.position_embeddings"] # (1, N+1, D)
        self.patch_embed_w       = weights["embeddings.patch_embeddings.projection.weight"].reshape(768, -1).T
        self.patch_embed_b       = weights["embeddings.patch_embeddings.projection.bias"].reshape(768, 1).T

    def pixel2patches(self, pixel_values): 
        B, C, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        # 将图片切分成patches
        # 输入: (B, C, H, W)
        # 切割为 (B, C, H//ps, ps, W//ps, ps)
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        patches = pixel_values.reshape(B, C, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        #(B, H//ps, W//ps, C, ps, ps) -> (B, N, C*ps*ps)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, num_patches_h * num_patches_w, -1)
        
        return patches

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        对位置编码进行插值以适应不同的输入分辨率，这里是我的创新点，我的模型可以不切割图片到指定分辨率而直接进行推理
        """
        # 1.分离CLS token和patch tokens的位置编码
        np_pos_embed = self.position_embeddings # (1, N_pretrain + 1, D)
        cls_pos_embed = np_pos_embed[:, 0:1, :]
        patch_pos_embed = np_pos_embed[:, 1:, :]

        # 2. 获取当前的patch数量和预训练的patch数量
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        target_seq_len = num_patches_h * num_patches_w
        
        N_pretrain = patch_pos_embed.shape[1]
        
        # 如果尺寸一致，直接返回
        if N_pretrain == target_seq_len:
            return self.position_embeddings

        # 3. 将预训练的位置编码reshape成2D网格 (sqrt(N), sqrt(N))
        dim = patch_pos_embed.shape[-1]
        w0 = h0 = int(np.sqrt(N_pretrain)) # 假设预训练是正方形的，通常是 14x14 (224/14)
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim)

        # 4. 使用scipy.ndimage.zoom进行插值，只缩放 h 和 w
        zoom_h = num_patches_h / h0
        zoom_w = num_patches_w / w0
        patch_pos_embed = zoom(patch_pos_embed, (1, zoom_h, zoom_w, 1), order=1) # order=1 for bilinear, often sufficient

        # 5. 拼接CLS token
        patch_pos_embed = patch_pos_embed.reshape(1, target_seq_len, dim)
        return np.concatenate((cls_pos_embed, patch_pos_embed), axis=1)

    def __call__(self, pixel_values):
        B, _, H, W = pixel_values.shape

        patch_values = self.pixel2patches(pixel_values) # (B, N, C*ps**2)
        
        # 线性投影
        embeddings = patch_values @ self.patch_embed_w + self.patch_embed_b
        
        # 添加CLS token
        cls_token  = np.tile(self.cls_token, (B, 1, 1))
        embeddings = np.concatenate([cls_token, embeddings], axis=1)

        # 添加位置编码(处理分辨率变化)
        pos_embed  = self.interpolate_pos_encoding(embeddings, H, W)
        
        embeddings = embeddings + pos_embed
        return embeddings

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight
        self.bias   = bias
        self.eps    = eps

    def __call__(self, x):
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return norm * self.weight + self.bias

class LayerScale: 
    def __init__(self, lambda1): 
        self.lambda1 = lambda1

    def __call__(self, x): 
        return x * self.lambda1

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias   = bias

    def __call__(self, x):
        return x @ self.weight.T + self.bias

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj   = Linear(q_w, q_b)
        self.k_proj   = Linear(k_w, k_b)
        self.v_proj   = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        B, N, C = x.shape
        
        # 1. 投影 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape为多头: (B, N, num_heads, head_dim)，转置为 (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 3. 计算 Attention Scores: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        # (B, nh, N, hd) @ (B, nh, hd, N) -> (B, nh, N, N)
        attn_weights = (q @ k.swapaxes(-2, -1)) * self.scale
        attn_weights = softmax(attn_weights, axis=-1)

        # 4. 加权求和
        # (B, nh, N, N) @ (B, nh, N, hd) -> (B, nh, N, hd)
        out = attn_weights @ v

        # 5. 还原形状 (B, N, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)

        # 6. 输出投影
        return self.out_proj(out)

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"]
        b1 = weights[f"{prefix}.mlp.fc1.bias"]
        w2 = weights[f"{prefix}.mlp.fc2.weight"]
        b2 = weights[f"{prefix}.mlp.fc2.bias"]

        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"
        
        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, f"{prefix}.attention", weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(f"{prefix}", weights)

    def __call__(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.mlp(self.norm2(x)))
        return x

class Dinov2Numpy:
    def __init__(self, weights, config=None):
        self.weights = weights
        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "patch_size": 14,
        }

        self.embeddings = Embeddings(weights)
        self.blocks     = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm       = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        pos_embed = self.embeddings(pixel_values)
        for blk in self.blocks:
            pos_embed = blk(pos_embed)
        pos_embed = self.norm(pos_embed)
        return pos_embed[:, 0]