import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import zoom as gpu_zoom 

def gelu(x):
    return 0.5 * x * (1.0 + cp.tanh(cp.sqrt(2.0 / cp.pi) * (x + 0.044715 * cp.power(x, 3))))

def softmax(x, axis=-1):
    x_max = cp.max(x, axis=axis, keepdims=True)
    x_exp = cp.exp(x - x_max)
    x_sum = cp.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum

class Embeddings:
    def __init__(self, weights):
        self.hidden_size = 768
        self.patch_size  = 14
        # 设置权重
        self.cls_token           = cp.asarray(weights["embeddings.cls_token"])
        self.position_embeddings = cp.asarray(weights["embeddings.position_embeddings"])
        self.patch_embed_w       = cp.asarray(weights["embeddings.patch_embeddings.projection.weight"].reshape(768, -1).T)
        self.patch_embed_b       = cp.asarray(weights["embeddings.patch_embeddings.projection.bias"].reshape(768, 1).T)

    def pixel2patches(self, pixel_values): 
        B, C, H, W = pixel_values.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # 显存操作
        patches = pixel_values.reshape(B, C, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, num_patches_h * num_patches_w, -1)
        return patches

    def interpolate_pos_encoding(self, embeddings, height, width):
        np_pos_embed = self.position_embeddings
        cls_pos_embed = np_pos_embed[:, 0:1, :]
        patch_pos_embed = np_pos_embed[:, 1:, :]

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        target_seq_len = num_patches_h * num_patches_w
        N_pretrain = patch_pos_embed.shape[1]
        
        if N_pretrain == target_seq_len:
            return self.position_embeddings

        dim = patch_pos_embed.shape[-1]
        w0 = h0 = int(np.sqrt(N_pretrain))
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim)

        zoom_h = num_patches_h / h0
        zoom_w = num_patches_w / w0
        patch_pos_embed = gpu_zoom(patch_pos_embed, (1, zoom_h, zoom_w, 1), order=1)
        patch_pos_embed = patch_pos_embed.reshape(1, target_seq_len, dim)
        return cp.concatenate((cls_pos_embed, patch_pos_embed), axis=1)

    def __call__(self, pixel_values):
        if not isinstance(pixel_values, cp.ndarray):
            pixel_values = cp.asarray(pixel_values)
        B, _, H, W = pixel_values.shape
        patch_values = self.pixel2patches(pixel_values)
        embeddings = cp.matmul(patch_values, self.patch_embed_w) + self.patch_embed_b
        cls_token  = cp.tile(self.cls_token, (B, 1, 1))
        embeddings = cp.concatenate([cls_token, embeddings], axis=1)
        pos_embed  = self.interpolate_pos_encoding(embeddings, H, W)
        embeddings = embeddings + pos_embed
        return embeddings

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = cp.asarray(weight)
        self.bias   = cp.asarray(bias)
        self.eps = eps
    def __call__(self, x):
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1, keepdims=True)
        norm = (x - mean) / cp.sqrt(var + self.eps)
        return norm * self.weight + self.bias

class LayerScale: 
    def __init__(self, lambda1): self.lambda1 = cp.asarray(lambda1)
    def __call__(self, x): return x * self.lambda1

class Linear:
    def __init__(self, weight, bias):
        self.weight = cp.asarray(weight)
        self.bias   = cp.asarray(bias)
    def __call__(self, x):
        return cp.matmul(x, self.weight.T) + self.bias

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        def get_w(name): return weights[f"{prefix}.{name}"]
        self.q_proj = Linear(get_w("attention.query.weight"), get_w("attention.query.bias"))
        self.k_proj = Linear(get_w("attention.key.weight"), get_w("attention.key.bias"))
        self.v_proj = Linear(get_w("attention.value.weight"), get_w("attention.value.bias"))
        self.out_proj = Linear(get_w("output.dense.weight"), get_w("output.dense.bias"))

    def __call__(self, x):
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn_weights = cp.matmul(q, k.swapaxes(-2, -1)) * self.scale
        attn_weights = softmax(attn_weights, axis=-1)
        out = cp.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"]
        b1 = weights[f"{prefix}.mlp.fc1.bias"]
        w2 = weights[f"{prefix}.mlp.fc2.weight"]
        b2 = weights[f"{prefix}.mlp.fc2.bias"]
        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)
    def __call__(self, x): return self.fc2(gelu(self.fc1(x)))

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

class Dinov2GPU:
    def __init__(self, weights, config=None):
        self.config = config or {"hidden_size": 768, "num_heads": 12, "num_layers": 12, "patch_size": 14}
        self.embeddings = Embeddings(weights)
        self.blocks     = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm       = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])
    def __call__(self, pixel_values):
        pos_embed = self.embeddings(pixel_values)
        for blk in self.blocks: pos_embed = blk(pos_embed)
        pos_embed = self.norm(pos_embed)
        return pos_embed[:, 0]