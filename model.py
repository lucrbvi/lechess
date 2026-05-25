from tinygrad import Tensor, nn
from tinygrad.llm.model import apply_rope

def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0) -> Tensor:
    assert dim % 4 == 0
    half_dim = dim // 4
    freqs = 1.0 / (theta ** (Tensor.arange(0, half_dim) / half_dim))
    row_pos = Tensor.arange(height).unsqueeze(1).repeat(1, width).flatten()
    col_pos = Tensor.arange(width).unsqueeze(0).repeat(height, 1).flatten()
    row_angles = row_pos.unsqueeze(1) * freqs.unsqueeze(0)
    col_angles = col_pos.unsqueeze(1) * freqs.unsqueeze(0)
    row_cos, row_sin = row_angles.cos(), row_angles.sin()
    col_cos, col_sin = col_angles.cos(), col_angles.sin()
    all_cos = row_cos.cat(col_cos, dim=-1)
    all_sin = row_sin.cat(col_sin, dim=-1)
    return all_cos.cat(all_sin, dim=-1).reshape(1, 1, -1, dim)

class MultiHeadAttention:
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: Tensor, freqs_cis: Tensor | None = None, attn_mask: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if freqs_cis is not None:
            q_p = apply_rope(q[:, :, 1:, :], freqs_cis)
            k_p = apply_rope(k[:, :, 1:, :], freqs_cis)
            q = q[:, :, :1, :].cat(q_p, dim=2)
            k = k[:, :, :1, :].cat(k_p, dim=2)

        attn = q.scaled_dot_product_attention(k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        return self.proj(attn.transpose(1, 2).reshape(B, L, D))

class TransformerEncoderBlock:
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, norm_eps: float = 1e-5):
        self.norm1 = nn.RMSNorm(dim, norm_eps)
        self.norm2 = nn.RMSNorm(dim, norm_eps)
        self.attn = MultiHeadAttention(dim, n_heads)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.ffn_up = nn.Linear(dim, hidden_dim, bias=False)
        self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: Tensor, freqs_cis: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        h = self.norm2(x)
        ffn = self.ffn_down(self.ffn_gate(h).silu() * self.ffn_up(h))
        return (x + ffn).contiguous()

class VisionTransformerEncoder:
    def __init__(self, img_size: int = 8, dim: int = 256, depth: int = 6,
                 n_heads: int = 8, mlp_ratio: float = 4.0, norm_eps: float = 1e-5):
        self.img_size = img_size
        self.n_patches = img_size * img_size
        self.dim = dim
        self.patch_conv = nn.Conv2d(2, dim, kernel_size=1, stride=1, bias=False)
        self.cls_token = Tensor.zeros(1, 1, dim)
        self.freqs_cis = precompute_freqs_cis_2d(dim // n_heads, img_size, img_size)
        self.blocks = [TransformerEncoderBlock(dim, n_heads, mlp_ratio, norm_eps) for _ in range(depth)]
        self.norm = nn.RMSNorm(dim, norm_eps)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B = x.shape[0]
        x = self.patch_conv(x.permute(0, 3, 1, 2))
        x = x.reshape(B, self.dim, self.n_patches).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = cls_tokens.cat(x, dim=1)
        for blk in self.blocks:
            x = blk(x, self.freqs_cis)
        x = self.norm(x)
        return x[:, 0, :], x[:, 1:, :]

class ProjectionHead:
    def __init__(self, in_dim: int, out_dim: int | None = None):
        out_dim = out_dim or in_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm(out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        x = self.linear(x)
        x = x.reshape(-1, x.shape[-1])
        x = self.bn(x)
        return x.reshape(*lead, -1)

class Encoder:
    def __init__(self, img_size: int = 8, dim: int = 192, depth: int = 12,
                 n_heads: int = 3, proj_dim: int | None = None, **kwargs):
        self.vit = VisionTransformerEncoder(img_size, dim, depth, n_heads, **kwargs)
        self.proj = ProjectionHead(dim, proj_dim or dim)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        cls, patches = self.vit(x)
        return self.proj(cls), patches

class AdaLNPredictorBlock:
    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, norm_eps: float = 1e-5):
        self.norm1 = nn.RMSNorm(dim, norm_eps, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, norm_eps, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, n_heads, dropout=dropout)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.ffn_up = nn.Linear(dim, hidden_dim, bias=False)
        self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)
        self.adaln = nn.Linear(cond_dim, dim * 6, bias=True)
        self.adaln.weight = Tensor.zeros(dim * 6, cond_dim)
        self.adaln.bias = Tensor.zeros(dim * 6)

    def __call__(self, x: Tensor, cond_emb: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = \
            self.adaln(cond_emb).chunk(6, dim=-1)

        x = x + gate_msa * self.attn(self._modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)

        h = self._modulate(self.norm2(x), shift_ffn, scale_ffn)
        ffn = self.ffn_down(self.ffn_gate(h).silu() * self.ffn_up(h))
        return (x + gate_ffn * ffn).contiguous()

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale) + shift

class ActionEmbedder:
    def __init__(self, dim: int):
        self.pair = nn.Embedding(64 * 64, dim)
        self.promo = nn.Embedding(7, dim)

    def __call__(self, actions: Tensor) -> Tensor:
        f = actions[..., 0].cast('int32')
        t = actions[..., 1].cast('int32')
        p = actions[..., 2].cast('int32')
        return self.pair(f * 64 + t) + self.promo(p)

class Predictor:
    def __init__(self, dim: int = 192, depth: int = 6, n_heads: int = 16,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, proj_dim: int | None = None):
        self.action_embed = ActionEmbedder(dim)
        self.blocks = [AdaLNPredictorBlock(dim, n_heads, dim, mlp_ratio, dropout) for _ in range(depth)]
        self.norm = nn.RMSNorm(dim)
        self.proj = ProjectionHead(dim, proj_dim or dim)

    def __call__(self, z: Tensor, actions: Tensor) -> Tensor:
        B, L, D = z.shape
        cond_emb = self.action_embed(actions)
        mask = Tensor.full((1, 1, L, L), float("-inf"), dtype=z.dtype, device=z.device).triu(1)
        for blk in self.blocks:
            z = blk(z, cond_emb, mask)
        return self.proj(self.norm(z))

class WorldModel:
    def __init__(self, img_size: int = 8, dim: int = 192, enc_depth: int = 12, pred_depth: int = 6,
                 enc_heads: int = 3, pred_heads: int = 16, proj_dim: int | None = None):
        self.encoder = Encoder(img_size, dim, enc_depth, enc_heads, proj_dim=proj_dim)
        self.predictor = Predictor(dim, pred_depth, pred_heads, proj_dim=proj_dim)

    def __call__(self, z: Tensor, actions: Tensor) -> Tensor:
        cls, _ = self.encoder(z)
        return self.predictor(cls, actions)
