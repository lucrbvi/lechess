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

class Encoder:
    def __init__(self, img_size: int = 8, dim: int = 192, depth: int = 12,
                 n_heads: int = 3, mlp_ratio: float = 4.0, proj_dim: int | None = None,
                 proj_depth: int = 1, proj_hidden_dim: int | None = None,
                 vocab_size: int = 13, norm_eps: float = 1e-5):
        self.img_size = img_size
        self.n_patches = img_size * img_size
        self.dim = dim

        self.piece_embed = nn.Embedding(vocab_size, dim)
        self.cls_token = Tensor.zeros(1, 1, dim)
        self.freqs_cis = precompute_freqs_cis_2d(dim // n_heads, img_size, img_size)

        self.blocks = [TransformerEncoderBlock(dim, n_heads, mlp_ratio, norm_eps) for _ in range(depth)]
        self.norm = nn.RMSNorm(dim, norm_eps)
        self.proj = ProjectionHead(dim, proj_dim or dim, proj_hidden_dim, proj_depth)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B = x.shape[0]

        x = self.piece_embed(x.cast('int32').reshape(B, -1))
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = cls_tokens.cat(x, dim=1)

        for blk in self.blocks:
            x = blk(x, self.freqs_cis)

        x = self.norm(x)

        return self.proj(x[:, 0, :]), x[:, 1:, :]

class ProjectionHead:
    def __init__(self, in_dim: int, out_dim: int | None = None,
                 hidden_dim: int | None = None, depth: int = 1):
        assert depth >= 1

        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        dims = [in_dim] + [hidden_dim] * (depth - 1) + [out_dim]
        self.layers = [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(depth)]
        self.bns = [nn.BatchNorm(dims[i + 1]) for i in range(depth)]

    def __call__(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]

        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            x = layer(x)
            x = x.reshape(-1, x.shape[-1])
            x = bn(x).reshape(*lead, -1)

            if i + 1 < len(self.layers):
                x = x.silu()

        return x.reshape(*lead, -1)

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

        x = x + gate_msa * self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa, attn_mask=attn_mask)
        h = self.norm2(x) * (1 + scale_ffn) + shift_ffn
        ffn = self.ffn_down(self.ffn_gate(h).silu() * self.ffn_up(h))

        return (x + gate_ffn * ffn).contiguous()

class Predictor:
    def __init__(self, dim: int = 192, depth: int = 6, n_heads: int = 16,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, proj_dim: int | None = None,
                 proj_depth: int = 1, proj_hidden_dim: int | None = None):
        self.action_pair = nn.Embedding(64 * 64, dim)
        self.action_promo = nn.Embedding(7, dim)

        self.blocks = [AdaLNPredictorBlock(dim, n_heads, dim, mlp_ratio, dropout) for _ in range(depth)]
        self.norm = nn.RMSNorm(dim)
        self.proj = ProjectionHead(dim, proj_dim or dim, proj_hidden_dim, proj_depth)

    def __call__(self, z: Tensor, actions: Tensor) -> Tensor:
        B, L, D = z.shape

        cond_emb = self.action_pair(actions[..., 0].cast('int32') * 64 + actions[..., 1].cast('int32')) + \
            self.action_promo(actions[..., 2].cast('int32'))
        mask = Tensor.full((1, 1, L, L), float("-inf"), dtype=z.dtype, device=z.device).triu(1)

        for blk in self.blocks:
            z = blk(z, cond_emb, mask)

        return self.proj(self.norm(z))

class WorldModelHeads:
    def __init__(self, dim: int = 192, hidden_dim: int | None = None,
                 mtp_steps: int = 8, reward_bins: int = 255, action_depth: int = 2,
                 reward_depth: int = 2, action_hidden_dim: int | None = None,
                 reward_hidden_dim: int | None = None):
        assert action_depth >= 1
        assert reward_depth >= 1

        hidden_dim = hidden_dim or dim * 2
        action_hidden_dim = action_hidden_dim or hidden_dim
        reward_hidden_dim = reward_hidden_dim or hidden_dim

        self.mtp_steps = mtp_steps
        self.reward_bins = reward_bins
        self.from_embed = nn.Embedding(64, action_hidden_dim)
        self.to_embed = nn.Embedding(64, action_hidden_dim)

        action_dims = [dim] + [action_hidden_dim] * action_depth
        reward_dims = [dim] + [reward_hidden_dim] * (reward_depth - 1) + [mtp_steps * reward_bins]

        self.action_layers = [nn.Linear(action_dims[i], action_dims[i + 1]) for i in range(action_depth)]
        self.from_heads = [nn.Linear(action_hidden_dim, 64) for _ in range(mtp_steps)]
        self.to_heads = [nn.Linear(action_hidden_dim, 64) for _ in range(mtp_steps)]
        self.promo_heads = [nn.Linear(action_hidden_dim, 7) for _ in range(mtp_steps)]
        self.reward_layers = [nn.Linear(reward_dims[i], reward_dims[i + 1]) for i in range(reward_depth)]

    def action_hidden(self, x: Tensor) -> Tensor:
        for layer in self.action_layers:
            x = layer(x).silu()

        return x

    def action_logits(self, h: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from_logits, to_logits, promo_logits = [], [], []

        for i in range(self.mtp_steps):
            frm = actions[:, :, i, 0].cast('int32')
            to = actions[:, :, i, 1].cast('int32')
            from_logits.append(self.from_heads[i](h))
            to_logits.append(self.to_heads[i](h + self.from_embed(frm)))
            promo_logits.append(self.promo_heads[i](h + self.from_embed(frm) + self.to_embed(to)))

        return Tensor.stack(*from_logits, dim=2), Tensor.stack(*to_logits, dim=2), Tensor.stack(*promo_logits, dim=2)

    def __call__(self, x: Tensor, actions: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
        lead = x.shape[:-1]

        reward = x
        for layer in self.reward_layers[:-1]:
            reward = layer(reward).silu()
        reward_logits = self.reward_layers[-1](reward).reshape(*lead, self.mtp_steps, self.reward_bins)

        return self.action_logits(self.action_hidden(x), actions), reward_logits
