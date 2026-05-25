from tinygrad import Tensor

def sigreg(z: Tensor, M: int = 1024, n_quad: int = 32, t_min: float = 0.2,
           t_max: float = 4.0, lam: float = 1.0) -> Tensor:
    N, B, D = z.shape
    Z = z.reshape(N * B, D)

    u = Tensor.randn(D, M, device=z.device, dtype=z.dtype)
    u = u / u.square().sum(0, keepdim=True).sqrt()
    h = Z @ u

    t = Tensor.linspace(t_min, t_max, n_quad, device=z.device, dtype=z.dtype)
    dt = (t_max - t_min) / (n_quad - 1)
    ht = h.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)
    cos_mean = ht.cos().mean(0)
    sin_mean = ht.sin().mean(0)

    t2 = t.square()
    phi0 = (-t2 / 2).exp()
    w = (-t2 / (2 * lam * lam)).exp()
    diff_sq = (cos_mean - phi0.unsqueeze(0)).square() + sin_mean.square()
    integrand = w.unsqueeze(0) * diff_sq
    T_per_proj = (integrand[:, 1:-1].sum(1) * 2 + integrand[:, 0] + integrand[:, -1]) * dt / 2
    return T_per_proj.mean()

def prediction_loss(z_pred: Tensor, z_target: Tensor) -> Tensor:
    return (z_pred - z_target).square().mean()

def lewm_loss(encoder, predictor, observations: Tensor, actions: Tensor,
              M: int = 1024, lam_sigreg: float = 0.1, **sigreg_kwargs) -> tuple[Tensor, Tensor, Tensor]:
    seq_len = observations.shape[1]
    N = seq_len - 1

    z_all = []
    for t in range(seq_len):
        cls_t, _ = encoder(observations[:, t])
        z_all.append(cls_t)
    z_all = Tensor.stack(*z_all, dim=0)
    z_input, z_target = z_all[:N], z_all[1:]

    z_pred = predictor(z_input.permute(1, 0, 2), actions).permute(1, 0, 2)
    pred_loss = prediction_loss(z_pred, z_target)
    sigreg_loss = sigreg(z_target, M=M, **sigreg_kwargs)
    return pred_loss + lam_sigreg * sigreg_loss, pred_loss, sigreg_loss
