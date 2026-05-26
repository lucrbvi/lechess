import argparse
import logging
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tinygrad import Tensor
from tinygrad import nn
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save

from model import Encoder, Predictor, WorldModelHeads

LOGGER = logging.getLogger(__name__)

def sigreg(z: Tensor, M: int = 1024, n_quad: int = 32, t_min: float = 0.2,
           t_max: float = 4.0, lam: float = 1.0) -> Tensor:
    B, T, D = z.shape
    Z = z.reshape(B * T, D)

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

def action_ids(actions: Tensor) -> Tensor:
    f = actions[..., 0].cast('int32')
    t = actions[..., 1].cast('int32')
    p = actions[..., 2].cast('int32')
    return (f * 64 + t) * 7 + p

def symexp(x: Tensor) -> Tensor:
    return x.sign() * (x.abs().exp() - 1)

def symlog(x: Tensor) -> Tensor:
    return x.sign() * (x.abs() + 1).log()

def twohot(values: Tensor, bins: Tensor, low: float, high: float) -> Tensor:
    values = symlog(values).clip(low, high)
    below = (values.unsqueeze(-1) >= bins).sum(-1).cast('int32') - 1
    below = below.clip(0, bins.shape[0] - 2)
    above = below + 1
    lo = bins[below]
    hi = bins[above]
    weight_hi = ((values - lo) / (hi - lo)).clip(0, 1)
    weight_lo = 1 - weight_hi
    return below.one_hot(bins.shape[0]) * weight_lo.unsqueeze(-1) + above.one_hot(bins.shape[0]) * weight_hi.unsqueeze(-1)

def twohot_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * logits.log_softmax(-1)).sum(-1).mean()

def encode_observations(encoder, observations: Tensor) -> Tensor:
    B, T = observations.shape[:2]
    cls, _ = encoder(observations.reshape(B * T, *observations.shape[2:]))
    return cls.reshape(B, T, -1)

def _predict_next(predictor, context: Tensor, actions: Tensor) -> Tensor:
    return predictor(context, actions)[:, -1]

def _anchored_predictions(predictor, z: Tensor, actions: Tensor,
                          context_window: int, rollout_steps: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    T = z.shape[1]
    teacher_preds = []
    teacher_targets = []
    rollout_preds = []
    rollout_targets = []

    last_anchor = T - rollout_steps - 1
    for anchor in range(context_window - 1, last_anchor + 1):
        context_start = anchor - context_window + 1
        context = z[:, context_start:anchor + 1]
        action_context = actions[:, context_start:anchor + 1]

        pred = _predict_next(predictor, context, action_context)
        teacher_preds.append(pred)
        teacher_targets.append(z[:, anchor + 1])

        rollout_context = context[:, 1:].cat(pred.unsqueeze(1), dim=1)
        for step in range(1, rollout_steps):
            target_index = anchor + step + 1
            action_start = target_index - context_window
            action_end = target_index
            action_context = actions[:, action_start:action_end]

            pred = _predict_next(predictor, rollout_context, action_context)
            rollout_preds.append(pred)
            rollout_targets.append(z[:, target_index])
            rollout_context = rollout_context[:, 1:].cat(pred.unsqueeze(1), dim=1)

    return (
        Tensor.stack(*teacher_preds, dim=1),
        Tensor.stack(*teacher_targets, dim=1),
        Tensor.stack(*rollout_preds, dim=1),
        Tensor.stack(*rollout_targets, dim=1),
    )

def lewm_loss(encoder, predictor, observations: Tensor, actions: Tensor,
              M: int = 1024, lam_sigreg: float = 0.1, lam_rollout: float = 1.0,
              context_window: int = 3, rollout_steps: int = 2,
              **sigreg_kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assert context_window >= 1
    assert rollout_steps >= 2
    assert observations.shape[1] >= context_window + rollout_steps
    assert actions.shape[1] == observations.shape[1] - 1

    z = encode_observations(encoder, observations)
    z_teacher, z_teacher_target, z_rollout, z_rollout_target = _anchored_predictions(
        predictor, z, actions, context_window, rollout_steps
    )

    teacher_loss = prediction_loss(z_teacher, z_teacher_target)
    rollout_loss = prediction_loss(z_rollout, z_rollout_target)
    sigreg_loss = sigreg(z, M=M, **sigreg_kwargs)
    total = teacher_loss + lam_rollout * rollout_loss + lam_sigreg * sigreg_loss
    return total, teacher_loss, rollout_loss, sigreg_loss

def bc_loss(encoder, heads, observations: Tensor, actions: Tensor, rewards: Tensor,
            mtp_steps: int = 8, reward_bins: int = 255,
            reward_symlog_min: float = -5.0, reward_symlog_max: float = 5.0,
            lam_reward: float = 1.0, freeze_encoder: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    assert actions.shape[1] >= mtp_steps
    z = encode_observations(encoder, observations)
    if freeze_encoder:
        z = z.detach()
    max_start = actions.shape[1] - mtp_steps + 1
    h = z[:, :max_start]
    action_logits, reward_logits = heads(h)

    future_actions = Tensor.stack(*[action_ids(actions[:, i:i + max_start]) for i in range(mtp_steps)], dim=2)
    future_rewards = Tensor.stack(*[rewards[:, i:i + max_start] for i in range(mtp_steps)], dim=2)

    action_loss = action_logits.reshape(-1, action_logits.shape[-1]).sparse_categorical_crossentropy(
        future_actions.reshape(-1)
    )
    bins = Tensor.linspace(reward_symlog_min, reward_symlog_max, reward_bins, device=rewards.device, dtype=rewards.dtype)
    reward_targets = twohot(future_rewards, bins, reward_symlog_min, reward_symlog_max)
    reward_loss = twohot_cross_entropy(reward_logits, reward_targets)
    return action_loss + lam_reward * reward_loss, action_loss, reward_loss

def bc_train_step(encoder, heads, optimizer, observations: Tensor, actions: Tensor,
                  rewards: Tensor, **loss_kwargs):
    with Tensor.train():
        optimizer.zero_grad()
        total, action_loss, reward_loss = bc_loss(encoder, heads, observations, actions, rewards, **loss_kwargs)
        total.backward()
        optimizer.step()
    return total, action_loss, reward_loss

def make_bc_train_step(encoder, heads, optimizer, jit: bool = True, **loss_kwargs):
    def step(observations: Tensor, actions: Tensor, rewards: Tensor):
        return bc_train_step(encoder, heads, optimizer, observations, actions, rewards, **loss_kwargs)
    return TinyJit(step) if jit else step

def train_step(encoder, predictor, optimizer, observations: Tensor, actions: Tensor, **loss_kwargs):
    with Tensor.train():
        optimizer.zero_grad()
        total, teacher_loss, rollout_loss, sigreg_loss = lewm_loss(
            encoder, predictor, observations, actions, **loss_kwargs
        )
        total.backward()
        optimizer.step()
    return total, teacher_loss, rollout_loss, sigreg_loss

def prepare_batch(batch: dict, device: str | None = None) -> tuple[Tensor, Tensor]:
    observations = Tensor(batch['boards'], device=device).clone().realize()
    actions = Tensor(batch['moves'], device=device).clone().realize()
    return observations, actions

def prepare_bc_batch(batch: dict, device: str | None = None) -> tuple[Tensor, Tensor, Tensor]:
    observations, actions = prepare_batch(batch, device)
    rewards = Tensor(batch['rewards'], device=device).clone().realize()
    return observations, actions, rewards

def collate(samples: list[dict]) -> dict:
    batch = {
        'boards': np.stack([sample['boards'] for sample in samples]),
        'moves': np.stack([sample['moves'] for sample in samples]),
    }
    if 'rewards' in samples[0]:
        batch['rewards'] = np.stack([sample['rewards'] for sample in samples])
    return batch

def _windows(sample: dict, window_size: int | None):
    boards = np.asarray(sample['boards'], dtype=np.int8)
    moves = np.asarray(sample['moves'], dtype=np.int16)
    rewards = np.asarray(sample['rewards'], dtype=np.float32) if 'rewards' in sample else None
    if window_size is None or len(moves) == window_size:
        window = {'boards': boards, 'moves': moves}
        if rewards is not None:
            if not np.isfinite(rewards).all():
                return
            window['rewards'] = rewards
        elif 'result' in sample:
            window['rewards'] = np.full(len(moves), sample['result'] * 2 - 1, dtype=np.float32)
        yield window
        return
    for i in range(len(moves) - window_size + 1):
        window = {
            'boards': boards[i:i + window_size + 1],
            'moves': moves[i:i + window_size],
        }
        if rewards is not None:
            window_rewards = rewards[i:i + window_size]
            if not np.isfinite(window_rewards).all():
                continue
            window['rewards'] = window_rewards
        elif 'result' in sample:
            window['rewards'] = np.full(window_size, sample['result'] * 2 - 1, dtype=np.float32)
        yield window

def batches(dataset, batch_size: int, window_size: int | None = None):
    batch = []
    for sample in dataset:
        for window in _windows(sample, window_size):
            batch.append(window)
            if len(batch) == batch_size:
                yield collate(batch)
                batch = []

def load_training_dataset(path: str, split: str = 'train', streaming: bool = True,
                          shuffle: bool = True, buffer_size: int = 10000, **kwargs):
    dataset = load_dataset(path, split=split, streaming=streaming, **kwargs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
    return dataset

def load_train_eval_datasets(path: str, train_split: str = 'train', eval_split: str = 'test',
                             streaming: bool = True, shuffle: bool = True,
                             buffer_size: int = 10000, **kwargs):
    train = load_training_dataset(path, train_split, streaming, shuffle, buffer_size, **kwargs)
    eval_dataset = load_training_dataset(path, eval_split, streaming, False, buffer_size, **kwargs)
    return train, eval_dataset

def make_train_step(encoder, predictor, optimizer, jit: bool = True, **loss_kwargs):
    def step(observations: Tensor, actions: Tensor):
        return train_step(encoder, predictor, optimizer, observations, actions, **loss_kwargs)
    return TinyJit(step) if jit else step

def _metrics(losses: tuple[Tensor, Tensor, Tensor, Tensor]) -> dict[str, float]:
    names = ('loss', 'teacher', 'rollout', 'sigreg')
    return {name: loss.realize().item() for name, loss in zip(names, losses)}

def _bc_metrics(losses: tuple[Tensor, Tensor, Tensor]) -> dict[str, float]:
    names = ('loss', 'action', 'reward')
    return {name: loss.realize().item() for name, loss in zip(names, losses)}

def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(item[key] for item in items) / len(items) for key in items[0]}

def evaluate(encoder, predictor, dataset, batch_size: int, steps: int = 100,
             device: str | None = None, **loss_kwargs) -> dict[str, float]:
    metrics = []
    window_size = loss_kwargs.get('context_window', 3) + loss_kwargs.get('rollout_steps', 2) - 1
    with Tensor.train(False):
        for step, batch in enumerate(batches(dataset, batch_size, window_size), start=1):
            observations, actions = prepare_batch(batch, device)
            metrics.append(_metrics(lewm_loss(encoder, predictor, observations, actions, **loss_kwargs)))
            if step >= steps:
                break
    if not metrics:
        raise ValueError('evaluation dataset produced no full batches')
    return _mean_metrics(metrics)

def evaluate_bc(encoder, heads, dataset, batch_size: int, steps: int = 100,
                device: str | None = None, **loss_kwargs) -> dict[str, float]:
    metrics = []
    window_size = loss_kwargs.get('mtp_steps', 8)
    with Tensor.train(False):
        for step, batch in enumerate(batches(dataset, batch_size, window_size), start=1):
            observations, actions, rewards = prepare_bc_batch(batch, device)
            metrics.append(_bc_metrics(bc_loss(encoder, heads, observations, actions, rewards, **loss_kwargs)))
            if step >= steps:
                break
    if not metrics:
        raise ValueError('evaluation dataset produced no full batches')
    return _mean_metrics(metrics)

def save_checkpoint(path: str | Path, encoder, predictor, optimizer, step: int, metrics: dict[str, float]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    state.update(get_state_dict(encoder, 'encoder.'))
    state.update(get_state_dict(predictor, 'predictor.'))
    state.update(get_state_dict(optimizer, 'optimizer.'))
    safe_save(state, str(path), metadata={'step': step, **metrics})

def _load_prefixed(obj, state: dict[str, Tensor], prefix: str):
    prefix_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if prefix_state:
        load_state_dict(obj, prefix_state, strict=False, verbose=False)

def load_checkpoint(path: str | Path, encoder, predictor, optimizer=None):
    state = safe_load(path)
    _load_prefixed(encoder, state, 'encoder.')
    _load_prefixed(predictor, state, 'predictor.')
    if optimizer is not None:
        _load_prefixed(optimizer, state, 'optimizer.')

def save_bc_checkpoint(path: str | Path, encoder, heads, optimizer, step: int, metrics: dict[str, float]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    state.update(get_state_dict(encoder, 'encoder.'))
    state.update(get_state_dict(heads, 'heads.'))
    state.update(get_state_dict(optimizer, 'optimizer.'))
    safe_save(state, str(path), metadata={'step': step, **metrics})

def load_bc_checkpoint(path: str | Path, encoder, heads, optimizer=None):
    state = safe_load(path)
    _load_prefixed(encoder, state, 'encoder.')
    _load_prefixed(heads, state, 'heads.')
    if optimizer is not None:
        _load_prefixed(optimizer, state, 'optimizer.')

def fit(encoder, predictor, optimizer, train_dataset, eval_dataset=None, batch_size: int = 64,
        steps: int = 10000, log_every: int = 50, eval_every: int = 500,
        eval_steps: int = 100, checkpoint_every: int = 1000,
        checkpoint_dir: str | Path = 'checkpoints', device: str | None = None,
        jit: bool = True, **loss_kwargs):
    train = make_train_step(encoder, predictor, optimizer, jit=jit, **loss_kwargs)
    window_size = loss_kwargs.get('context_window', 3) + loss_kwargs.get('rollout_steps', 2) - 1
    start_time = time.perf_counter()
    recent = []
    metrics = None

    for step, batch in enumerate(batches(train_dataset, batch_size, window_size), start=1):
        observations, actions = prepare_batch(batch, device)
        metrics = _metrics(train(observations, actions))
        recent.append(metrics)

        if step % log_every == 0:
            avg = _mean_metrics(recent)
            recent = []
            elapsed = time.perf_counter() - start_time
            LOGGER.info(
                'step=%d/%d loss=%.6f teacher=%.6f rollout=%.6f sigreg=%.6f steps_per_sec=%.2f',
                step, steps, avg['loss'], avg['teacher'], avg['rollout'], avg['sigreg'], step / elapsed
            )

        if eval_dataset is not None and step % eval_every == 0:
            eval_metrics = evaluate(encoder, predictor, eval_dataset, batch_size, eval_steps, device, **loss_kwargs)
            LOGGER.info(
                'eval step=%d loss=%.6f teacher=%.6f rollout=%.6f sigreg=%.6f',
                step, eval_metrics['loss'], eval_metrics['teacher'], eval_metrics['rollout'], eval_metrics['sigreg']
            )

        if step % checkpoint_every == 0:
            save_checkpoint(Path(checkpoint_dir) / f'step_{step}.safetensors', encoder, predictor, optimizer, step, metrics)
            LOGGER.info('checkpoint saved step=%d path=%s', step, Path(checkpoint_dir) / f'step_{step}.safetensors')

        if step >= steps:
            save_checkpoint(Path(checkpoint_dir) / 'last.safetensors', encoder, predictor, optimizer, step, metrics)
            return metrics

    if metrics is None:
        raise ValueError('training dataset produced no full batches')
    save_checkpoint(Path(checkpoint_dir) / 'last.safetensors', encoder, predictor, optimizer, step, metrics)
    return metrics

def fit_bc(encoder, heads, optimizer, train_dataset, eval_dataset=None, batch_size: int = 64,
           steps: int = 10000, log_every: int = 50, eval_every: int = 500,
           eval_steps: int = 100, checkpoint_every: int = 1000,
           checkpoint_dir: str | Path = 'checkpoints_bc', device: str | None = None,
           jit: bool = True, **loss_kwargs):
    train = make_bc_train_step(encoder, heads, optimizer, jit=jit, **loss_kwargs)
    window_size = loss_kwargs.get('mtp_steps', 8)
    start_time = time.perf_counter()
    recent = []
    metrics = None

    for step, batch in enumerate(batches(train_dataset, batch_size, window_size), start=1):
        observations, actions, rewards = prepare_bc_batch(batch, device)
        metrics = _bc_metrics(train(observations, actions, rewards))
        recent.append(metrics)

        if step % log_every == 0:
            avg = _mean_metrics(recent)
            recent = []
            elapsed = time.perf_counter() - start_time
            LOGGER.info(
                'bc step=%d/%d loss=%.6f action=%.6f reward=%.6f steps_per_sec=%.2f',
                step, steps, avg['loss'], avg['action'], avg['reward'], step / elapsed
            )

        if eval_dataset is not None and step % eval_every == 0:
            eval_metrics = evaluate_bc(encoder, heads, eval_dataset, batch_size, eval_steps, device, **loss_kwargs)
            LOGGER.info(
                'bc eval step=%d loss=%.6f action=%.6f reward=%.6f',
                step, eval_metrics['loss'], eval_metrics['action'], eval_metrics['reward']
            )

        if step % checkpoint_every == 0:
            path = Path(checkpoint_dir) / f'step_{step}.safetensors'
            save_bc_checkpoint(path, encoder, heads, optimizer, step, metrics)
            LOGGER.info('bc checkpoint saved step=%d path=%s', step, path)

        if step >= steps:
            save_bc_checkpoint(Path(checkpoint_dir) / 'last.safetensors', encoder, heads, optimizer, step, metrics)
            return metrics

    if metrics is None:
        raise ValueError('training dataset produced no full batches')
    save_bc_checkpoint(Path(checkpoint_dir) / 'last.safetensors', encoder, heads, optimizer, step, metrics)
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--bc-dataset')
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--eval-split', default='test')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--resume')
    parser.add_argument('--phase', choices=('world', 'bc'), default='world')
    parser.add_argument('--world-checkpoint')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--device')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--enc-depth', type=int, default=12)
    parser.add_argument('--pred-depth', type=int, default=6)
    parser.add_argument('--enc-heads', type=int, default=3)
    parser.add_argument('--pred-heads', type=int, default=16)
    parser.add_argument('--context-window', type=int, default=3)
    parser.add_argument('--rollout-steps', type=int, default=2)
    parser.add_argument('--lam-sigreg', type=float, default=0.1)
    parser.add_argument('--lam-rollout', type=float, default=1.0)
    parser.add_argument('--sigreg-m', type=int, default=1024)
    parser.add_argument('--sigreg-n-quad', type=int, default=32)
    parser.add_argument('--mtp-steps', type=int, default=8)
    parser.add_argument('--reward-bins', type=int, default=255)
    parser.add_argument('--lam-reward', type=float, default=1.0)
    parser.add_argument('--train-encoder-in-bc', action='store_true')
    parser.add_argument('--no-jit', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(checkpoint_dir / 'training.log'),
        ],
    )

    dataset_path = args.bc_dataset if args.phase == 'bc' and args.bc_dataset else args.dataset
    if args.phase == 'bc' and args.bc_dataset is None:
        LOGGER.warning('bc phase is using --dataset because --bc-dataset was not provided')
    train_dataset, eval_dataset = load_train_eval_datasets(
        dataset_path,
        train_split=args.train_split,
        eval_split=args.eval_split,
        buffer_size=args.buffer_size,
    )
    encoder = Encoder(dim=args.dim, depth=args.enc_depth, n_heads=args.enc_heads)
    predictor = Predictor(dim=args.dim, depth=args.pred_depth, n_heads=args.pred_heads)

    if args.phase == 'world':
        optimizer = nn.optim.Adam(get_parameters(encoder) + get_parameters(predictor), lr=args.lr)
        if args.resume:
            load_checkpoint(args.resume, encoder, predictor, optimizer)
            LOGGER.info('checkpoint loaded path=%s', args.resume)
        fit(
            encoder,
            predictor,
            optimizer,
            train_dataset,
            eval_dataset=eval_dataset,
            batch_size=args.batch_size,
            steps=args.steps,
            log_every=args.log_every,
            eval_every=args.eval_every,
            eval_steps=args.eval_steps,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=checkpoint_dir,
            device=args.device,
            jit=not args.no_jit,
            M=args.sigreg_m,
            n_quad=args.sigreg_n_quad,
            lam_sigreg=args.lam_sigreg,
            lam_rollout=args.lam_rollout,
            context_window=args.context_window,
            rollout_steps=args.rollout_steps,
        )
        return

    heads = WorldModelHeads(dim=args.dim, mtp_steps=args.mtp_steps, reward_bins=args.reward_bins)
    params = get_parameters(heads)
    if args.train_encoder_in_bc:
        params = get_parameters(encoder) + params
    optimizer = nn.optim.Adam(params, lr=args.lr)
    if args.world_checkpoint:
        load_checkpoint(args.world_checkpoint, encoder, predictor)
        LOGGER.info('world checkpoint loaded path=%s', args.world_checkpoint)
    if args.resume:
        load_bc_checkpoint(args.resume, encoder, heads, optimizer)
        LOGGER.info('bc checkpoint loaded path=%s', args.resume)
    fit_bc(
        encoder,
        heads,
        optimizer,
        train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        steps=args.steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        jit=not args.no_jit,
        mtp_steps=args.mtp_steps,
        reward_bins=args.reward_bins,
        lam_reward=args.lam_reward,
        freeze_encoder=not args.train_encoder_in_bc,
    )

if __name__ == '__main__':
    main()
