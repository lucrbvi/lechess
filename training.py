import argparse
from collections.abc import Iterator
from datetime import datetime
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tinygrad import Tensor, nn
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_load_metadata, safe_save
import trackio

from datagen.data import boards_from_moves
from model import Encoder, Predictor, WorldModelHeads

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET = 'data/games'
DEFAULT_BC_DATASET = 'data/stockfish'
DEFAULT_TRACKIO_PROJECT = 'lechess'
DEFAULT_TRACKIO_SPACE = 'lechess-trackio'
STDOUT_WARMUP_STEPS = 100
STDOUT_EVERY = 2000

def run_name(args: argparse.Namespace) -> str:
    if args.trackio_run:
        return args.trackio_run

    return f'{args.phase}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

def trackio_space(args: argparse.Namespace) -> str | None:
    return f'{args.hf_user}/{DEFAULT_TRACKIO_SPACE}' if args.hf_user else None

def encode(encoder, boards: Tensor) -> Tensor:
    B, T = boards.shape[:2]
    cls, _ = encoder(boards.reshape(B * T, *boards.shape[2:]))

    return cls.reshape(B, T, -1)

def sigreg(z: Tensor, M: int = 1024, n_quad: int = 32, t_min: float = 0.2,
           t_max: float = 4.0, lam: float = 1.0) -> Tensor:
    B, T, D = z.shape

    u = Tensor.randn(D, M, device=z.device, dtype=z.dtype)
    u = u / u.square().sum(0, keepdim=True).sqrt()

    h = z.reshape(B * T, D) @ u
    t = Tensor.linspace(t_min, t_max, n_quad, device=z.device, dtype=z.dtype)
    ht = h.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)

    phi0 = (-t.square() / 2).exp()
    weight = (-t.square() / (2 * lam * lam)).exp()
    diff = (ht.cos().mean(0) - phi0.unsqueeze(0)).square() + ht.sin().mean(0).square()
    integral = weight.unsqueeze(0) * diff
    dt = (t_max - t_min) / (n_quad - 1)

    return ((integral[:, 1:-1].sum(1) * 2 + integral[:, 0] + integral[:, -1]) * dt / 2).mean()

def world_loss(encoder, predictor, boards: Tensor, actions: Tensor,
               context_window: int = 3, lam_sigreg: float = 0.1,
               sigreg_m: int = 1024, sigreg_n_quad: int = 32) -> tuple[Tensor, dict[str, Tensor]]:
    assert boards.shape[1] == context_window + 1

    z = encode(encoder, boards)
    teacher = (predictor(z[:, :-1], actions)[:, -1] - z[:, -1]).square().mean()
    regularizer = sigreg(z, M=sigreg_m, n_quad=sigreg_n_quad)
    total = teacher + lam_sigreg * regularizer

    return total, {'loss': total, 'teacher': teacher, 'sigreg': regularizer}

def bc_loss(encoder, heads, boards: Tensor, actions: Tensor, rewards: Tensor,
            mtp_steps: int = 8, reward_bins: int = 255,
            lam_reward: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
    assert actions.shape[1] >= mtp_steps

    max_start = actions.shape[1] - mtp_steps + 1

    with Tensor.train(False):
        z = encode(encoder, boards[:, :max_start]).detach()

    future_actions = Tensor.stack(*[actions[:, i:i + max_start] for i in range(mtp_steps)], dim=2)
    action_logits, reward_logits = heads(z[:, :max_start], future_actions)

    future_rewards = Tensor.stack(*[rewards[:, i:i + max_start] for i in range(mtp_steps)], dim=2)
    from_logits, to_logits, promo_logits = action_logits

    action = (
        from_logits.reshape(-1, 64).sparse_categorical_crossentropy(future_actions[..., 0].reshape(-1)) +
        to_logits.reshape(-1, 64).sparse_categorical_crossentropy(future_actions[..., 1].reshape(-1)) +
        promo_logits.reshape(-1, 7).sparse_categorical_crossentropy(future_actions[..., 2].reshape(-1))
    )

    bins = Tensor.linspace(-5.0, 5.0, reward_bins, device=rewards.device, dtype=rewards.dtype)
    values = future_rewards.sign() * (future_rewards.abs() + 1).log()
    values = values.clip(-5.0, 5.0)

    below = (values.unsqueeze(-1) >= bins).sum(-1).cast('int32').clip(1, reward_bins - 1) - 1
    above = below + 1
    weight_hi = ((values - bins[below]) / (bins[above] - bins[below])).clip(0, 1)
    targets = below.one_hot(reward_bins) * (1 - weight_hi).unsqueeze(-1) + above.one_hot(reward_bins) * weight_hi.unsqueeze(-1)

    reward = -(targets * reward_logits.log_softmax(-1)).sum(-1).mean()
    total = action + lam_reward * reward

    return total, {'loss': total, 'action': action, 'reward': reward}

def batches(dataset, batch_size: int, window_size: int, phase: str, device: str | None) -> Iterator[dict[str, Tensor]]:
    batch: list[tuple] = []

    for sample in dataset:
        moves = np.asarray(sample['moves'], dtype=np.int16)
        boards = boards_from_moves(moves)
        rewards = np.asarray(sample['rewards'], dtype=np.float32) if 'rewards' in sample else None

        for i in range(len(moves) - window_size + 1):
            if phase == 'bc':
                reward = rewards[i:i + window_size] if rewards is not None else np.full(
                    window_size, sample['result'] * 2 - 1, dtype=np.float32
                )

                if not np.isfinite(reward).all():
                    continue

                batch.append((boards[i:i + window_size + 1], moves[i:i + window_size], reward))
            else:
                batch.append((boards[i:i + window_size + 1], moves[i:i + window_size]))

            if len(batch) == batch_size:
                out = {
                    'boards': Tensor(np.stack([x[0] for x in batch]), device=device).clone().realize(),
                    'moves': Tensor(np.stack([x[1] for x in batch]), device=device).clone().realize(),
                }

                if phase == 'bc':
                    out['rewards'] = Tensor(np.stack([x[-1] for x in batch]), device=device).clone().realize()

                yield out
                batch = []

def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    _, _, metadata = safe_load_metadata(path)

    return metadata.get('__metadata__', {})

def checkpoint(path: str | Path, objects: dict[str, object], optimizer=None, step: int | None = None,
               metrics: dict[str, float] | None = None) -> dict[str, Any] | None:
    if step is None:
        state = safe_load(path)

        for prefix, obj in objects.items():
            weights = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}

            if weights:
                load_state_dict(obj, weights, strict=False, verbose=False)

        if optimizer is not None:
            weights = {k.removeprefix('optimizer.'): v for k, v in state.items() if k.startswith('optimizer.')}

            if weights:
                load_state_dict(optimizer, weights, strict=False, verbose=False)

        return checkpoint_metadata(path)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {}

    for prefix, obj in objects.items():
        state.update(get_state_dict(obj, prefix))

    if optimizer is not None:
        state.update(get_state_dict(optimizer, 'optimizer.'))

    safe_save(state, str(path), metadata={'step': step, **(metrics or {})})

    return None

def start_tracking(args: argparse.Namespace) -> bool:
    config = {
        key: value for key, value in vars(args).items()
        if isinstance(value, str | int | float | bool) or value is None
    }

    trackio.init(
        project=DEFAULT_TRACKIO_PROJECT,
        name=run_name(args),
        group=args.phase,
        space_id=trackio_space(args),
        config=config,
        resume='allow',
        auto_log_gpu=False,
    )

    return True

def should_log_stdout(step: int) -> bool:
    return step <= STDOUT_WARMUP_STEPS or step % STDOUT_EVERY == 0

def run_eval(phase: str, dataset, encoder, predictor, heads, args) -> dict[str, float]:
    window = args.context_window if phase == 'world' else args.mtp_steps
    metrics = []

    with Tensor.train(False):
        for step, batch in enumerate(batches(dataset, args.batch_size, window, phase, args.device), start=1):
            if phase == 'world':
                _, values = world_loss(
                    encoder, predictor, batch['boards'], batch['moves'],
                    args.context_window, args.lam_sigreg, args.sigreg_m, args.sigreg_n_quad,
                )
            else:
                _, values = bc_loss(encoder, heads, batch['boards'], batch['moves'], batch['rewards'],
                                    args.mtp_steps, args.reward_bins, args.lam_reward)

            metrics.append({key: value.realize().item() for key, value in values.items()})

            if step >= args.eval_steps:
                break

    if not metrics:
        raise ValueError('evaluation dataset produced no full batches')

    return {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}

def train(args: argparse.Namespace) -> dict[str, float]:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tracking = start_tracking(args)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()],
    )

    dataset_path = args.bc_dataset if args.phase == 'bc' else args.dataset

    train_dataset = load_dataset(dataset_path, split=args.train_split, streaming=True).shuffle(
        buffer_size=args.buffer_size, seed=42
    )

    try:
        eval_dataset = load_dataset(dataset_path, split=args.eval_split, streaming=True)
    except Exception as exc:
        eval_dataset = None
        LOGGER.warning('eval split disabled path=%s split=%s error=%s', dataset_path, args.eval_split, exc)

    encoder = Encoder(
        dim=args.dim,
        depth=args.enc_depth,
        n_heads=args.enc_heads,
        mlp_ratio=args.enc_mlp_ratio,
        proj_dim=args.enc_proj_dim,
        proj_depth=args.enc_proj_depth,
        proj_hidden_dim=args.enc_proj_hidden_dim,
    )

    predictor = Predictor(
        dim=args.dim,
        depth=args.pred_depth,
        n_heads=args.pred_heads,
        mlp_ratio=args.pred_mlp_ratio,
        dropout=args.pred_dropout,
        proj_dim=args.pred_proj_dim,
        proj_depth=args.pred_proj_depth,
        proj_hidden_dim=args.pred_proj_hidden_dim,
    )

    heads = WorldModelHeads(
        dim=args.dim,
        hidden_dim=args.head_hidden_dim,
        mtp_steps=args.mtp_steps,
        reward_bins=args.reward_bins,
        action_depth=args.action_head_depth,
        reward_depth=args.reward_head_depth,
        action_hidden_dim=args.action_head_hidden_dim,
        reward_hidden_dim=args.reward_head_hidden_dim,
    ) if args.phase == 'bc' else None

    params = get_parameters(encoder) + get_parameters(predictor) if args.phase == 'world' else get_parameters(heads)
    optimizer = nn.optim.Adam(params, lr=args.lr)

    objects: dict[str, object] = {'encoder.': encoder, 'predictor.': predictor}

    if heads is not None:
        objects['heads.'] = heads

    if args.phase == 'bc' and not args.world_checkpoint and not args.resume:
        raise ValueError('phase=bc requires --world-checkpoint or --resume to avoid training heads on a random encoder')

    if args.world_checkpoint:
        checkpoint(args.world_checkpoint, {'encoder.': encoder, 'predictor.': predictor})
        LOGGER.info('world checkpoint loaded path=%s', args.world_checkpoint)

    start_step = 0

    if args.resume:
        metadata = checkpoint(args.resume, objects, optimizer)
        start_step = int(metadata.get('step', 0)) if metadata else 0
        LOGGER.info('checkpoint loaded path=%s step=%d', args.resume, start_step)

    window = args.context_window if args.phase == 'world' else args.mtp_steps
    started = time.perf_counter()

    def world_step(boards: Tensor, moves: Tensor, rewards: Tensor | None = None):
        with Tensor.train():
            optimizer.zero_grad()

            total, values = world_loss(
                encoder, predictor, boards, moves,
                args.context_window, args.lam_sigreg, args.sigreg_m, args.sigreg_n_quad,
            )

            total.backward()
            optimizer.step()

            return values

    def bc_step(boards: Tensor, moves: Tensor, rewards: Tensor | None = None):
        assert rewards is not None

        with Tensor.train():
            optimizer.zero_grad()

            total, values = bc_loss(encoder, heads, boards, moves, rewards,
                                    args.mtp_steps, args.reward_bins, args.lam_reward)

            total.backward()
            optimizer.step()

            return values

    train_step = bc_step if args.phase == 'bc' else world_step

    if not args.no_jit:
        train_step = TinyJit(train_step)

    try:
        for local_step, batch in enumerate(batches(train_dataset, args.batch_size, window, args.phase, args.device), start=1):
            n = start_step + local_step

            metrics = {
                key: value.realize().item()
                for key, value in train_step(batch['boards'], batch['moves'], batch.get('rewards')).items()
            }

            steps_per_sec = local_step / (time.perf_counter() - started)

            if tracking:
                trackio.log({
                    **{f'train/{key}': value for key, value in metrics.items()},
                    'train/steps_per_sec': steps_per_sec,
                }, step=n)

            if should_log_stdout(n):
                LOGGER.info('phase=%s step=%d/%d metrics=%s steps_per_sec=%.2f',
                            args.phase, n, start_step + args.steps, metrics, steps_per_sec)

            if eval_dataset is not None and n % args.eval_every == 0:
                eval_metrics = run_eval(args.phase, eval_dataset, encoder, predictor, heads, args)

                if should_log_stdout(n):
                    LOGGER.info('eval phase=%s step=%d metrics=%s', args.phase, n, eval_metrics)

                if tracking:
                    trackio.log({f'eval/{key}': value for key, value in eval_metrics.items()}, step=n)

            if n % args.checkpoint_every == 0:
                checkpoint(checkpoint_dir / f'step_{n}.safetensors', objects, optimizer, n, metrics)

                if tracking:
                    trackio.log({'checkpoint/step': n}, step=n)

            if local_step >= args.steps:
                checkpoint(checkpoint_dir / 'last.safetensors', objects, optimizer, n, metrics)

                if tracking:
                    trackio.log({f'final/{key}': value for key, value in metrics.items()}, step=n)

                return metrics
    finally:
        if tracking:
            trackio.finish()

    raise ValueError('training dataset produced no full batches')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('world', 'bc'), default='world')
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--bc-dataset', default=DEFAULT_BC_DATASET)
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--eval-split', default='test')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--resume')
    parser.add_argument('--world-checkpoint')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--device')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--enc-depth', type=int, default=12)
    parser.add_argument('--enc-heads', type=int, default=3)
    parser.add_argument('--enc-mlp-ratio', type=float, default=4.0)
    parser.add_argument('--enc-proj-dim', type=int)
    parser.add_argument('--enc-proj-depth', type=int, default=1)
    parser.add_argument('--enc-proj-hidden-dim', type=int)
    parser.add_argument('--pred-depth', type=int, default=6)
    parser.add_argument('--pred-heads', type=int, default=16)
    parser.add_argument('--pred-mlp-ratio', type=float, default=4.0)
    parser.add_argument('--pred-dropout', type=float, default=0.1)
    parser.add_argument('--pred-proj-dim', type=int)
    parser.add_argument('--pred-proj-depth', type=int, default=1)
    parser.add_argument('--pred-proj-hidden-dim', type=int)
    parser.add_argument('--head-hidden-dim', type=int)
    parser.add_argument('--action-head-depth', type=int, default=2)
    parser.add_argument('--reward-head-depth', type=int, default=2)
    parser.add_argument('--action-head-hidden-dim', type=int)
    parser.add_argument('--reward-head-hidden-dim', type=int)
    parser.add_argument('--context-window', type=int, default=3)
    parser.add_argument('--lam-sigreg', type=float, default=0.1)
    parser.add_argument('--sigreg-m', type=int, default=1024)
    parser.add_argument('--sigreg-n-quad', type=int, default=32)
    parser.add_argument('--mtp-steps', type=int, default=8)
    parser.add_argument('--reward-bins', type=int, default=255)
    parser.add_argument('--lam-reward', type=float, default=1.0)
    parser.add_argument('--no-jit', action='store_true')
    parser.add_argument('--run', dest='trackio_run')
    parser.add_argument('--hf-user')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
