import argparse
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import io
import logging
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import urlopen

from huggingface_hub import HfApi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rust_pgn_reader_python_binding import parse_games_from_strings
import zstandard as zstd

LOGGER = logging.getLogger(__name__)
DEFAULT_PGN_DIR = Path('data/pgn')
DEFAULT_OUT_DIR = Path('data/games')
DEFAULT_STOCKFISH_OUT_DIR = Path('data/stockfish')
HF_GAMES_REPO = 'luc/lechess-games'
HF_STOCKFISH_REPO = 'luc/lechess-stockfish'
DEFAULT_SHARD_GAMES = 100000
EVAL_RE = re.compile(r'\[%eval\s+([-+]?\d+(?:\.\d+)?)')
MOVE_ONE_RE = re.compile(r'\b1\.(?:\s|\.)')
RESULTS = {'1-0': 1.0, '0-1': 0.0, '1/2-1/2': 0.5}
Counts = dict[str, int | float]

class ReplayBoard:
    def __init__(self) -> None:
        self.squares = np.zeros(64, dtype=np.int8)

        self.squares[0:8] = [4, 2, 3, 5, 6, 3, 2, 4]
        self.squares[8:16] = 1
        self.squares[48:56] = -1
        self.squares[56:64] = [-4, -2, -3, -5, -6, -3, -2, -4]

        self.turn = 1
        self.ep = -1

    def array(self) -> np.ndarray:
        out = np.zeros((8, 8), dtype=np.int8)

        for sq, piece in enumerate(self.squares):
            if piece:
                out[7 - sq // 8, sq % 8] = abs(int(piece)) + (0 if piece > 0 else 6)

        return out

    def push(self, frm: int, to: int, promo: int) -> None:
        piece = self.squares[frm]

        if abs(int(piece)) == 1 and to == self.ep and not self.squares[to] and frm % 8 != to % 8:
            self.squares[to - 8 * self.turn] = 0

        self.squares[to] = self.turn * promo if promo else piece
        self.squares[frm] = 0

        if abs(int(piece)) == 6 and abs(to - frm) == 2:
            rook_from, rook_to = (7, 5) if to == 6 else (0, 3) if to == 2 else (63, 61) if to == 62 else (56, 59)
            self.squares[rook_to], self.squares[rook_from] = self.squares[rook_from], 0

        self.ep = frm + 8 * self.turn if abs(int(piece)) == 1 and abs(to - frm) == 16 else -1
        self.turn *= -1

def boards_from_moves(moves) -> np.ndarray:
    board = ReplayBoard()
    boards = [board.array()]

    for frm, to, promo in np.asarray(moves, dtype=np.int16):
        board.push(int(frm), int(to), int(promo))
        boards.append(board.array())

    return np.stack(boards)

def _pgn_strings(path: str) -> Iterator[str]:
    current = []
    raw = urlopen(path, timeout=300) if path.startswith(('http://', 'https://')) else open(path, 'rb')
    stream = zstd.ZstdDecompressor().stream_reader(raw) if path.endswith('.zst') else raw
    f = io.TextIOWrapper(stream, encoding='utf-8', errors='replace')

    with f:
        for line in f:
            if line.startswith('[Event ') and current:
                yield ''.join(current)
                current = []
            current.append(line)

    if current:
        yield ''.join(current)

def _header_int(headers: dict, key: str) -> int:
    try:
        return int(headers.get(key, 0))
    except ValueError:
        return 0

def _initial_eval(text: str) -> float | None:
    first_move = MOVE_ONE_RE.search(text)
    prefix = text[:first_move.start()] if first_move else text
    match = EVAL_RE.search(prefix)

    return float(match.group(1)) if match else None

def _record(game, text: str) -> dict | None:
    headers = game.headers
    result = RESULTS.get(headers.get('Result', '*'))

    if result is None or headers.get('Variant', 'Standard') not in ('Standard', 'Chess') or not game.is_valid:
        return None

    promotions = np.asarray(game.promotions, dtype=np.int16)
    promotions = np.where(promotions < 0, 0, promotions)
    moves = np.stack([
        np.asarray(game.from_squares, dtype=np.int16),
        np.asarray(game.to_squares, dtype=np.int16),
        promotions,
    ], axis=1)

    if len(moves) == 0:
        return None

    evals = np.asarray(game.evals, dtype=np.float32)
    initial = _initial_eval(text)
    rewards = np.full(len(evals), np.nan, dtype=np.float32)

    if len(evals):
        prev = np.empty(len(evals), dtype=np.float32)
        prev[0] = 0.0 if initial is None else initial
        prev[1:] = evals[:-1]
        valid = np.isfinite(evals) & np.isfinite(prev)
        side = np.where(np.arange(len(evals))[valid] % 2 == 0, 1, -1)
        rewards[valid] = (evals[valid] - prev[valid]) * side

    key = hashlib.blake2b(digest_size=8)
    key.update('\n'.join([
        headers.get('Site', ''),
        headers.get('Event', ''),
        headers.get('UTCDate', ''),
        headers.get('UTCTime', ''),
        headers.get('White', ''),
        headers.get('Black', ''),
        headers.get('WhiteElo', ''),
        headers.get('BlackElo', ''),
        headers.get('Result', ''),
    ]).encode())
    key.update(moves.tobytes())

    return {
        'game_hash': key.hexdigest(),
        'game_id': headers.get('Site', ''),
        'white_elo': _header_int(headers, 'WhiteElo'),
        'black_elo': _header_int(headers, 'BlackElo'),
        'time_control': headers.get('TimeControl', ''),
        'eco': headers.get('ECO', ''),
        'opening': headers.get('Opening', ''),
        'termination': headers.get('Termination', ''),
        'utc_date': headers.get('UTCDate', ''),
        'utc_time': headers.get('UTCTime', ''),
        'moves': moves,
        'plies': len(moves),
        'result': result,
        'stockfish_ready': len(evals) == len(moves) and np.isfinite(evals).all() and np.isfinite(rewards).all(),
        'eval_cp': np.nan_to_num(evals * 100).astype(np.int32),
        'rewards': rewards,
    }

def _parse_pgn_records(path: str, test_ratio: float = 0.05,
                       parse_threads: int | None = None) -> Iterator[tuple[str, dict]]:
    batch = []

    for text in _pgn_strings(path):
        batch.append(text)

        if len(batch) >= 4096:
            for text, game in zip(batch, parse_games_from_strings(
                batch, num_threads=parse_threads, store_comments=False, store_legal_moves=False
            )):
                record = _record(game, text)

                if record is not None:
                    value = int(record['game_hash'], 16) / 2 ** 64
                    yield 'test' if value < test_ratio else 'train', record

            batch = []

    if batch:
        for text, game in zip(batch, parse_games_from_strings(
            batch, num_threads=parse_threads, store_comments=False, store_legal_moves=False
        )):
            record = _record(game, text)

            if record is not None:
                value = int(record['game_hash'], 16) / 2 ** 64
                yield 'test' if value < test_ratio else 'train', record

def _write_shard(rows: list[dict], out_dir: Path, split: str, source_id: str, shard_id: int,
                 include_text_meta: bool, include_stockfish: bool = False) -> Path:
    path = out_dir / split / f'{split}-{source_id}-{shard_id:06d}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)

    lengths = np.fromiter((len(row['moves']) for row in rows), dtype=np.int32, count=len(rows))
    offsets = np.empty(len(rows) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    moves = np.concatenate([row['moves'] for row in rows]) if offsets[-1] else np.empty((0, 3), dtype=np.int16)
    columns = {
        'game_hash': pa.array([row['game_hash'] for row in rows], type=pa.string()),
        'white_elo': pa.array([row['white_elo'] for row in rows], type=pa.int32()),
        'black_elo': pa.array([row['black_elo'] for row in rows], type=pa.int32()),
        'plies': pa.array([row['plies'] for row in rows], type=pa.int32()),
        'result': pa.array([row['result'] for row in rows], type=pa.float32()),
        'moves': pa.ListArray.from_arrays(
            pa.array(offsets),
            pa.FixedSizeListArray.from_arrays(pa.array(moves.reshape(-1), type=pa.int16()), 3),
        ),
    }

    if include_stockfish:
        for key, dtype in (('eval_cp', np.int32), ('rewards', np.float32)):
            lengths = np.fromiter((len(row[key]) for row in rows), dtype=np.int32, count=len(rows))
            offsets = np.empty(len(rows) + 1, dtype=np.int32)
            offsets[0] = 0
            np.cumsum(lengths, out=offsets[1:])
            values = np.concatenate([row[key] for row in rows]) if offsets[-1] else np.array([], dtype=dtype)

            columns[key] = pa.ListArray.from_arrays(
                pa.array(offsets), pa.array(values, type=pa.from_numpy_dtype(dtype))
            )

    if include_text_meta:
        for key in ('game_id', 'time_control', 'eco', 'opening', 'termination', 'utc_date', 'utc_time'):
            columns[key] = pa.array([row[key] for row in rows], type=pa.string())

    pq.write_table(pa.table(columns), path, compression='zstd')

    return path

def _export_pgn_file(
    args: tuple[str, str, str | None, int, float, bool, int, int, int | None, int]
) -> Counts:
    path, out_dir, stockfish_out_dir, shard_games, test_ratio, include_text_meta, min_elo, writer_threads, parse_threads, log_every = args
    out_dir = Path(out_dir)
    stockfish_out_dir = Path(stockfish_out_dir) if stockfish_out_dir else None

    source_name = Path(path).name.removesuffix('.zst').removesuffix('.pgn')
    source_name = re.sub(r'[^a-zA-Z0-9_.-]+', '-', source_name).strip('-') or 'pgn'
    source_id = f'{source_name}-{hashlib.blake2b(str(Path(path).resolve()).encode(), digest_size=4).hexdigest()}'

    buffers = {'train': [], 'test': []}
    stockfish_buffers = {'train': [], 'test': []}
    shard_ids = {'train': 0, 'test': 0}
    stockfish_shard_ids = {'train': 0, 'test': 0}
    counts: Counts = {'train': 0, 'test': 0, 'stockfish_train': 0, 'stockfish_test': 0}

    pending = []
    started = last_log = time.perf_counter()
    last_games = 0

    def submit(writer, rows, directory, split, shard_id, include_stockfish):
        pending.append(writer.submit(
            _write_shard, rows, directory, split, source_id, shard_id, include_text_meta, include_stockfish
        ))

        if len(pending) >= writer_threads * 2:
            done, waiting = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:
                future.result()

            pending[:] = list(waiting)

    with ThreadPoolExecutor(max_workers=writer_threads) as writer:
        for split, record in _parse_pgn_records(path, test_ratio, parse_threads):
            row = {
                'game_hash': record['game_hash'],
                'white_elo': record['white_elo'],
                'black_elo': record['black_elo'],
                'plies': record['plies'],
                'result': record['result'],
                'moves': record['moves'],
            }

            if include_text_meta:
                row.update({
                    'game_id': record['game_id'],
                    'time_control': record['time_control'],
                    'eco': record['eco'],
                    'opening': record['opening'],
                    'termination': record['termination'],
                    'utc_date': record['utc_date'],
                    'utc_time': record['utc_time'],
                })

            buffers[split].append(row)
            counts[split] += 1

            if len(buffers[split]) >= shard_games:
                submit(writer, buffers[split], out_dir, split, shard_ids[split], False)
                buffers[split] = []
                shard_ids[split] += 1

            strong_enough = record['white_elo'] >= min_elo and record['black_elo'] >= min_elo

            if stockfish_out_dir is not None and record['stockfish_ready'] and strong_enough:
                stockfish_buffers[split].append({**row, 'eval_cp': record['eval_cp'], 'rewards': record['rewards']})
                counts[f'stockfish_{split}'] += 1

                if len(stockfish_buffers[split]) >= shard_games:
                    submit(writer, stockfish_buffers[split], stockfish_out_dir, split, stockfish_shard_ids[split], True)
                    stockfish_buffers[split] = []
                    stockfish_shard_ids[split] += 1

            games = counts['train'] + counts['test']

            if log_every and games - last_games >= log_every:
                now = time.perf_counter()

                LOGGER.info(
                    'progress path=%s games=%d fine=%d elapsed=%.1fs rate=%.1f/s interval=%.1f/s',
                    path, games, counts['stockfish_train'] + counts['stockfish_test'],
                    now - started, games / max(now - started, 1e-9), (games - last_games) / max(now - last_log, 1e-9)
                )

                last_log = now
                last_games = games

        for split, rows in buffers.items():
            if rows:
                submit(writer, rows, out_dir, split, shard_ids[split], False)

        if stockfish_out_dir is not None:
            for split, rows in stockfish_buffers.items():
                if rows:
                    submit(writer, rows, stockfish_out_dir, split, stockfish_shard_ids[split], True)

        for future in pending:
            future.result()

    elapsed = time.perf_counter() - started
    games = counts['train'] + counts['test']
    fine = counts['stockfish_train'] + counts['stockfish_test']
    counts['seconds'] = elapsed

    LOGGER.info(
        'exported path=%s games=%d fine=%d train=%d test=%d stockfish_train=%d stockfish_test=%d elapsed=%.1fs rate=%.1f/s',
        path, games, fine, counts['train'], counts['test'], counts['stockfish_train'], counts['stockfish_test'],
        elapsed, games / max(elapsed, 1e-9)
    )

    return counts

def export_game_shards(pgn_paths: list[str], out_dir: str | Path = DEFAULT_OUT_DIR, shard_games: int = DEFAULT_SHARD_GAMES,
                       test_ratio: float = 0.05, include_text_meta: bool = False,
                       stockfish_out_dir: str | Path | None = DEFAULT_STOCKFISH_OUT_DIR,
                       min_elo: int = 2000, workers: int | None = None, writer_threads: int | None = None,
                       parse_threads: int | None = None,
                       log_every_games: int = 100000) -> Counts:
    started = time.perf_counter()
    out_dir = Path(out_dir)
    stockfish_out_dir = Path(stockfish_out_dir) if stockfish_out_dir else None

    if not pgn_paths:
        raise ValueError('no PGN files to export')

    if shard_games <= 0 or (workers is not None and workers <= 0) or (writer_threads is not None and writer_threads <= 0):
        raise ValueError('shard_games, writer_threads and workers must be positive')

    if parse_threads is not None and parse_threads <= 0:
        raise ValueError('parse_threads must be positive')

    cores = cpu_count()
    active_files = min(len(pgn_paths), cores)
    parse_threads = parse_threads or max(1, cores // active_files)
    workers = workers or min(len(pgn_paths), max(1, cores // parse_threads))
    writer_threads = writer_threads or max(1, min(4, cores // workers))

    jobs = [
        (path, str(out_dir), str(stockfish_out_dir) if stockfish_out_dir else None,
         shard_games, test_ratio, include_text_meta, min_elo, writer_threads, parse_threads, log_every_games)
        for path in pgn_paths
    ]

    main_file = getattr(sys.modules.get('__main__'), '__file__', '')

    if workers > 1 and len(jobs) > 1 and main_file and not main_file.startswith('<'):
        with Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_export_pgn_file, jobs))
    else:
        results = [_export_pgn_file(job) for job in jobs]

    counts: Counts = {
        key: sum(int(result[key]) for result in results)
        for key in ('train', 'test', 'stockfish_train', 'stockfish_test')
    }
    elapsed = time.perf_counter() - started
    games = counts['train'] + counts['test']
    fine = counts['stockfish_train'] + counts['stockfish_test']
    counts['seconds'] = elapsed

    LOGGER.info('export total games=%d fine=%d elapsed=%.1fs rate=%.1f/s', games, fine, elapsed, games / max(elapsed, 1e-9))

    return counts

def push_datasets_to_hub(games_dir: str | Path, stockfish_dir: str | Path | None,
                         private: bool = False, revision: str = 'main') -> None:
    if stockfish_dir is None:
        raise ValueError('stockfish dataset output is disabled, cannot push both datasets')

    targets = [
        (Path(games_dir), os.environ.get('LECHESS_HF_GAMES_REPO', HF_GAMES_REPO)),
        (Path(stockfish_dir), os.environ.get('LECHESS_HF_STOCKFISH_REPO', HF_STOCKFISH_REPO)),
    ]

    for path, repo in targets:
        if not list(path.glob('train/*.parquet')) and not list(path.glob('test/*.parquet')):
            raise ValueError(f'no parquet shards found for {repo} in {path}')

    api = HfApi()

    for path, repo in targets:
        LOGGER.info('pushing dataset repo=%s path=%s', repo, path)
        api.create_repo(repo, repo_type='dataset', private=private, exist_ok=True)
        api.upload_large_folder(
            repo_id=repo,
            repo_type='dataset',
            folder_path=path,
            revision=revision,
            allow_patterns=['train/*.parquet', 'test/*.parquet'],
            private=private,
        )
        LOGGER.info('pushed dataset repo=%s', repo)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', nargs='*')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--shard-games', type=int, default=DEFAULT_SHARD_GAMES)
    parser.add_argument('--test-ratio', type=float, default=0.05)
    parser.add_argument('--include-text-meta', action='store_true')
    parser.add_argument('--stockfish-out-dir', default=DEFAULT_STOCKFISH_OUT_DIR)
    parser.add_argument('--min-elo', type=int, default=2000)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--writer-threads', type=int)
    parser.add_argument('--parse-threads', type=int)
    parser.add_argument('--log-every-games', type=int, default=100000)
    parser.add_argument('--push-to-hub', action='store_true')
    parser.add_argument('--hf-private', action='store_true')
    parser.add_argument('--hf-revision', default='main')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    pgn_paths = args.pgn or sorted(str(path) for pattern in ('*.pgn', '*.pgn.zst') for path in DEFAULT_PGN_DIR.glob(pattern))

    if not pgn_paths:
        raise FileNotFoundError(f'no PGN files passed and none found in {DEFAULT_PGN_DIR}')

    counts = export_game_shards(
        pgn_paths,
        args.out_dir,
        shard_games=args.shard_games,
        test_ratio=args.test_ratio,
        include_text_meta=args.include_text_meta,
        stockfish_out_dir=args.stockfish_out_dir,
        min_elo=args.min_elo,
        workers=args.workers,
        writer_threads=args.writer_threads,
        parse_threads=args.parse_threads,
        log_every_games=args.log_every_games,
    )

    LOGGER.info('export complete train=%d test=%d stockfish_train=%d stockfish_test=%d out_dir=%s stockfish_out_dir=%s',
                counts['train'], counts['test'], counts['stockfish_train'], counts['stockfish_test'], args.out_dir, args.stockfish_out_dir)

    if args.push_to_hub:
        push_datasets_to_hub(
            args.out_dir,
            args.stockfish_out_dir,
            private=args.hf_private,
            revision=args.hf_revision,
        )

if __name__ == '__main__':
    main()
