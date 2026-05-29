import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import io
import logging
from multiprocessing import Pool
from pathlib import Path
import re
import sys
import time

import numpy as np
from datasets import Dataset, Features, Sequence, Value, disable_progress_bars
from rust_pgn_reader_python_binding import parse_games_from_strings
import zstandard as zstd

LOGGER = logging.getLogger(__name__)
disable_progress_bars()
DEFAULT_PGN_DIR = Path('data/pgn')
DEFAULT_OUT_DIR = Path('data/games')
DEFAULT_STOCKFISH_OUT_DIR = Path('data/stockfish')
EVAL_RE = re.compile(r'\[%eval\s+([-+]?\d+(?:\.\d+)?)')
MOVE_ONE_RE = re.compile(r'\b1\.(?:\s|\.)')
RESULTS = {'1-0': 1.0, '0-1': 0.0, '1/2-1/2': 0.5}

class ReplayBoard:
    def __init__(self):
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

    def push(self, frm: int, to: int, promo: int):
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

def _open(path: str):
    if path.endswith('.zst'):
        return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(open(path, 'rb')), encoding='utf-8', errors='replace')
    return open(path, encoding='utf-8', errors='replace')

def _pgn_strings(path: str):
    current = []
    with _open(path) as f:
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
    if initial is not None:
        prev = initial
        for i, value in enumerate(evals):
            if np.isfinite(value):
                rewards[i] = (float(value) - prev) * (1 if i % 2 == 0 else -1)
                prev = float(value)
            else:
                prev = np.nan
    key = '\n'.join([
        headers.get('Site', ''),
        headers.get('Event', ''),
        headers.get('UTCDate', ''),
        headers.get('UTCTime', ''),
        headers.get('White', ''),
        headers.get('Black', ''),
        headers.get('WhiteElo', ''),
        headers.get('BlackElo', ''),
        headers.get('Result', ''),
        ' '.join(f'{m[0]}-{m[1]}-{m[2]}' for m in moves),
    ])
    return {
        'game_hash': hashlib.blake2b(key.encode(), digest_size=8).hexdigest(),
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
        'has_stockfish': initial is not None or np.isfinite(evals).any(),
        'eval_cp': np.nan_to_num(evals * 100).astype(np.int32),
        'eval_mate': np.zeros(len(evals), dtype=np.int16),
        'has_eval': np.isfinite(evals),
        'rewards': rewards,
    }

def _parse_pgn_records(path: str, test_ratio: float = 0.05, parse_threads: int | None = None):
    batch = []
    for text in _pgn_strings(path):
        batch.append(text)
        if len(batch) >= 4096:
            yield from _parse_batch(batch, test_ratio, parse_threads)
            batch = []
    if batch:
        yield from _parse_batch(batch, test_ratio, parse_threads)

def _parse_batch(texts: list[str], test_ratio: float, parse_threads: int | None):
    games = parse_games_from_strings(texts, num_threads=parse_threads, store_comments=False, store_legal_moves=False)
    for text, game in zip(texts, games):
        record = _record(game, text)
        if record is not None:
            value = int(record['game_hash'], 16) / 2 ** 64
            yield 'test' if value < test_ratio else 'train', record

def _export_features(include_text_meta: bool, include_stockfish: bool = False) -> Features:
    features = {
        'game_hash': Value('string'),
        'white_elo': Value('int32'),
        'black_elo': Value('int32'),
        'plies': Value('int32'),
        'result': Value('float32'),
        'moves': Sequence(Sequence(Value('int16'))),
    }
    if include_stockfish:
        features.update({
            'eval_cp': Sequence(Value('int32')),
            'eval_mate': Sequence(Value('int16')),
            'has_eval': Sequence(Value('bool')),
            'rewards': Sequence(Value('float32')),
        })
    if include_text_meta:
        features.update({
            'game_id': Value('string'),
            'time_control': Value('string'),
            'eco': Value('string'),
            'opening': Value('string'),
            'termination': Value('string'),
            'utc_date': Value('string'),
            'utc_time': Value('string'),
        })
    return Features(features)

def _export_row(record: dict, include_text_meta: bool, include_stockfish: bool = False) -> dict:
    row = {
        'game_hash': record['game_hash'],
        'white_elo': record['white_elo'],
        'black_elo': record['black_elo'],
        'plies': record['plies'],
        'result': record['result'],
        'moves': record['moves'].tolist(),
    }
    if include_stockfish:
        row.update({
            'eval_cp': record['eval_cp'].tolist(),
            'eval_mate': record['eval_mate'].tolist(),
            'has_eval': record['has_eval'].tolist(),
            'rewards': record['rewards'].tolist(),
        })
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
    return row

def _write_shard(rows: list[dict], out_dir: Path, split: str, source_id: str, shard_id: int,
                 include_text_meta: bool, include_stockfish: bool = False) -> Path:
    path = out_dir / split / f'{split}-{source_id}-{shard_id:06d}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=_export_features(include_text_meta, include_stockfish)).to_parquet(path)
    return path

def _source_id(path: str) -> str:
    name = Path(path).name.removesuffix('.zst').removesuffix('.pgn')
    name = re.sub(r'[^a-zA-Z0-9_.-]+', '-', name).strip('-') or 'pgn'
    digest = hashlib.blake2b(str(Path(path).resolve()).encode(), digest_size=4).hexdigest()
    return f'{name}-{digest}'

def _export_pgn_file(args: tuple[str, str, str | None, int, float, bool, int, int | None, int]) -> dict[str, float]:
    path, out_dir, stockfish_out_dir, shard_games, test_ratio, include_text_meta, min_elo, parse_threads, log_every = args
    out_dir = Path(out_dir)
    stockfish_out_dir = Path(stockfish_out_dir) if stockfish_out_dir else None
    source_id = _source_id(path)
    buffers = {'train': [], 'test': []}
    stockfish_buffers = {'train': [], 'test': []}
    shard_ids = {'train': 0, 'test': 0}
    stockfish_shard_ids = {'train': 0, 'test': 0}
    counts = {'train': 0, 'test': 0, 'stockfish_train': 0, 'stockfish_test': 0}
    pending = []
    started = last_log = time.perf_counter()
    last_games = 0

    def submit(writer, rows, directory, split, shard_id, include_stockfish):
        pending.append(writer.submit(_write_shard, rows, directory, split, source_id, shard_id, include_text_meta, include_stockfish))
        if len(pending) >= 4:
            done, waiting = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            pending[:] = list(waiting)

    with ThreadPoolExecutor(max_workers=2) as writer:
        for split, record in _parse_pgn_records(path, test_ratio, parse_threads):
            buffers[split].append(_export_row(record, include_text_meta))
            counts[split] += 1
            if len(buffers[split]) >= shard_games:
                submit(writer, buffers[split], out_dir, split, shard_ids[split], False)
                buffers[split] = []
                shard_ids[split] += 1

            fully_annotated = record['has_stockfish'] and record['has_eval'].all() and np.isfinite(record['rewards']).all()
            strong_enough = record['white_elo'] >= min_elo and record['black_elo'] >= min_elo
            if stockfish_out_dir is not None and fully_annotated and strong_enough:
                stockfish_buffers[split].append(_export_row(record, include_text_meta, True))
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

def export_game_shards(pgn_paths: list[str], out_dir: str | Path = DEFAULT_OUT_DIR, shard_games: int = 10000,
                       test_ratio: float = 0.05, include_text_meta: bool = False,
                       stockfish_out_dir: str | Path | None = DEFAULT_STOCKFISH_OUT_DIR,
                       min_elo: int = 2000, workers: int = 1, parse_threads: int | None = None,
                       log_every_games: int = 100000) -> dict[str, float]:
    started = time.perf_counter()
    out_dir = Path(out_dir)
    stockfish_out_dir = Path(stockfish_out_dir) if stockfish_out_dir else None
    jobs = [(path, str(out_dir), str(stockfish_out_dir) if stockfish_out_dir else None,
             shard_games, test_ratio, include_text_meta, min_elo, parse_threads, log_every_games) for path in pgn_paths]
    main_file = getattr(sys.modules.get('__main__'), '__file__', '')
    if workers > 1 and len(jobs) > 1 and main_file and not main_file.startswith('<'):
        with Pool(processes=workers) as pool:
            results = pool.map(_export_pgn_file, jobs)
    else:
        results = [_export_pgn_file(job) for job in jobs]
    counts = {key: sum(result[key] for result in results) for key in ('train', 'test', 'stockfish_train', 'stockfish_test')}
    elapsed = time.perf_counter() - started
    games = counts['train'] + counts['test']
    fine = counts['stockfish_train'] + counts['stockfish_test']
    counts['seconds'] = elapsed
    LOGGER.info('export total games=%d fine=%d elapsed=%.1fs rate=%.1f/s', games, fine, elapsed, games / max(elapsed, 1e-9))
    return counts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', nargs='*')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--shard-games', type=int, default=10000)
    parser.add_argument('--test-ratio', type=float, default=0.05)
    parser.add_argument('--include-text-meta', action='store_true')
    parser.add_argument('--stockfish-out-dir', default=DEFAULT_STOCKFISH_OUT_DIR)
    parser.add_argument('--min-elo', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--parse-threads', type=int)
    parser.add_argument('--log-every-games', type=int, default=100000)
    return parser.parse_args()

def main():
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
        parse_threads=args.parse_threads,
        log_every_games=args.log_every_games,
    )
    LOGGER.info('export complete train=%d test=%d stockfish_train=%d stockfish_test=%d out_dir=%s stockfish_out_dir=%s',
                counts['train'], counts['test'], counts['stockfish_train'], counts['stockfish_test'], args.out_dir, args.stockfish_out_dir)

if __name__ == '__main__':
    main()
