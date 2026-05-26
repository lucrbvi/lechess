import argparse
import hashlib
import io
import logging
import math
from pathlib import Path
import chess.pgn
import numpy as np
import zstandard as zstd

from datasets import Dataset, Features, Sequence, Value
from datasets import IterableDataset

LOGGER = logging.getLogger(__name__)

def _open(path: str):
    if path.endswith('.zst'):
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(open(path, 'rb')), encoding='utf-8', errors='replace')
    return open(path)

def _piece_value(piece: chess.Piece) -> int:
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)

def _header_int(headers: chess.pgn.Headers, key: str) -> int:
    try:
        return int(headers.get(key, 0))
    except ValueError:
        return 0

def result_to_float(result: str) -> float | None:
    if result == '1-0':
        return 1.0
    if result == '0-1':
        return 0.0
    if result == '1/2-1/2':
        return 0.5
    return None

def _move_to_action(move: chess.Move) -> tuple[int, int, int]:
    promo = move.promotion or 0
    return move.from_square, move.to_square, promo

def _score_to_value(score) -> float:
    mate = score.mate()
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    return math.tanh(score.score() / 400)

def _node_eval_white(node) -> tuple[int, int, float] | None:
    score = node.eval()
    if score is None:
        return None
    white_score = score.white()
    mate = white_score.mate()
    if mate is not None:
        return 0, mate, _score_to_value(white_score)
    cp = white_score.score()
    return cp, 0, _score_to_value(white_score)

def _split_key(game: chess.pgn.Game) -> str:
    return game.headers.get('Site') or game.headers.get('Event') or str(game.headers)

def _game_hash(game: chess.pgn.Game) -> str:
    return hashlib.blake2b(_split_key(game).encode(), digest_size=8).hexdigest()

def _game_split(game: chess.pgn.Game, test_ratio: float) -> str:
    h = hashlib.blake2b(_split_key(game).encode(), digest_size=8).digest()
    value = int.from_bytes(h, 'big') / 2 ** 64
    return 'test' if value < test_ratio else 'train'

def board_to_array(board: chess.Board) -> np.ndarray:
    arr = np.zeros((8, 8), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        arr[row, col] = _piece_value(piece)
    return arr

def _game_meta(game: chess.pgn.Game) -> dict:
    return {
        'game_hash': _game_hash(game),
        'game_id': game.headers.get('Site', ''),
        'white_elo': _header_int(game.headers, 'WhiteElo'),
        'black_elo': _header_int(game.headers, 'BlackElo'),
        'time_control': game.headers.get('TimeControl', ''),
        'eco': game.headers.get('ECO', ''),
        'opening': game.headers.get('Opening', ''),
        'termination': game.headers.get('Termination', ''),
        'utc_date': game.headers.get('UTCDate', ''),
        'utc_time': game.headers.get('UTCTime', ''),
    }

def _parse_game_record(game: chess.pgn.Game) -> dict | None:
    board = game.board()
    result = result_to_float(game.headers.get('Result', '*'))
    if result is None:
        return None

    boards = [board_to_array(board)]
    moves = []
    eval_cp = []
    eval_mate = []
    has_eval = []
    rewards = []
    prev_eval = _node_eval_white(game)
    has_stockfish = prev_eval is not None

    node = game
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        actor_is_white = board.turn == chess.WHITE
        moves.append(_move_to_action(move))
        board.push(move)
        boards.append(board_to_array(board))
        current_eval = _node_eval_white(next_node)
        has_stockfish = has_stockfish or current_eval is not None

        if current_eval is None:
            eval_cp.append(0)
            eval_mate.append(0)
            has_eval.append(False)
            rewards.append(np.nan)
        else:
            cp, mate, value = current_eval
            eval_cp.append(cp)
            eval_mate.append(mate)
            has_eval.append(True)
            if prev_eval is None:
                rewards.append(np.nan)
            else:
                delta = value - prev_eval[2]
                rewards.append(delta if actor_is_white else -delta)

        prev_eval = current_eval
        node = next_node

    if not moves:
        return None

    return {
        'boards': np.stack(boards).astype(np.int8),
        'moves': np.array(moves, dtype=np.int16),
        'plies': len(moves),
        'result': result,
        'has_stockfish': has_stockfish,
        'eval_cp': np.array(eval_cp, dtype=np.int32),
        'eval_mate': np.array(eval_mate, dtype=np.int16),
        'has_eval': np.array(has_eval, dtype=np.bool_),
        'rewards': np.array(rewards, dtype=np.float32),
        **_game_meta(game),
    }

def _parse_game(game: chess.pgn.Game) -> list[dict]:
    record = _parse_game_record(game)
    if record is None:
        return []

    samples = []
    for ply, move in enumerate(record['moves']):
        samples.append({
            'board_before': record['boards'][ply],
            'move_from': int(move[0]),
            'move_to': int(move[1]),
            'promotion': int(move[2]),
            'board_after': record['boards'][ply + 1],
            'ply': ply,
            'result': record['result'],
            **{k: record[k] for k in (
                'game_hash', 'game_id', 'white_elo', 'black_elo', 'time_control',
                'eco', 'opening', 'termination', 'utc_date', 'utc_time'
            )},
        })

    return samples

def _parse_pgn_file(path: str, split: str = 'train', test_ratio: float = 0.05):
    with _open(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if _game_split(game, test_ratio) != split:
                continue
            samples = _parse_game(game)
            if samples:
                yield samples

def _parse_pgn_records(path: str, test_ratio: float = 0.05):
    with _open(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            record = _parse_game_record(game)
            if record is not None:
                yield _game_split(game, test_ratio), record

def create_dataset(pgn_paths: list[str], split: str = 'train', test_ratio: float = 0.05,
                   shuffle: bool = True, buffer_size: int = 10000):
    def gen():
        for path in pgn_paths:
            for game_samples in _parse_pgn_file(path, split, test_ratio):
                yield from game_samples

    ds = IterableDataset.from_generator(gen)
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    return ds

def create_windowed_dataset(pgn_paths: list[str], window_size: int, split: str = 'train',
                            test_ratio: float = 0.05,
                            shuffle: bool = True, buffer_size: int = 1000):
    assert window_size >= 1
    assert split in ('train', 'test')

    def gen():
        for path in pgn_paths:
            for game_samples in _parse_pgn_file(path, split, test_ratio):
                for i in range(len(game_samples) - window_size + 1):
                    win = game_samples[i:i + window_size + 1]
                    boards = np.stack([w['board_before'] for w in win[:-1]] + [win[-1]['board_after']])
                    moves = np.array([[w['move_from'], w['move_to'], w['promotion']] for w in win[:-1]], dtype=np.int16)
                    yield {
                        'boards': boards,
                        'moves': moves,
                        'result': win[-1]['result'],
                        'white_elo': win[-1]['white_elo'],
                        'black_elo': win[-1]['black_elo'],
                        'game_id': win[-1]['game_id'],
                        'game_hash': win[-1]['game_hash'],
                        'start_ply': win[0]['ply'],
                        'termination': win[-1]['termination'],
                        'opening': win[-1]['opening'],
                    }

    ds = IterableDataset.from_generator(gen)
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    return ds

def _export_features(include_text_meta: bool, include_stockfish: bool = False) -> Features:
    features = {
        'game_hash': Value('string'),
        'white_elo': Value('int32'),
        'black_elo': Value('int32'),
        'plies': Value('int32'),
        'result': Value('float32'),
        'boards': Sequence(Sequence(Sequence(Value('int8')))),
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
        'boards': record['boards'].tolist(),
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

def _write_shard(rows: list[dict], out_dir: Path, split: str, shard_id: int,
                 include_text_meta: bool, include_stockfish: bool = False) -> Path:
    path = out_dir / split / f'{split}-{shard_id:06d}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=_export_features(include_text_meta, include_stockfish)).to_parquet(path)
    return path

def export_game_shards(pgn_paths: list[str], out_dir: str | Path, shard_games: int = 10000,
                       test_ratio: float = 0.05, include_text_meta: bool = False,
                       stockfish_out_dir: str | Path | None = None) -> dict[str, int]:
    out_dir = Path(out_dir)
    buffers = {'train': [], 'test': []}
    shard_ids = {'train': 0, 'test': 0}
    counts = {'train': 0, 'test': 0}
    stockfish_out_dir = Path(stockfish_out_dir) if stockfish_out_dir else None
    stockfish_buffers = {'train': [], 'test': []}
    stockfish_shard_ids = {'train': 0, 'test': 0}
    stockfish_counts = {'train': 0, 'test': 0}

    for path in pgn_paths:
        for split, record in _parse_pgn_records(path, test_ratio):
            if record['has_stockfish']:
                if stockfish_out_dir is not None and np.isfinite(record['rewards']).any():
                    stockfish_buffers[split].append(_export_row(record, include_text_meta, True))
                    stockfish_counts[split] += 1
                    if len(stockfish_buffers[split]) >= shard_games:
                        _write_shard(
                            stockfish_buffers[split],
                            stockfish_out_dir,
                            split,
                            stockfish_shard_ids[split],
                            include_text_meta,
                            True,
                        )
                        stockfish_buffers[split] = []
                        stockfish_shard_ids[split] += 1
                continue

            buffers[split].append(_export_row(record, include_text_meta))
            counts[split] += 1
            if len(buffers[split]) >= shard_games:
                _write_shard(buffers[split], out_dir, split, shard_ids[split], include_text_meta)
                buffers[split] = []
                shard_ids[split] += 1

    for split, rows in buffers.items():
        if rows:
            _write_shard(rows, out_dir, split, shard_ids[split], include_text_meta)

    if stockfish_out_dir is not None:
        for split, rows in stockfish_buffers.items():
            if rows:
                _write_shard(rows, stockfish_out_dir, split, stockfish_shard_ids[split], include_text_meta, True)

    return {
        'train': counts['train'],
        'test': counts['test'],
        'stockfish_train': stockfish_counts['train'],
        'stockfish_test': stockfish_counts['test'],
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--shard-games', type=int, default=10000)
    parser.add_argument('--test-ratio', type=float, default=0.05)
    parser.add_argument('--include-text-meta', action='store_true')
    parser.add_argument('--stockfish-out-dir')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    counts = export_game_shards(
        args.pgn,
        args.out_dir,
        shard_games=args.shard_games,
        test_ratio=args.test_ratio,
        include_text_meta=args.include_text_meta,
        stockfish_out_dir=args.stockfish_out_dir,
    )
    LOGGER.info(
        'export complete train=%d test=%d stockfish_train=%d stockfish_test=%d out_dir=%s stockfish_out_dir=%s',
        counts['train'], counts['test'], counts['stockfish_train'], counts['stockfish_test'],
        args.out_dir, args.stockfish_out_dir
    )

if __name__ == '__main__':
    main()
