import io
import chess.pgn
import numpy as np
import zstandard as zstd

from datasets import IterableDataset

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

def board_to_array(board: chess.Board) -> np.ndarray:
    arr = np.zeros((8, 8), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        arr[row, col] = _piece_value(piece)
    return arr

def _parse_game(game: chess.pgn.Game) -> list[dict]:
    board = game.board()
    result = result_to_float(game.headers.get('Result', '*'))
    if result is None:
        return []

    meta = {
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

    samples = []

    for move in game.mainline_moves():
        board_before = board_to_array(board)
        board.push(move)
        board_after = board_to_array(board)

        from_sq, to_sq, promo = _move_to_action(move)
        samples.append({
            'board_before': board_before,
            'move_from': from_sq,
            'move_to': to_sq,
            'promotion': promo,
            'board_after': board_after,
            'result': result,
            **meta,
        })

    return samples

def _parse_pgn_file(path: str):
    with _open(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            samples = _parse_game(game)
            if samples:
                yield samples

def create_dataset(pgn_paths: list[str], shuffle: bool = True, buffer_size: int = 10000):
    def gen():
        for path in pgn_paths:
            for game_samples in _parse_pgn_file(path):
                yield from game_samples

    ds = IterableDataset.from_generator(gen)
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    return ds

def create_windowed_dataset(pgn_paths: list[str], window_size: int,
                            shuffle: bool = True, buffer_size: int = 1000):
    assert window_size >= 1

    def gen():
        for path in pgn_paths:
            for game_samples in _parse_pgn_file(path):
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
                        'termination': win[-1]['termination'],
                        'opening': win[-1]['opening'],
                    }

    ds = IterableDataset.from_generator(gen)
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    return ds
