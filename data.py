import io
import chess.pgn
import numpy as np
import zstandard as zstd

from datasets import IterableDataset

def _open(path: str):
    if path.endswith('.zst'):
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(open(path, 'rb')))
    return open(path)

def _piece_value(piece: chess.Piece) -> int:
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)

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

def _rook_castle_squares(board: chess.Board, move: chess.Move) -> tuple[int, int] | None:
    if not board.is_castling(move):
        return None
    rank = chess.square_rank(move.from_square)
    if chess.square_file(move.to_square) == 6:
        return chess.square(7, rank), chess.square(5, rank)
    else:
        return chess.square(0, rank), chess.square(3, rank)

def _en_passant_captured(board: chess.Board, move: chess.Move) -> int | None:
    if board.is_en_passant(move):
        return chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    return None

class PieceTracker:
    def __init__(self):
        self._ids: dict[int, int] = {}
        self._next_id = 0

    def init_from_board(self, board: chess.Board):
        self._ids.clear()
        for sq, piece in board.piece_map().items():
            self._ids[sq] = self._next_id
            self._next_id += 1

    def apply_move(self, board: chess.Board, move: chess.Move):
        rook = _rook_castle_squares(board, move)
        ep_captured = _en_passant_captured(board, move)

        if rook is not None:
            self._ids[move.to_square] = self._ids.pop(move.from_square)
            self._ids[rook[1]] = self._ids.pop(rook[0])
            return

        captured = ep_captured or move.to_square if board.is_capture(move) else None
        if captured is not None and captured in self._ids:
            del self._ids[captured]

        self._ids[move.to_square] = self._ids.pop(move.from_square)

    def snapshot(self) -> dict[int, int]:
        return dict(self._ids)

    def board_with_ids(self, board: chess.Board) -> np.ndarray:
        arr = np.zeros((8, 8, 2), dtype=np.int16)
        for sq, piece in board.piece_map().items():
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)
            arr[row, col, 0] = _piece_value(piece)
            arr[row, col, 1] = self._ids.get(sq, 0)
        return arr

def board_to_array(board: chess.Board, piece_ids: dict[int, int] | None = None) -> np.ndarray:
    if piece_ids is not None:
        arr = np.zeros((8, 8, 2), dtype=np.int16)
        for sq, piece in board.piece_map().items():
            row = 7 - chess.square_rank(sq)
            col = chess.square_file(sq)
            arr[row, col, 0] = _piece_value(piece)
            arr[row, col, 1] = piece_ids.get(sq, 0)
        return arr
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
        'white_elo': int(game.headers.get('WhiteElo', 0)),
        'black_elo': int(game.headers.get('BlackElo', 0)),
        'time_control': game.headers.get('TimeControl', ''),
        'eco': game.headers.get('ECO', ''),
    }

    pt = PieceTracker()
    pt.init_from_board(board)
    samples = []

    for move in game.mainline_moves():
        ids_before = pt.snapshot()
        board_before = board_to_array(board, ids_before)

        pt.apply_move(board, move)
        board.push(move)

        ids_after = pt.snapshot()
        board_after = board_to_array(board, ids_after)

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
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break
            yield from _parse_game(game)

def create_dataset(pgn_paths: list[str], shuffle: bool = True, buffer_size: int = 10000):
    def gen():
        for path in pgn_paths:
            yield from _parse_pgn_file(path)

    ds = IterableDataset.from_generator(gen)
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    return ds

def create_windowed_dataset(pgn_paths: list[str], window_size: int,
                            shuffle: bool = True, buffer_size: int = 1000):
    assert window_size >= 1

    def gen():
        for path in pgn_paths:
            samples = []
            for s in _parse_pgn_file(path):
                samples.append(s)
                if len(samples) >= window_size + 1:
                    win = samples[-window_size - 1:]
                    boards = np.stack([w['board_before'] for w in win[:-1]] + [win[-1]['board_after']])
                    moves = np.array([[w['move_from'], w['move_to'], w['promotion']] for w in win[:-1]], dtype=np.int16)
                    yield {
                        'boards': boards,
                        'moves': moves,
                        'result': win[-1]['result'],
                        'white_elo': win[-1]['white_elo'],
                        'black_elo': win[-1]['black_elo'],
                    }

    ds = IterableDataset.from_generator(gen)
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)
    return ds
