import argparse

import chess
import numpy as np
from tinygrad import Tensor

from model import Encoder, WorldModelHeads
from training import checkpoint

PIECE_TO_PROMO = {
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
}

def board_array(board: chess.Board) -> np.ndarray:
    out = np.zeros((8, 8), dtype=np.int8)

    for square, piece in board.piece_map().items():
        out[7 - chess.square_rank(square), chess.square_file(square)] = piece.piece_type + (0 if piece.color else 6)

    return out

def pick(logits: np.ndarray, legal: set[int], temperature: float) -> int:
    masked = np.full_like(logits, -np.inf, dtype=np.float64)
    masked[list(legal)] = logits[list(legal)]

    if temperature <= 0:
        return int(masked.argmax())

    probs = np.exp((masked - np.nanmax(masked)) / temperature)
    probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))

def move_from_parts(board: chess.Board, frm: int, to: int, promo: int) -> chess.Move:
    promotion = None

    for piece_type, value in PIECE_TO_PROMO.items():
        if promo == value:
            promotion = piece_type
            break

    move = chess.Move(frm, to, promotion)

    if move in board.legal_moves:
        return move

    return chess.Move(frm, to)

def legal_promos(board: chess.Board, frm: int, to: int) -> set[int]:
    values = {
        PIECE_TO_PROMO[move.promotion] if move.promotion else 0
        for move in board.legal_moves
        if move.from_square == frm and move.to_square == to
    }

    return values or {0}

def choose_move(board: chess.Board, encoder, heads, device: str | None, temperature: float) -> chess.Move:
    boards = Tensor(board_array(board).reshape(1, 1, 8, 8), device=device).realize()
    cls, _ = encoder(boards.reshape(1, 8, 8))
    h = heads.action_hidden(cls.reshape(1, 1, -1))

    legal = list(board.legal_moves)
    legal_from = {move.from_square for move in legal}
    from_logits = heads.from_heads[0](h).realize().numpy()[0, 0]
    frm = pick(from_logits, legal_from, temperature)

    legal_to = {move.to_square for move in legal if move.from_square == frm}
    frm_tensor = Tensor([[frm]], device=device)
    to_logits = heads.to_heads[0](h + heads.from_embed(frm_tensor)).realize().numpy()[0, 0]
    to = pick(to_logits, legal_to, temperature)

    promos = legal_promos(board, frm, to)
    to_tensor = Tensor([[to]], device=device)
    promo_logits = heads.promo_heads[0](h + heads.from_embed(frm_tensor) + heads.to_embed(to_tensor)).realize().numpy()[0, 0]
    promo = pick(promo_logits, promos, temperature)

    move = move_from_parts(board, frm, to, promo)
    assert move in board.legal_moves

    return move

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--fen', default=chess.STARTING_FEN)
    parser.add_argument('--plies', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--device')
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--enc-depth', type=int, default=12)
    parser.add_argument('--enc-heads', type=int, default=3)
    parser.add_argument('--enc-mlp-ratio', type=float, default=4.0)
    parser.add_argument('--enc-proj-dim', type=int)
    parser.add_argument('--enc-proj-depth', type=int, default=1)
    parser.add_argument('--enc-proj-hidden-dim', type=int)
    parser.add_argument('--head-hidden-dim', type=int)
    parser.add_argument('--action-head-depth', type=int, default=2)
    parser.add_argument('--reward-head-depth', type=int, default=2)
    parser.add_argument('--action-head-hidden-dim', type=int)
    parser.add_argument('--reward-head-hidden-dim', type=int)
    parser.add_argument('--mtp-steps', type=int, default=8)
    parser.add_argument('--reward-bins', type=int, default=255)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    board = chess.Board(args.fen)
    encoder = Encoder(
        dim=args.dim,
        depth=args.enc_depth,
        n_heads=args.enc_heads,
        mlp_ratio=args.enc_mlp_ratio,
        proj_dim=args.enc_proj_dim,
        proj_depth=args.enc_proj_depth,
        proj_hidden_dim=args.enc_proj_hidden_dim,
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
    )

    checkpoint(args.checkpoint, {'encoder.': encoder, 'heads.': heads})

    for _ in range(args.plies):
        if board.is_game_over():
            break

        move = choose_move(board, encoder, heads, args.device, args.temperature)
        print(move.uci())
        board.push(move)

    print(board.fen())

if __name__ == '__main__':
    main()
