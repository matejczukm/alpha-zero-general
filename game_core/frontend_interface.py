from .triangular_mode import frontend_interface as tri_fi
from .triangular_mode import data_conversions as tri_dc
from .hexagonal_mode import frontend_interface as hex_fi
from .hexagonal_mode import data_conversions as hex_dc
import sys

sys.path.append('..')
from alpha_zero_general.PackitAIPlayer import AIPlayer

ai_players = {}


def start_new_game(board_size, mode, ai_mode, ai_starts):
    if not ai_mode or not ai_starts:
        return start_game(
            board_size=board_size,
            mode=mode
        )

    model_name = mode + str(board_size)
    if model_name not in ai_players:
        ai_players[model_name] = AIPlayer(board_size, mode)
    ai_player = ai_players[model_name]
    if mode == 'triangular':
        board = tri_fi.get_board(board_size)
        move = ai_player.mcts_get_action(board, 1)
        board = tri_dc.convert_numpy_array_to_triangle(board)
        move = tri_dc.convert_numpy_array_to_triangle(move)
        return perform_move(
            board=board,
            move=move,
            turn=2,
            mode=mode
        )
    board = hex_fi.get_board(board_size)
    move = ai_player.mcts_get_action(board, 1)
    board = hex_dc.convert_numpy_board_to_list(board)
    move = hex_dc.convert_numpy_board_to_list(move)
    return perform_move(
        board=board,
        move=move,
        turn=2,
        mode=mode
    )


def confirm_move(board, move, turn, mode, ai_mode):
    turn = int(turn)
    if not ai_mode:
        return perform_move(
            board=board,
            move=move,
            turn=turn,
            mode=mode
        )

    board_size = len(board[-1]) if mode == 'hexagonal' else len(board)
    model_name = mode + str(board_size)
    if model_name not in ai_players:
        ai_players[model_name] = AIPlayer(board_size, mode)
    ai_player = ai_players[model_name]
    if mode == 'triangular':
        board_np = tri_dc.convert_triangle_to_numpy_array(board).astype(bool).astype(int)
        move_np = tri_dc.convert_triangle_to_numpy_array(move).astype(bool).astype(int)
        board_np = board_np + move_np
        next_move = ai_player.mcts_get_action(board_np, turn)
        board = tri_dc.convert_numpy_array_to_triangle(board_np)
        next_move = tri_dc.convert_numpy_array_to_triangle(next_move)
        return perform_move(
            board=board,
            move=next_move,
            turn=turn + 1,
            mode=mode
        )

    board_np = hex_dc.convert_list_board_to_numpy(board, 1).astype(bool).astype(int)
    move_np = hex_dc.convert_list_board_to_numpy(move).astype(bool).astype(int)
    board_np = board_np + move_np
    next_move = ai_player.mcts_get_action(board_np, turn)
    board = hex_dc.convert_numpy_board_to_list(board_np)
    next_move = hex_dc.convert_numpy_board_to_list(next_move)
    return perform_move(
        board=board,
        move=next_move,
        turn=turn + 1,
        mode=mode
    )


def start_game(board_size, mode):
    if mode == 'triangular':
        return tri_fi.start_game(board_size)
    elif mode == 'hexagonal':
        return hex_fi.start_game(board_size)
    else:
        raise ValueError(f"Invalid game mode: '{mode}'. Supported modes are 'triangular' and 'hexagonal'.")


def perform_move(board, move, turn, mode):
    if mode == 'triangular':
        return tri_fi.perform_move(board, move, turn)
    elif mode == 'hexagonal':
        return hex_fi.perform_move(board, move, turn)
    else:
        raise ValueError(f"Invalid game mode: '{mode}'. Supported modes are 'triangular' and 'hexagonal'.")
