from .triangular_mode import frontend_interface as tri_fi
from .triangular_mode import data_conversions as tri_dc
from .hexagonal_mode import frontend_interface as hex_fi
from .hexagonal_mode import data_conversions as hex_dc
from ..alpha_zero_general.PackitAIPlayer import AIPlayer

ai_players = {}


def start_new_game(
        board_size: int,
        mode: str, ai_mode: bool = False,
        ai_starts: bool = False,
        custom_ai_players: dict = None):
    if not ai_mode or not ai_starts:
        return _start_game(
            board_size=board_size,
            mode=mode
        )

    if custom_ai_players:
        # ai_players.update(custom_ai_players)
        for name, player in custom_ai_players.items():
            if isinstance(player, AIPlayer):
                ai_players[name] = player
                print(f'AIPlayer named {name} added.')
            else:
                print('Custom AI Players should be instances of AIPlayer.')

    model_name = mode + str(board_size)
    if model_name not in ai_players:
        ai_players[model_name] = AIPlayer(board_size, mode)
    ai_player = ai_players[model_name]
    if mode == 'triangular':
        board = tri_fi.get_board(board_size)
        move = ai_player.mcts_get_action(board, 1)
        board = tri_dc.convert_numpy_array_to_triangle(board)
        move = tri_dc.convert_numpy_array_to_triangle(move)
        return _perform_move(
            board=board,
            move=move,
            turn=2,
            mode=mode
        )
    board = hex_fi.get_board(board_size)
    move = ai_player.mcts_get_action(board, 1)
    board = hex_dc.convert_numpy_board_to_list(board)
    move = hex_dc.convert_numpy_board_to_list(move)
    return _perform_move(
        board=board,
        move=move,
        turn=2,
        mode=mode
    )


def confirm_move(board: list,
                 move: list,
                 turn: int,
                 mode: str,
                 ai_mode: bool = False):
    if not ai_mode:
        return _perform_move(
            board=board,
            move=move,
            turn=turn,
            mode=mode
        )

    board_size = len(board[0]) if mode == 'hexagonal' else len(board)
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
        return _perform_move(
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
    return _perform_move(
        board=board,
        move=next_move,
        turn=turn + 1,
        mode=mode
    )


def _start_game(board_size, mode):
    if mode == 'triangular':
        return tri_fi.start_game(board_size)
    elif mode == 'hexagonal':
        return hex_fi.start_game(board_size)
    else:
        raise ValueError(f"Invalid game mode: '{mode}'. Supported modes are 'triangular' and 'hexagonal'.")


def _perform_move(board, move, turn, mode):
    if mode == 'triangular':
        return tri_fi.perform_move(board, move, turn)
    elif mode == 'hexagonal':
        return hex_fi.perform_move(board, move, turn)
    else:
        raise ValueError(f"Invalid game mode: '{mode}'. Supported modes are 'triangular' and 'hexagonal'.")
