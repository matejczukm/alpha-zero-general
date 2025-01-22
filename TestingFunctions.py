from PackitArena import Arena
from PackitAIPlayer import AIPlayer

def runGames(num, players, aliases):
    assert isinstance(players, list), "Players should be provided as list of player objects"
    assert isinstance(aliases, list), "Aliases should be provided as list"
    assert len(players) == len(aliases)

    g= players[0].game
    num_players = len(players)
    results = [[(0,0) for _ in range(num_players)] for _ in range(num_players)]


    for i in range(num_players):
        for j in range(i+1, num_players):
            arena = Arena(players[i].get_action_for_arena, players[j].get_action_for_arena, game=g)
            p1, d1, p2, d2 = arena.playGames(num,verbose=False,technical_output=True)

            results[i][j] = (p1, d2)
            results[j][i] = (p2, d1)
    
    wrs = [0 for _ in range(num_players)]
    total_games = (num_players-1)*num
    for k in range(num_players):
        wins_otp = sum([tup[0] for tup in results[k]])
        wins_otd = sum([row[k][1] for row in results])
        total_wins = wins_otp+wins_otd
        wrs[k] = total_wins/total_games


    print_results(results, aliases, wrs)

    return results, wrs
            
def print_results(results, aliases, wrs):

    n=len(results)
    col_widths = [
        max(len(str(value)) for value in col) 
        for col in zip(*results, aliases, aliases)
    ]
    header_width = max(len("Player 1 \\ Player 2"), max(len(name) for name in aliases))
    col_widths = [header_width] + col_widths 

    header = ["Player 1 \\ Player 2"] + aliases
    print(" | ".join(f"{name:<{col_widths[i]}}" for i, name in enumerate(header)))


    print("-" * sum(col_widths) + "-" * (3 * n))

 
    for row_name, row in zip(aliases, results):
        formatted_row = [row_name] + row
        print(" | ".join(f"{str(value):<{col_widths[i]}}" for i, value in enumerate(formatted_row)))

    print('\n')

    header_wr = ["Player", "Win ratio"]

    alias_width = max(len(str(alias)) for alias in aliases + [header_wr[0]])
    number_width = max(len(str(number)) for number in wrs + [header_wr[1]])

    print(f"{header_wr[0]:<{alias_width}} | {header_wr[1]:<{number_width}}")
    print("-" * (alias_width + number_width + 3))

    for alias, number in zip(aliases, wrs):
        print(f"{alias:<{alias_width}} | {number:<{number_width}}")









