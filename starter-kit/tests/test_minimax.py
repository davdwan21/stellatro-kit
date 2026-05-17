import random
import time
import unittest
from collections.abc import Callable
from unittest.mock import patch

from bots.minimax_bot import MinimaxBot
from bots.random_bot import SimpleBot
from stellatro_game import Game, JOKER_HAND_SIZE, Phase, PlayerTurn


TEST_JOKER_POOL_SIZE = 32
TEST_ROUNDS = 10
TEST_TIMING_SEED = 100
TEST_TIMING_MAX_DEPTH = 2
MAX_DRAFT_MOVE_SECONDS = 0.2
MAX_PLAY_MOVE_SECONDS = 0.1

BotFactory = Callable[[], object]


def run_single_game(seed: int) -> tuple[int, int]:
    random.seed(seed)
    game = Game()
    minimax_bot = MinimaxBot()
    simple_bot = SimpleBot()
    with patch("builtins.print"):
        game.start_round()
        game.jokers = game.jokers[:TEST_JOKER_POOL_SIZE]

        while game.phase == Phase.DRAFT:
            state = game.get_game_state()
            if state.current_turn == PlayerTurn.PLAYER1:
                draft_index = minimax_bot.pick_joker(state)
                ok, _ = game.step(1, action=draft_index)
                if not ok:
                    raise AssertionError("Minimax draft move was rejected.")
                continue

            picked_index = simple_bot.pick_joker(state)
            ok, _ = game.step(2, action=picked_index)
            if not ok:
                raise AssertionError("Simple bot draft move was rejected.")

        if game.phase != Phase.PLAY:
            raise AssertionError("Game did not reach play phase.")

        state = game.get_game_state()
        play_indices = minimax_bot.pick_hand(state)
        ok, _ = game.step(1, hand_list=play_indices)
        if not ok:
            raise AssertionError("Minimax play move was rejected.")

        state = game.get_game_state()
        if state.phase != Phase.PLAY or state.current_turn != PlayerTurn.PLAYER2:
            raise AssertionError("Game did not hand over to Player 2 for play.")

        ok, final_state = game.step(2, hand_list=simple_bot.pick_hand(state))
        if not ok:
            raise AssertionError("Simple bot play move was rejected.")

    if final_state.phase != Phase.OVER:
        raise AssertionError("Game did not end after both players played.")

    return final_state.player1_score, final_state.player2_score


def run_match_series(rounds: int = TEST_ROUNDS, seed_base: int = 0) -> dict[str, int]:
    results = {"rounds": rounds, "minimax_wins": 0, "simple_wins": 0, "ties": 0}

    for round_number, seed in enumerate(range(seed_base, seed_base + rounds), start=1):
        print(f"[{round_number}/{rounds}] Running seed {seed}...")
        p1_score, p2_score = run_single_game(seed)
        if p1_score > p2_score:
            results["minimax_wins"] += 1
            outcome = "minimax"
        elif p2_score > p1_score:
            results["simple_wins"] += 1
            outcome = "simple"
        else:
            results["ties"] += 1
            outcome = "tie"

        print(
            f"[{round_number}/{rounds}] Complete: "
            f"p1={p1_score} p2={p2_score} winner={outcome}"
        )

    return results


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_game_with_timings(
    player1_factory: BotFactory,
    player2_factory: BotFactory,
    seed: int = TEST_TIMING_SEED,
) -> dict[str, float | int | list[float]]:
    game = Game(verbose=False)
    game.start_round(rng=random.Random(seed))
    game.jokers = game.jokers[:TEST_JOKER_POOL_SIZE]

    player1 = player1_factory()
    player2 = player2_factory()
    player1_draft_times: list[float] = []
    player2_draft_times: list[float] = []
    player1_play_times: list[float] = []
    player2_play_times: list[float] = []
    total_steps = 0

    with patch("builtins.print"):
        while game.phase == Phase.DRAFT:
            state = game.get_game_state()
            player = 1 if state.current_turn == PlayerTurn.PLAYER1 else 2
            bot = player1 if player == 1 else player2

            start = time.perf_counter()
            draft_index = bot.pick_joker(state)
            elapsed = time.perf_counter() - start
            if player == 1:
                player1_draft_times.append(elapsed)
            else:
                player2_draft_times.append(elapsed)

            ok, _ = game.step(player, action=draft_index)
            if not ok:
                raise AssertionError(f"Player {player} draft move was rejected.")
            total_steps += 1

        if game.phase != Phase.PLAY:
            raise AssertionError("Timed game did not reach play phase.")

        state = game.get_game_state()
        start = time.perf_counter()
        play_indices = player1.pick_hand(state)
        player1_play_times.append(time.perf_counter() - start)
        ok, _ = game.step(1, hand_list=play_indices)
        if not ok:
            raise AssertionError("Player 1 play move was rejected.")
        total_steps += 1

        state = game.get_game_state()
        if state.phase != Phase.PLAY or state.current_turn != PlayerTurn.PLAYER2:
            raise AssertionError("Timed game did not hand over to Player 2 for play.")

        start = time.perf_counter()
        play_indices = player2.pick_hand(state)
        player2_play_times.append(time.perf_counter() - start)
        ok, final_state = game.step(2, hand_list=play_indices)
        if not ok:
            raise AssertionError("Player 2 play move was rejected.")
        total_steps += 1

    if final_state.phase != Phase.OVER:
        raise AssertionError("Timed game did not end after both players played.")

    all_draft_times = player1_draft_times + player2_draft_times
    all_play_times = player1_play_times + player2_play_times
    all_move_times = all_draft_times + all_play_times

    return {
        "player1_draft_times": player1_draft_times,
        "player2_draft_times": player2_draft_times,
        "player1_play_times": player1_play_times,
        "player2_play_times": player2_play_times,
        "player1_draft_steps": len(player1_draft_times),
        "player2_draft_steps": len(player2_draft_times),
        "player1_play_steps": len(player1_play_times),
        "player2_play_steps": len(player2_play_times),
        "player1_total_steps": len(player1_draft_times) + len(player1_play_times),
        "player2_total_steps": len(player2_draft_times) + len(player2_play_times),
        "draft_steps": len(all_draft_times),
        "play_steps": len(all_play_times),
        "total_steps": total_steps,
        "player1_average_draft_time": _average(player1_draft_times),
        "player2_average_draft_time": _average(player2_draft_times),
        "player1_average_play_time": _average(player1_play_times),
        "player2_average_play_time": _average(player2_play_times),
        "player1_average_move_time": _average(player1_draft_times + player1_play_times),
        "player2_average_move_time": _average(player2_draft_times + player2_play_times),
        "average_draft_time": _average(all_draft_times),
        "average_play_time": _average(all_play_times),
        "average_move_time": _average(all_move_times),
    }


def run_minimax_mirror_game_with_timings(
    seed: int = TEST_TIMING_SEED,
    max_depth: int = TEST_TIMING_MAX_DEPTH,
) -> dict[str, float | int | list[float]]:
    return run_game_with_timings(
        player1_factory=lambda: MinimaxBot(max_depth=max_depth),
        player2_factory=lambda: MinimaxBot(max_depth=max_depth),
        seed=seed,
    )


class TestMinimaxAgainstSimpleBot(unittest.TestCase):
    def test_minimax_can_play_full_games_against_random_bot(self):
        results = run_match_series(rounds=TEST_ROUNDS, seed_base=100)
        print(
            "Minimax vs Random over {rounds} rounds: "
            "minimax={minimax_wins}, random={random_wins}, ties={ties}".format(**results)
        )

        self.assertEqual(
            results["rounds"],
            results["minimax_wins"] + results["random_wins"] + results["ties"],
        )
        self.assertGreater(results["minimax_wins"], 0)

    def test_minimax_mirror_move_times_stay_within_budget(self):
        timings = run_minimax_mirror_game_with_timings()
        print(
            "Minimax mirror timing: "
            f"draft_steps={timings['draft_steps']} "
            f"play_steps={timings['play_steps']} "
            f"total_steps={timings['total_steps']} "
            f"player1(avg_draft={timings['player1_average_draft_time']:.4f}s "
            f"avg_play={timings['player1_average_play_time']:.4f}s "
            f"avg_move={timings['player1_average_move_time']:.4f}s "
            f"steps={timings['player1_total_steps']}) "
            f"player2(avg_draft={timings['player2_average_draft_time']:.4f}s "
            f"avg_play={timings['player2_average_play_time']:.4f}s "
            f"avg_move={timings['player2_average_move_time']:.4f}s "
            f"steps={timings['player2_total_steps']}) "
            f"overall(avg_draft={timings['average_draft_time']:.4f}s "
            f"avg_play={timings['average_play_time']:.4f}s "
            f"avg_move={timings['average_move_time']:.4f}s)"
        )

        self.assertEqual(timings["player1_draft_steps"], JOKER_HAND_SIZE)
        self.assertEqual(timings["player2_draft_steps"], JOKER_HAND_SIZE)
        self.assertEqual(timings["player1_play_steps"], 1)
        self.assertEqual(timings["player2_play_steps"], 1)
        self.assertEqual(timings["player1_total_steps"], JOKER_HAND_SIZE + 1)
        self.assertEqual(timings["player2_total_steps"], JOKER_HAND_SIZE + 1)
        self.assertEqual(timings["draft_steps"], JOKER_HAND_SIZE * 2)
        self.assertEqual(timings["play_steps"], 2)
        self.assertEqual(timings["total_steps"], (JOKER_HAND_SIZE * 2) + 2)

        for label in ("player1_draft_times", "player2_draft_times"):
            for elapsed in timings[label]:
                self.assertLess(
                    elapsed,
                    MAX_DRAFT_MOVE_SECONDS,
                    f"{label} had a draft move at {elapsed:.3f}s which exceeds the {MAX_DRAFT_MOVE_SECONDS:.3f}s budget.",
                )

        for label in ("player1_play_times", "player2_play_times"):
            for elapsed in timings[label]:
                self.assertLess(
                    elapsed,
                    MAX_PLAY_MOVE_SECONDS,
                    f"{label} had a play move at {elapsed:.3f}s which exceeds the {MAX_PLAY_MOVE_SECONDS:.3f}s budget.",
                )


if __name__ == "__main__":
    summary = run_match_series(rounds=TEST_ROUNDS, seed_base=100)
    print(
        "Minimax vs Random over {rounds} rounds: "
        "minimax={minimax_wins}, random={simple_wins}, ties={ties}".format(**summary)
    )
