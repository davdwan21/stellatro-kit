"""
tune_evaluator.py

Grid-search tuner for SmartBot's three key constants:
  - COMPLETE_ENGINE_BONUS  (stella synergy: already own the other half)
  - HALF_ENGINE_IN_POOL    (stella synergy: other half still draftable)
  - DENIAL_WEIGHT          (fraction of opponent gain used as denial bonus)

Usage:
  python tune_evaluator.py [--rounds N] [--workers N] [--quick]

  --rounds N    Matches per parameter combo (default: 50). More = slower but
                more reliable. 200+ recommended for final tuning.
  --workers N   Parallel workers (default: CPU count). Set to 1 to disable.
  --quick       Run a coarse sweep first, then refine around the best region.
                Recommended for a first run.

Output:
  Prints a results table sorted by win rate.
  Writes full results to tune_results.csv.

Example:
  python tune_evaluator.py --rounds 100 --quick
"""

import argparse
import csv
import importlib
import itertools
import multiprocessing
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Make sure the starter-kit is on the path.
# Adjust this if your directory layout differs.
# ---------------------------------------------------------------------------
STARTER_KIT = Path(__file__).parent.parent
sys.path.insert(0, str(STARTER_KIT))

# We import the game runner lazily inside workers to avoid pickling issues.

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

@dataclass
class ParamGrid:
    complete_engine_bonus: List[float] = field(default_factory=lambda: [
        2_000, 5_000, 8_000, 12_000, 18_000
    ])
    half_engine_in_pool: List[float] = field(default_factory=lambda: [
        500, 1_000, 2_000, 4_000
    ])
    denial_weight: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.4, 0.6, 0.8
    ])

COARSE_GRID = ParamGrid()

# Fine grid — centred on the best coarse result; filled in after coarse sweep.
FINE_GRID = None


def all_combos(grid: ParamGrid) -> List[Tuple]:
    return list(itertools.product(
        grid.complete_engine_bonus,
        grid.half_engine_in_pool,
        grid.denial_weight,
    ))


# ---------------------------------------------------------------------------
# Single-combo worker
# ---------------------------------------------------------------------------

def _patch_smart_evaluator(se, complete_bonus, half_bonus, denial_w):
    """Monkey-patch SmartBot's tunable constants onto an already-imported module."""

    def patched_synergy(candidate_name, my_joker_names, pool_names):
        is_gen = candidate_name in se.STELLA_GENERATORS
        is_con = candidate_name in se.STELLA_CONSUMERS
        if not is_gen and not is_con:
            return 0.0
        dead_penalty = -(half_bonus // 2)
        if is_gen:
            if my_joker_names & se.STELLA_CONSUMERS:  return complete_bonus
            elif pool_names   & se.STELLA_CONSUMERS:  return half_bonus
            else:                                      return dead_penalty
        if is_con:
            if my_joker_names & se.STELLA_GENERATORS: return complete_bonus
            elif pool_names   & se.STELLA_GENERATORS: return half_bonus
            else:                                      return dead_penalty
        return 0.0

    def patched_denial(candidate_name, opp_hand, opp_jokers, candidate_joker):
        opp_baseline, _   = se._best_hand(opp_hand, opp_jokers)
        opp_with_joker, _ = se._best_hand(opp_hand, opp_jokers + [candidate_joker])
        return denial_w * max(0, opp_with_joker - opp_baseline)

    se._stella_synergy_bonus  = patched_synergy
    se._opponent_denial_score = patched_denial


def _run_one_match(game, smart_bot, greedy_bot, smart_is_p1: bool) -> str:
    """
    Drive a single match through the step() API, then score with auto_score().
    Returns 'win', 'loss', or 'draw' from SmartBot's perspective.

    Draft loop: alternate step() calls, each bot picks a joker via pick_joker().
    Play phase:  auto_score() exhaustively finds optimal hands for both players,
                 removing any hand-selection variance from the benchmark.
    """
    from stellatro_common import Phase, PlayerTurn

    p1_bot = smart_bot if smart_is_p1 else greedy_bot
    p2_bot = greedy_bot if smart_is_p1 else smart_bot

    # --- Draft phase ---
    while game.phase == Phase.DRAFT:
        state = game.get_game_state()
        if game.current_turn == PlayerTurn.PLAYER1:
            action = p1_bot.pick_joker(state)
            ok, _ = game.step(1, action=action)
        else:
            action = p2_bot.pick_joker(state)
            ok, _ = game.step(2, action=action)
        if not ok:
            # Invalid action — fall back to index 0
            if game.current_turn == PlayerTurn.PLAYER1:
                game.step(1, action=0)
            else:
                game.step(2, action=0)

    # --- Score both players optimally, removing hand-pick variance ---
    p1_score, p2_score = game.auto_score()

    smart_score  = p1_score if smart_is_p1 else p2_score
    greedy_score = p2_score if smart_is_p1 else p1_score

    if   smart_score > greedy_score: return "win"
    elif smart_score < greedy_score: return "loss"
    else:                            return "draw"


def _run_combo(args):
    """
    Run `rounds` matches of SmartBot(params) vs GreedyBot and return win rate.
    Each worker imports modules fresh to avoid cross-combo contamination.

    We use GameSetup to run each seed twice — once with SmartBot as P1,
    once as P2 — on the same deal. This halves variance from first-mover
    advantage without doubling runtime.
    """
    complete_bonus, half_bonus, denial_w, rounds, seed_offset = args

    import sys, random

    # Fresh import of smart_evaluator so patches don't bleed between processes.
    if "bots.smart_evaluator" in sys.modules:
        del sys.modules["bots.smart_evaluator"]
    import bots.smart_evaluator as se
    _patch_smart_evaluator(se, complete_bonus, half_bonus, denial_w)
    SmartBot = se.SmartBot

    if "bots.greedy_bot" in sys.modules:
        del sys.modules["bots.greedy_bot"]
    import bots.greedy_bot as gb
    GreedyBot = gb.GreedyBot

    from stellatro_game.game import Game, GameSetup

    wins = losses = draws = 0

    for i in range(rounds):
        import random as _random
        rng = _random.Random(seed_offset + i)

        try:
            # Generate a shared deal so both sides play the same cards.
            setup = GameSetup.generate(rng=rng)

            smart  = SmartBot()
            greedy = GreedyBot()

            # Play A: SmartBot is Player 1.
            game_a = Game(verbose=False)
            game_a.load_setup(setup, swap_hands=False)
            result_a = _run_one_match(game_a, smart, greedy, smart_is_p1=True)

            # Play B: SmartBot is Player 2 (same deal, hands swapped).
            game_b = Game(verbose=False)
            game_b.load_setup(setup, swap_hands=True)
            result_b = _run_one_match(game_b, smart, greedy, smart_is_p1=False)

            for result in (result_a, result_b):
                if   result == "win":  wins   += 1
                elif result == "loss": losses += 1
                else:                  draws  += 1

        except Exception:
            # Don't let one bad seed crash the whole sweep.
            losses += 2   # penalise both sub-games

    total    = wins + losses + draws
    win_rate = wins / total if total else 0.0
    return (complete_bonus, half_bonus, denial_w, wins, losses, draws, win_rate)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    grid: ParamGrid,
    rounds: int,
    workers: int,
    seed_offset: int = 0,
    label: str = "Sweep",
) -> List[Tuple]:
    combos = all_combos(grid)
    total  = len(combos)
    print(f"\n{label}: {total} combos × {rounds} rounds each "
          f"({total * rounds} total matches)")
    print(f"Using {workers} parallel worker(s).\n")

    args = [
        (cb, hb, dw, rounds, seed_offset)
        for cb, hb, dw in combos
    ]

    results = []
    t0 = time.time()

    if workers == 1:
        for i, a in enumerate(args, 1):
            r = _run_combo(a)
            results.append(r)
            elapsed = time.time() - t0
            pct = i / total * 100
            print(f"  [{i:3d}/{total}] CE={r[0]:6.0f} HE={r[1]:5.0f} "
                  f"DW={r[2]:.2f}  win={r[6]:.1%}  ({elapsed:.1f}s elapsed)")
    else:
        with multiprocessing.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_combo, args), 1):
                results.append(r)
                elapsed = time.time() - t0
                pct = i / total * 100
                print(f"  [{i:3d}/{total}] CE={r[0]:6.0f} HE={r[1]:5.0f} "
                      f"DW={r[2]:.2f}  win={r[6]:.1%}  ({elapsed:.1f}s elapsed)")

    results.sort(key=lambda x: x[6], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Fine grid builder
# ---------------------------------------------------------------------------

def build_fine_grid(best: Tuple) -> ParamGrid:
    cb, hb, dw = best[0], best[1], best[2]

    def neighbours(val, candidates):
        """Return nearby values centred on val from candidates list."""
        idx = min(range(len(candidates)), key=lambda i: abs(candidates[i] - val))
        lo  = candidates[max(0, idx - 1)]
        hi  = candidates[min(len(candidates) - 1, idx + 1)]
        step_cb = (hi - lo) / 4 if hi != lo else val * 0.1
        return sorted({
            max(0, val - step_cb * 2),
            max(0, val - step_cb),
            val,
            val + step_cb,
            val + step_cb * 2,
        })

    cb_vals = COARSE_GRID.complete_engine_bonus
    hb_vals = COARSE_GRID.half_engine_in_pool

    fine_cb = neighbours(cb, cb_vals)
    fine_hb = neighbours(hb, hb_vals)
    fine_dw = sorted({
        max(0.0, dw - 0.15),
        max(0.0, dw - 0.05),
        dw,
        min(1.0, dw + 0.05),
        min(1.0, dw + 0.15),
    })

    return ParamGrid(
        complete_engine_bonus=fine_cb,
        half_engine_in_pool=fine_hb,
        denial_weight=fine_dw,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

HEADER = (
    "complete_engine_bonus",
    "half_engine_in_pool",
    "denial_weight",
    "wins",
    "losses",
    "draws",
    "win_rate",
)


def print_table(results: List[Tuple], top_n: int = 10):
    print(f"\n{'Rank':>4}  {'CE Bonus':>10}  {'HE Pool':>8}  "
          f"{'Denial W':>8}  {'W':>5}  {'L':>5}  {'D':>5}  {'Win%':>6}")
    print("-" * 62)
    for rank, r in enumerate(results[:top_n], 1):
        cb, hb, dw, w, l, d, wr = r
        print(f"{rank:4d}  {cb:10.0f}  {hb:8.0f}  {dw:8.2f}  "
              f"{w:5d}  {l:5d}  {d:5d}  {wr:6.1%}")


def save_csv(results: List[Tuple], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(results)
    print(f"\nFull results saved to {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tune SmartBot constants.")
    parser.add_argument("--rounds",  type=int, default=50,
                        help="Matches per parameter combo (default 50)")
    parser.add_argument("--workers", type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="Parallel workers (default: CPU count - 1)")
    parser.add_argument("--quick",   action="store_true",
                        help="Coarse sweep then fine sweep around best result")
    parser.add_argument("--output",  default="tune_results.csv",
                        help="CSV output path (default: tune_results.csv)")
    args = parser.parse_args()

    all_results = []

    if args.quick:
        # --- Coarse sweep ---
        coarse_results = run_sweep(
            COARSE_GRID,
            rounds=args.rounds,
            workers=args.workers,
            seed_offset=0,
            label="Coarse sweep",
        )
        print_table(coarse_results, top_n=5)
        all_results.extend(coarse_results)

        # --- Fine sweep around best coarse result ---
        best = coarse_results[0]
        fine_grid = build_fine_grid(best)
        print(f"\nBest coarse result: CE={best[0]:.0f} HE={best[1]:.0f} "
              f"DW={best[2]:.2f}  win={best[6]:.1%}")
        print("Running fine sweep around this point...")

        fine_results = run_sweep(
            fine_grid,
            rounds=args.rounds * 2,   # more rounds for fine sweep
            workers=args.workers,
            seed_offset=10_000,
            label="Fine sweep",
        )
        print_table(fine_results, top_n=5)
        all_results.extend(fine_results)

    else:
        # --- Single full sweep ---
        all_results = run_sweep(
            COARSE_GRID,
            rounds=args.rounds,
            workers=args.workers,
            seed_offset=0,
            label="Full sweep",
        )
        print_table(all_results, top_n=10)

    # Sort combined results and save
    all_results.sort(key=lambda x: x[6], reverse=True)
    save_csv(all_results, args.output)

    best = all_results[0]
    print(f"""
Best constants found:
  COMPLETE_ENGINE_BONUS = {best[0]:.0f}
  HALF_ENGINE_IN_POOL   = {best[1]:.0f}
  DENIAL_WEIGHT         = {best[2]:.2f}
  Win rate vs GreedyBot = {best[6]:.1%}

Copy these into smart_evaluator.py to lock them in.
""")


if __name__ == "__main__":
    main()