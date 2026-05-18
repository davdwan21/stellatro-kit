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
  --workers N   Parallel workers (default: CPU count - 1). Use 1 to disable
                multiprocessing (easier to read live logs).
  --quick       Coarse sweep first, then fine sweep around the best region.
                Recommended for a first run.
  --log FILE    Log file path (default: tune_evaluator.log).

Output:
  - Terminal : INFO lines - one per completed combo, plus ETA and live best.
  - Log file : DEBUG lines - every individual round result inside each combo.
  - CSV      : full results table written at the end.

Example:
  python tune_evaluator.py --rounds 50 --quick --workers 1
"""

import argparse
import csv
import itertools
import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Path setup
# tune_evaluator.py lives in starter-kit/bots/, so:
#   parent     = starter-kit/bots
#   parent.parent = starter-kit   <-- the importable root
# ---------------------------------------------------------------------------
STARTER_KIT = Path(__file__).parent.parent
sys.path.insert(0, str(STARTER_KIT))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging(log_path: str) -> None:
    """DEBUG to file, INFO to terminal."""
    fmt     = "%(asctime)s  %(levelname)-7s  %(message)s"
    datefmt = "%H:%M:%S"
    root    = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    return f"{m:02d}m{sec:02d}s"


log = logging.getLogger(__name__)


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


def all_combos(grid: ParamGrid) -> List[Tuple]:
    return list(itertools.product(
        grid.complete_engine_bonus,
        grid.half_engine_in_pool,
        grid.denial_weight,
    ))


# ---------------------------------------------------------------------------
# Monkey-patching helper
# ---------------------------------------------------------------------------

def _patch_smart_evaluator(se, complete_bonus, half_bonus, denial_w) -> None:
    """Overwrite SmartBot's tunable functions with patched versions."""

    def patched_synergy(candidate_name, my_joker_names, pool_names):
        is_gen = candidate_name in se.STELLA_GENERATORS
        is_con = candidate_name in se.STELLA_CONSUMERS
        if not is_gen and not is_con:
            return 0.0
        dead_penalty = -(half_bonus // 2)
        if is_gen:
            if my_joker_names & se.STELLA_CONSUMERS:   return complete_bonus
            elif pool_names   & se.STELLA_CONSUMERS:   return half_bonus
            else:                                       return dead_penalty
        if is_con:
            if my_joker_names & se.STELLA_GENERATORS:  return complete_bonus
            elif pool_names   & se.STELLA_GENERATORS:  return half_bonus
            else:                                       return dead_penalty
        return 0.0

    def patched_denial(candidate_name, opp_hand, opp_jokers, candidate_joker):
        opp_baseline, _   = se._best_hand(opp_hand, opp_jokers)
        opp_with_joker, _ = se._best_hand(opp_hand, opp_jokers + [candidate_joker])
        return denial_w * max(0, opp_with_joker - opp_baseline)

    se._stella_synergy_bonus  = patched_synergy
    se._opponent_denial_score = patched_denial


# ---------------------------------------------------------------------------
# Single match driver
# ---------------------------------------------------------------------------

def _run_one_match(game, smart_bot, greedy_bot, smart_is_p1: bool) -> str:
    """
    Drive draft via step(), score via auto_score().
    Returns 'win', 'loss', or 'draw' from SmartBot's perspective.
    """
    from stellatro_common import Phase, PlayerTurn

    p1_bot = smart_bot  if smart_is_p1 else greedy_bot
    p2_bot = greedy_bot if smart_is_p1 else smart_bot

    while game.phase == Phase.DRAFT:
        state = game.get_game_state()
        if game.current_turn == PlayerTurn.PLAYER1:
            action = p1_bot.pick_joker(state)
            ok, _  = game.step(1, action=action)
            if not ok:
                game.step(1, action=0)
        else:
            action = p2_bot.pick_joker(state)
            ok, _  = game.step(2, action=action)
            if not ok:
                game.step(2, action=0)

    p1_score, p2_score = game.auto_score()
    smart_score  = p1_score if smart_is_p1 else p2_score
    greedy_score = p2_score if smart_is_p1 else p1_score

    if   smart_score > greedy_score: return "win"
    elif smart_score < greedy_score: return "loss"
    else:                            return "draw"


# ---------------------------------------------------------------------------
# Worker - one parameter combo
# ---------------------------------------------------------------------------

def _run_combo(args: Tuple) -> Tuple:
    """
    Run `rounds` matches for one (complete_bonus, half_bonus, denial_w) combo.

    Each round generates a shared deal and plays it twice (SmartBot as P1,
    then P2) to remove first-mover bias.

    Logging:
      DEBUG lines go to the log file for every round.
      The result tuple is returned so the sweep runner can emit INFO lines.
    """
    complete_bonus, half_bonus, denial_w, rounds, seed_offset = args

    import logging as _logging
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path

    # Worker processes spawn fresh interpreters that don't inherit sys.path.
    # Re-insert the starter-kit root before any project imports.
    _starter_kit = str(_Path(__file__).parent.parent)
    if _starter_kit not in _sys.path:
        _sys.path.insert(0, _starter_kit)

    _log = _logging.getLogger(__name__)

    # Fresh imports per process so patches don't bleed between combos
    if "bots.smart_evaluator" in _sys.modules:
        del _sys.modules["bots.smart_evaluator"]
    import bots.smart_evaluator as se
    _patch_smart_evaluator(se, complete_bonus, half_bonus, denial_w)
    SmartBot = se.SmartBot

    if "bots.greedy_bot" in _sys.modules:
        del _sys.modules["bots.greedy_bot"]
    import bots.greedy_bot as gb
    GreedyBot = gb.GreedyBot

    from stellatro_game.game import Game, GameSetup

    wins = losses = draws = 0
    t0   = _time.time()
    tag  = f"CE={complete_bonus:.0f} HE={half_bonus:.0f} DW={denial_w:.2f}"

    _log.debug(f"[{tag}] starting {rounds} rounds")

    for i in range(rounds):
        import random as _random
        rng = _random.Random(seed_offset + i)

        try:
            setup  = GameSetup.generate(rng=rng)
            smart  = SmartBot()
            greedy = GreedyBot()

            # Same deal played twice, sides swapped to cancel first-mover edge
            game_a = Game(verbose=False)
            game_a.load_setup(setup, swap_hands=False)
            result_a = _run_one_match(game_a, smart, greedy, smart_is_p1=True)

            game_b = Game(verbose=False)
            game_b.load_setup(setup, swap_hands=True)
            result_b = _run_one_match(game_b, smart, greedy, smart_is_p1=False)

            for result in (result_a, result_b):
                if   result == "win":  wins   += 1
                elif result == "loss": losses += 1
                else:                  draws  += 1

            # Per-round entry (log file only)
            total_so_far = wins + losses + draws
            wr_so_far    = wins / total_so_far if total_so_far else 0.0
            _log.debug(
                f"[{tag}] round {i+1:3d}/{rounds}  "
                f"A={result_a:<4} B={result_b:<4}  "
                f"running W/L/D={wins}/{losses}/{draws} ({wr_so_far:.1%})  "
                f"elapsed={_time.time()-t0:.1f}s"
            )

        except Exception as exc:
            _log.debug(f"[{tag}] round {i+1} EXCEPTION: {exc}")
            losses += 2

    total    = wins + losses + draws
    win_rate = wins / total if total else 0.0

    _log.debug(
        f"[{tag}] DONE  W/L/D={wins}/{losses}/{draws}  "
        f"win={win_rate:.1%}  elapsed={_time.time()-t0:.1f}s"
    )
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
    combos   = all_combos(grid)
    n_combos = len(combos)

    log.info("=" * 65)
    log.info(f"{label}: {n_combos} combos x {rounds} rounds "
             f"= {n_combos * rounds * 2} total matches")
    log.info(f"Workers: {workers}   Seed offset: {seed_offset}")
    log.info("=" * 65)

    job_args = [
        (cb, hb, dw, rounds, seed_offset)
        for cb, hb, dw in combos
    ]

    results  = []
    best_wr  = -1.0
    best_tag = ""
    t0       = time.time()

    def _on_result(i: int, r: Tuple) -> None:
        nonlocal best_wr, best_tag
        cb, hb, dw, w, l, d, wr = r
        results.append(r)

        elapsed   = time.time() - t0
        per_combo = elapsed / i
        eta       = _fmt_duration(per_combo * (n_combos - i))
        tag       = f"CE={cb:.0f} HE={hb:.0f} DW={dw:.2f}"

        if wr > best_wr:
            best_wr  = wr
            best_tag = tag
            marker   = "  *** NEW BEST ***"
        else:
            marker = ""

        log.info(
            f"[{i:3d}/{n_combos}] {tag}  "
            f"W/L/D={w}/{l}/{d}  win={wr:.1%}  "
            f"ETA={eta}{marker}"
        )

    if workers == 1:
        for i, a in enumerate(job_args, 1):
            _on_result(i, _run_combo(a))
    else:
        with multiprocessing.Pool(workers) as pool:
            for i, r in enumerate(
                pool.imap_unordered(_run_combo, job_args), 1
            ):
                _on_result(i, r)

    results.sort(key=lambda x: x[6], reverse=True)
    elapsed = time.time() - t0
    log.info("=" * 65)
    log.info(f"{label} done in {_fmt_duration(elapsed)}.  "
             f"Best: {best_tag}  win={best_wr:.1%}")
    log.info("=" * 65)
    return results


# ---------------------------------------------------------------------------
# Fine grid builder
# ---------------------------------------------------------------------------

def build_fine_grid(best: Tuple) -> ParamGrid:
    cb, hb, dw = best[0], best[1], best[2]

    def neighbours(val, candidates):
        idx  = min(range(len(candidates)), key=lambda i: abs(candidates[i] - val))
        lo   = candidates[max(0, idx - 1)]
        hi   = candidates[min(len(candidates) - 1, idx + 1)]
        step = (hi - lo) / 4 if hi != lo else val * 0.1
        return sorted({
            max(0, val - step * 2),
            max(0, val - step),
            val,
            val + step,
            val + step * 2,
        })

    fine_cb = neighbours(cb, COARSE_GRID.complete_engine_bonus)
    fine_hb = neighbours(hb, COARSE_GRID.half_engine_in_pool)
    fine_dw = sorted({
        max(0.0, dw - 0.15),
        max(0.0, dw - 0.05),
        dw,
        min(1.0, dw + 0.05),
        min(1.0, dw + 0.15),
    })

    log.info(f"Fine grid  CE={fine_cb}  HE={fine_hb}  DW={fine_dw}")
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


def print_table(results: List[Tuple], top_n: int = 10) -> None:
    log.info(f"{'Rank':>4}  {'CE Bonus':>10}  {'HE Pool':>8}  "
             f"{'Denial W':>8}  {'W':>5}  {'L':>5}  {'D':>5}  {'Win%':>6}")
    log.info("-" * 65)
    for rank, r in enumerate(results[:top_n], 1):
        cb, hb, dw, w, l, d, wr = r
        log.info(
            f"{rank:4d}  {cb:10.0f}  {hb:8.0f}  {dw:8.2f}  "
            f"{w:5d}  {l:5d}  {d:5d}  {wr:6.1%}"
        )


def save_csv(results: List[Tuple], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(results)
    log.info(f"Full results saved to {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
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
    parser.add_argument("--log",     default="tune_evaluator.log",
                        help="Log file path (default: tune_evaluator.log)")
    args = parser.parse_args()

    _setup_logging(args.log)
    log.info(f"Logging DEBUG detail to: {args.log}")
    log.info(f"rounds={args.rounds}  workers={args.workers}  quick={args.quick}")

    all_results: List[Tuple] = []

    if args.quick:
        coarse = run_sweep(
            COARSE_GRID,
            rounds=args.rounds,
            workers=args.workers,
            seed_offset=0,
            label="Coarse sweep",
        )
        log.info("Top 5 coarse results:")
        print_table(coarse, top_n=5)
        all_results.extend(coarse)

        best      = coarse[0]
        fine_grid = build_fine_grid(best)
        log.info(f"Best coarse: CE={best[0]:.0f} HE={best[1]:.0f} "
                 f"DW={best[2]:.2f}  win={best[6]:.1%}")

        fine = run_sweep(
            fine_grid,
            rounds=args.rounds * 2,
            workers=args.workers,
            seed_offset=10_000,
            label="Fine sweep",
        )
        log.info("Top 5 fine results:")
        print_table(fine, top_n=5)
        all_results.extend(fine)

    else:
        all_results = run_sweep(
            COARSE_GRID,
            rounds=args.rounds,
            workers=args.workers,
            seed_offset=0,
            label="Full sweep",
        )
        log.info("Top 10 results:")
        print_table(all_results, top_n=10)

    all_results.sort(key=lambda x: x[6], reverse=True)
    save_csv(all_results, args.output)

    best = all_results[0]
    log.info(
        "\n" + "=" * 65 + "\n"
        "Best constants found:\n"
        f"  COMPLETE_ENGINE_BONUS = {best[0]:.0f}\n"
        f"  HALF_ENGINE_IN_POOL   = {best[1]:.0f}\n"
        f"  DENIAL_WEIGHT         = {best[2]:.2f}\n"
        f"  Win rate vs GreedyBot = {best[6]:.1%}\n"
        "\nCopy these into smart_evaluator.py to lock them in.\n"
        + "=" * 65
    )


if __name__ == "__main__":
    main()