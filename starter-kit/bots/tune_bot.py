"""
tune_bot.py — Parameter tuner for bot_v4.

Place this file anywhere. Set STELLATRO_KIT_ROOT and BOT_V4_PATH below.

Usage (run from anywhere):
    python tune_bot.py                            # tune all groups, 40 rounds
    python tune_bot.py --group floor synergy      # tune just scalars (fast)
    python tune_bot.py --group denial --rounds 60 # tune denial curves
    python tune_bot.py --opponent bots.bot2:Bot   # different opponent
    python tune_bot.py --passes 3 --seeds 2       # more thorough

Output: bot_v4_tuned.py written next to bot_v4.py.
"""

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# *** SET THESE TWO PATHS ***
# STELLATRO_KIT_ROOT : root of the repo (the folder containing starter-kit/)
# BOT_V4_PATH        : absolute path to bot_v4.py
# ---------------------------------------------------------------------------
STELLATRO_KIT_ROOT = Path("/Users/davidwan/Documents/GitHub/stellatro-kit")
BOT_V4_PATH        = STELLATRO_KIT_ROOT / "starter-kit" / "bots" / "bot_v4.py"

# Runner is invoked from starter-kit/ as cwd, so bots.* imports work
_RUNNER_ABS    = STELLATRO_KIT_ROOT / "starter-kit" / "scripts" / "run_bot_match.py"
_RUNNER_CWD    = STELLATRO_KIT_ROOT / "starter-kit"

# Candidate bot written here so it's importable as bots.tune_candidate
_CANDIDATE_PATH = STELLATRO_KIT_ROOT / "starter-kit" / "bots" / "tune_candidate.py"
_TUNED_OUT      = BOT_V4_PATH.parent / "bot_v4_tuned.py"

# ---------------------------------------------------------------------------
# Default parameter values — must match bot_v4.py exactly
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "denial_p1_xmult":      [0.30, 0.40, 0.38, 0.32, 0.25],
    "denial_p1_other":      [0.10, 0.25, 0.35, 0.35, 0.30],
    "denial_p2_xmult":      [0.85, 0.90, 0.55, 0.38, 0.28],
    "denial_p2_chip":       [0.60, 0.45, 0.35, 0.30, 0.25],
    "denial_p2_other":      [0.70, 0.45, 0.35, 0.35, 0.30],

    "flush_high_mult":      800.0,
    "flush_high_thresh":    0.6,
    "flush_low_mult":       300.0,
    "flush_low_thresh":     0.4,
    "straight_high_mult":   800.0,
    "straight_high_thresh": 0.8,
    "straight_low_mult":    300.0,
    "straight_low_thresh":  0.6,
    "pair_high_bonus":      600.0,
    "pair_low_bonus":       200.0,
    "face_high_bonus":      550.0,
    "face_low_bonus":       220.0,

    "floor_scale":          1.0,
    "synergy_scale":        1.0,
    "opp_engine_coeff":     0.35,
}

# ---------------------------------------------------------------------------
# Search grid
# ---------------------------------------------------------------------------
GRID: dict[str, Any] = {
    # "curve" means coordinate descent over individual weights (±20%, ±40%)
    "denial_p1_xmult":      "curve",
    "denial_p1_other":      "curve",
    "denial_p2_xmult":      "curve",
    "denial_p2_chip":       "curve",
    "denial_p2_other":      "curve",

    "flush_high_mult":      [400, 600, 800, 1000, 1200],
    "flush_high_thresh":    [0.5, 0.6, 0.7],
    "flush_low_mult":       [150, 220, 300, 400],
    "flush_low_thresh":     [0.3, 0.4, 0.5],
    "straight_high_mult":   [400, 600, 800, 1000, 1200],
    "straight_high_thresh": [0.7, 0.8, 0.9],
    "straight_low_mult":    [150, 220, 300, 400],
    "straight_low_thresh":  [0.5, 0.6, 0.7],
    "pair_high_bonus":      [300, 450, 600, 800, 1000],
    "pair_low_bonus":       [100, 150, 200, 300],
    "face_high_bonus":      [300, 400, 550, 700, 900],
    "face_low_bonus":       [100, 150, 220, 300],

    "floor_scale":          [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "synergy_scale":        [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "opp_engine_coeff":     [0.0, 0.15, 0.25, 0.35, 0.50, 0.65],
}

GROUPS: dict[str, list[str]] = {
    "denial":     ["denial_p1_xmult", "denial_p1_other",
                   "denial_p2_xmult", "denial_p2_chip", "denial_p2_other"],
    "floor":      ["floor_scale"],
    "synergy":    ["synergy_scale"],
    "shape":      ["flush_high_mult", "flush_high_thresh", "flush_low_mult",
                   "flush_low_thresh", "straight_high_mult", "straight_high_thresh",
                   "straight_low_mult", "straight_low_thresh",
                   "pair_high_bonus", "pair_low_bonus",
                   "face_high_bonus", "face_low_bonus"],
    "opp_engine": ["opp_engine_coeff"],
}


# ---------------------------------------------------------------------------
# Bot source patching — rebuilds affected functions wholesale to avoid
# fragile multi-line regex on the original source layout.
# ---------------------------------------------------------------------------

def _fmt_curve(lst: list) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in lst) + "]"


def _patch_bot(source: str, params: dict[str, Any]) -> str:
    p = params

    # ---- Rebuild _denial_weight entirely ----
    new_denial = (
        "def _denial_weight(pick_num: int, is_p1: bool, joker_name: str) -> float:\n"
        "    is_xmult = joker_name in _XMULT_JOKER_NAMES\n"
        "    is_chip  = joker_name in _CHIP_JOKER_NAMES\n\n"
        "    if is_p1:\n"
        f"        if is_xmult:\n"
        f"            weights = {_fmt_curve(p['denial_p1_xmult'])}\n"
        f"        else:\n"
        f"            weights = {_fmt_curve(p['denial_p1_other'])}\n"
        "    else:\n"
        f"        if is_xmult:\n"
        f"            weights = {_fmt_curve(p['denial_p2_xmult'])}\n"
        f"        elif is_chip:\n"
        f"            weights = {_fmt_curve(p['denial_p2_chip'])}\n"
        f"        else:\n"
        f"            weights = {_fmt_curve(p['denial_p2_other'])}\n\n"
        "    return weights[min(pick_num, len(weights) - 1)]"
    )
    source = re.sub(
        r"def _denial_weight\(pick_num.*?return weights\[min\(pick_num.*?\)\]",
        new_denial,
        source,
        flags=re.DOTALL,
    )

    # ---- Floor scale: wrap _joker_floor lookup ----
    fs = p["floor_scale"]
    source = re.sub(
        r"def _joker_floor\(joker_cls: type\) -> float:\n    return _JOKER_FLOOR\.get\([^)]+\)",
        f'def _joker_floor(joker_cls: type) -> float:\n'
        f'    return _JOKER_FLOOR.get(getattr(joker_cls, "name", ""), 0.0) * {fs:.6f}',
        source,
        flags=re.DOTALL,
    )

    # ---- Synergy scale: multiply in _pair_synergy_bonus ----
    ss = p["synergy_scale"]
    # Replace "total += val\n    return total" (unique in the file)
    source = source.replace(
        "total += val\n    return total",
        f"total += val * {ss:.6f}\n    return total",
    )

    # ---- Rebuild _synergy_bonus entirely ----
    new_syn = (
        "def _synergy_bonus(joker_cls: type, cards: List[Card]) -> float:\n"
        '    name = getattr(joker_cls, "name", "")\n'
        "    bonus = 0.0\n\n"
        "    if name in _FLUSH_JOKER_NAMES:\n"
        "        fp = _flush_potential(cards)\n"
        f"        bonus += ({p['flush_high_mult']:.1f} * fp if fp >= {p['flush_high_thresh']:.2f}"
        f" else {p['flush_low_mult']:.1f} * fp if fp >= {p['flush_low_thresh']:.2f} else 0.0)\n\n"
        "    if name in _STRAIGHT_JOKER_NAMES:\n"
        "        sp = _straight_potential(cards)\n"
        f"        bonus += ({p['straight_high_mult']:.1f} * sp if sp >= {p['straight_high_thresh']:.2f}"
        f" else {p['straight_low_mult']:.1f} * sp if sp >= {p['straight_low_thresh']:.2f} else 0.0)\n\n"
        "    if name in _PAIR_JOKER_NAMES:\n"
        "        pp = _pair_potential(cards)\n"
        f"        bonus += ({p['pair_high_bonus']:.1f} if pp >= 2 else {p['pair_low_bonus']:.1f} if pp >= 1 else 0.0)\n\n"
        "    if name in _FACE_JOKER_NAMES:\n"
        "        fc = _face_count(cards)\n"
        f"        bonus += ({p['face_high_bonus']:.1f} if fc >= 3 else {p['face_low_bonus']:.1f} if fc >= 2 else 0.0)\n\n"
        "    return bonus"
    )
    source = re.sub(
        r"def _synergy_bonus\(joker_cls: type.*?return bonus",
        new_syn,
        source,
        flags=re.DOTALL,
    )

    # ---- opp_engine_coeff: replace coefficient (appears twice) ----
    oec = p["opp_engine_coeff"]
    source = re.sub(
        r"\d+\.\d{4} \* _pair_synergy_bonus\(opp_picks",
        f"{oec:.4f} * _pair_synergy_bonus(opp_picks",
        source,
    )

    return source


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def run_match(opponent: str, rounds: int, seed_base: int) -> float:
    """Run a mirrored match. Returns win rate 0–1, or -1.0 on failure."""
    cmd = [
        sys.executable, str(_RUNNER_ABS),
        "bots.tune_candidate:Bot", opponent,
        "--rounds", str(rounds),
        "--mirror",
        "--seed-base", str(seed_base),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_RUNNER_CWD),
        )
        out = result.stdout
        err = result.stderr

        # Primary parse: "bots.tune_candidate:Bot=54 (45.0%)"
        m = re.search(r"tune_candidate:Bot=\d+ \((\d+(?:\.\d+)?)%\)", out)
        if m:
            return float(m.group(1)) / 100.0

        # Fallback: wins / last game index
        m_wins = re.search(r"tune_candidate:Bot=(\d+)", out)
        m_last = re.search(r"\[(\d+)/\d+\]", out.rsplit("\n", 5)[-1] if out else "")
        if m_wins and m_last:
            return int(m_wins.group(1)) / int(m_last.group(1))

        # Nothing parsed — print debug info
        snippet = (out + err)[-600:]
        print(f"  [WARN] could not parse output. Snippet:\n{snippet}", file=sys.stderr)
        return -1.0

    except subprocess.TimeoutExpired:
        print("  [WARN] timed out", file=sys.stderr)
        return -1.0
    except Exception as exc:
        print(f"  [WARN] {exc}", file=sys.stderr)
        return -1.0


# ---------------------------------------------------------------------------
# Coordinate descent helpers
# ---------------------------------------------------------------------------

def _curve_candidates(current: list) -> list:
    """Perturb each weight in a denial curve at ±20% and ±40%."""
    out = []
    for i in range(len(current)):
        for pct in (-0.40, -0.20, +0.20, +0.40):
            new = list(current)
            new[i] = round(max(0.0, min(2.5, current[i] * (1 + pct))), 4)
            if new != list(current):
                out.append(new)
    return out


# Module-level so _eval can reach it after main() sets it
_BOT_SOURCE: str = ""


def _eval(params: dict, opponent: str, rounds: int, seeds: int, seed_base: int) -> float:
    total = 0.0
    for i in range(seeds):
        seed = seed_base + i * 1000
        _CANDIDATE_PATH.write_text(_patch_bot(_BOT_SOURCE, params))
        wr = run_match(opponent, rounds, seed)
        if wr < 0:
            return -1.0
        total += wr
    return total / seeds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _BOT_SOURCE

    parser = argparse.ArgumentParser(description="Coordinate-descent tuner for bot_v4")
    parser.add_argument("--group", nargs="+",
                        choices=list(GROUPS.keys()) + ["all"], default=["all"])
    parser.add_argument("--rounds",    type=int, default=40,
                        help="Rounds per evaluation match")
    parser.add_argument("--seeds",     type=int, default=1,
                        help="Seed bases to average per evaluation (more = less variance)")
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--opponent",  default="bots.fixed_synergy:Bot")
    parser.add_argument("--passes",    type=int, default=2,
                        help="Coordinate descent passes over all groups")
    args = parser.parse_args()

    # Validate paths up front
    for label, path in [("STELLATRO_KIT_ROOT", STELLATRO_KIT_ROOT),
                        ("BOT_V4_PATH", BOT_V4_PATH),
                        ("runner", _RUNNER_ABS)]:
        if not path.exists():
            sys.exit(f"ERROR: {label} not found at {path}\n"
                     f"Edit the path constants at the top of tune_bot.py.")

    _BOT_SOURCE = BOT_V4_PATH.read_text()

    # Param list
    if "all" in args.group:
        param_names = [p for g in GROUPS.values() for p in g]
    else:
        param_names = [p for g in args.group for p in GROUPS[g]]
    seen: set = set()
    param_names = [p for p in param_names if not (p in seen or seen.add(p))]

    print(f"Bot:      {BOT_V4_PATH}")
    print(f"Opponent: {args.opponent}")
    print(f"Rounds per eval: {args.rounds}  Seeds: {args.seeds}  Passes: {args.passes}")
    print(f"Params ({len(param_names)}): {param_names}\n")

    # Smoke test — 4 rounds, confirm runner works
    print("Smoke test (4 rounds)...")
    current_params = copy.deepcopy(DEFAULTS)
    _CANDIDATE_PATH.write_text(_patch_bot(_BOT_SOURCE, current_params))
    smoke = run_match(args.opponent, 4, args.seed_base)
    if smoke < 0:
        sys.exit(
            "\nSmoke test failed. Try running manually to see the error:\n\n"
            f"  cd {_RUNNER_CWD}\n"
            f"  {sys.executable} {_RUNNER_ABS} "
            f"bots.tune_candidate:Bot {args.opponent} "
            f"--rounds 4 --mirror --seed-base {args.seed_base}\n"
        )
    print(f"Smoke test OK — baseline win rate: {smoke:.3f}\n")

    # Coordinate descent
    for pass_num in range(1, args.passes + 1):
        print(f"{'='*60}\nPASS {pass_num}/{args.passes}\n{'='*60}")
        improved_any = False

        for param_name in param_names:
            print(f"\n--- {param_name} ---")
            is_curve = GRID[param_name] == "curve"
            candidates = (
                _curve_candidates(current_params[param_name])
                if is_curve else GRID[param_name]
            )

            baseline_wr = _eval(current_params, args.opponent,
                                 args.rounds, args.seeds, args.seed_base)
            best_wr  = baseline_wr
            best_val = current_params[param_name]
            print(f"  baseline={current_params[param_name]}  →  {baseline_wr:.3f}")

            for candidate in candidates:
                if candidate == current_params[param_name]:
                    continue
                test_params = dict(current_params)
                test_params[param_name] = candidate
                wr = _eval(test_params, args.opponent,
                           args.rounds, args.seeds, args.seed_base)
                label = (f"[{', '.join(str(v) for v in candidate)}]"
                         if is_curve else str(candidate))
                marker = " ✓" if wr > best_wr else ""
                print(f"  {param_name}={label}  →  {wr:.3f}{marker}")
                if wr > best_wr:
                    best_wr  = wr
                    best_val = candidate

            if best_val != current_params[param_name]:
                print(f"  → UPDATED {current_params[param_name]} → {best_val}  "
                      f"(Δ={best_wr - baseline_wr:+.3f})")
                current_params[param_name] = best_val
                improved_any = True
            else:
                print(f"  → no improvement")

        if not improved_any:
            print(f"\nNo improvements found in pass {pass_num}, stopping early.")
            break

    # Output
    print(f"\n{'='*60}\nFINAL CHANGES vs DEFAULTS:")
    any_change = False
    for k, v in current_params.items():
        if v != DEFAULTS[k]:
            print(f"  {k}: {DEFAULTS[k]}  →  {v}")
            any_change = True
    if not any_change:
        print("  (no parameters changed from defaults)")
    print(f"{'='*60}")

    _TUNED_OUT.write_text(_patch_bot(_BOT_SOURCE, current_params))
    print(f"\nTuned bot written to: {_TUNED_OUT}")


if __name__ == "__main__":
    main()