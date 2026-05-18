"""
Asymmetric Denial Bot
=====================
Core philosophy: don't just maximise your own score — cap the opponent's ceiling
by denying the jokers that make their build work, then pick up whatever synergy
remains for yourself.

Draft heuristic (per pick):
    score(joker) = own_gain(joker) * OWN_WEIGHT
                 - denial_value(joker) * DENIAL_WEIGHT

  own_gain      = improvement to my best 5-card score if I take this joker.
  denial_value  = improvement to *their* best 5-card score if *they* took it
                  instead (i.e. what we're saving them from).

  DENIAL_WEIGHT > OWN_WEIGHT so the bot biases toward taking high-impact
  opponent pieces even when they don't directly help us — but only when the
  denial value is genuinely large.  When nothing is worth denying, it falls
  back to pure self-improvement.

Joker taxonomy used in denial scoring
--------------------------------------
The most dangerous opponent jokers are x-mult multipliers (The Duo x2, The
Tribe x3, The Order x3, Flower Pot x3, UC Socially Dead x8, The Trio x2.5).
These are priced with a large flat denial bonus on top of the score delta so
the bot always prioritises taking them off the board, even if they don't help
our hand at all.

After those, conditional +mult jokers (Daring, Witty, Zany, Jolly, Cheeky,
Merry, Lively, Vibrant, Jovial) are the backbone of the greedy/minimax builds
we expect to face.  They get a moderate denial bonus.

Hand selection: identical to minimax/greedy — exact brute-force over all
C(10,5) = 252 subsets.
"""

import math
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Tuple

from stellatro_common import CardModel, GameState, JokerModel, PlayerTurn
from stellatro_game import Card, JOKER_HAND_SIZE, Suit, evaluate_hand, PLAYER_CARDS
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

# How much we value our own score gain vs. denying the opponent.
# Set DENIAL_WEIGHT > OWN_WEIGHT so we lean toward denial when the opponent
# joker is genuinely powerful, but fall back to self-improvement otherwise.
OWN_WEIGHT: float = 1.0
DENIAL_WEIGHT: float = 1.6

# Flat bonus added to the denial score for jokers in each tier.
# These are on top of the raw score-delta so that xMult jokers are always
# prioritised for denial even when they don't help our hand.
XMULT_DENIAL_BONUS: float = 800.0   # x-mult jokers: The Duo, Tribe, Order, etc.
COND_MULT_DENIAL_BONUS: float = 250.0  # conditional +mult jokers
RETRIGGER_DENIAL_BONUS: float = 350.0  # retrigger jokers amplify everything

# ---------------------------------------------------------------------------
# Joker classification
# ---------------------------------------------------------------------------

# xMult jokers — taking these off the board hard-caps the opponent's ceiling.
_XMULT_JOKERS = frozenset({
    "The Duo",
    "The Trio",
    "The Tribe",
    "The Order",
    "UC Socially Dead",
    "Flower Pot",
    "Sun God",        # x1.5 per heart scored — compounds with suit jokers
    "Plasma",         # balances chips/mult — can be huge in balanced builds
})

# Conditional +mult jokers — the bread-and-butter of greedy/minimax builds.
_COND_MULT_JOKERS = frozenset({
    "Jolly Joker",
    "Zany Joker",
    "Cheeky Joker",
    "Witty Joker",
    "Daring Joker",
    "Merry Joker",
    "Jovial Joker",
    "Lively Joker",
    "Vibrant Joker",
    "Half Joker",
    "Blackjack",
    "Six Seven",
    "Student ID",
    "Fibonacci Joker",
})

# Retrigger jokers — dangerous because they multiply every apply_card_phase
# effect.  Especially bad to let the opponent have these alongside xMult jokers.
_RETRIGGER_JOKERS = frozenset({
    "Sock and Buskin",
    "Seltzer",
    "Last Lecture",
    "Stargazing",
    "Encore",
})

# ---------------------------------------------------------------------------
# Helpers shared with greedy/minimax bots
# ---------------------------------------------------------------------------

_JOKER_NAME_TO_CLASS: Dict[str, type] = {
    cls.name: cls for cls in ALL_JOKER_CLASSES
}


def _card_from_model(c: CardModel) -> Card:
    suits = [Suit(s) for s in c.suits]
    card = Card(c.rank, suits[0])
    for s in suits[1:]:
        card.add_suit(s)
    card.scored = c.scored
    card.num_triggers = c.num_triggers
    return card


def _joker_from_model(j: JokerModel):
    return _JOKER_NAME_TO_CLASS.get(j.name, RegularJoker)()


def _hand_for(state: GameState, turn: PlayerTurn) -> List[Card]:
    models = state.player1_hand if turn == PlayerTurn.PLAYER1 else state.player2_hand
    return [_card_from_model(c) for c in models]


def _jokers_for(state: GameState, turn: PlayerTurn):
    models = (
        state.player1_jokers if turn == PlayerTurn.PLAYER1 else state.player2_jokers
    )
    return [_joker_from_model(j) for j in models]


def _best_score(cards: List[Card], jokers: list) -> int:
    """Brute-force best score across all C(n,5) subsets."""
    best = -1
    n = min(PLAYER_CARDS, len(cards))
    for combo in combinations(range(n), 5):
        try:
            s = evaluate_hand(
                [deepcopy(cards[i]) for i in combo],
                deepcopy(jokers),
            )
        except Exception:
            continue
        if s > best:
            best = s
    return max(best, 0)


def _best_indices(cards: List[Card], jokers: list) -> List[int]:
    """Return the indices of the highest-scoring 5-card subset."""
    best_score = -1
    best_combo: Tuple[int, ...] = tuple(range(5))
    n = min(PLAYER_CARDS, len(cards))
    for combo in combinations(range(n), 5):
        try:
            s = evaluate_hand(
                [deepcopy(cards[i]) for i in combo],
                deepcopy(jokers),
            )
        except Exception:
            continue
        if s > best_score:
            best_score = s
            best_combo = combo
    return list(best_combo)


def _normalize(indices: List[int], hand_size: int) -> List[int]:
    n = min(PLAYER_CARDS, hand_size)
    seen: List[int] = []
    for i in indices:
        if 0 <= i < n and i not in seen:
            seen.append(i)
        if len(seen) == 5:
            return seen
    for i in range(n):
        if i not in seen:
            seen.append(i)
        if len(seen) == 5:
            break
    return seen[:5]


# ---------------------------------------------------------------------------
# Denial scoring
# ---------------------------------------------------------------------------

def _denial_bonus(joker_name: str) -> float:
    """Flat bonus reflecting how dangerous it is to let the opponent have this."""
    if joker_name in _XMULT_JOKERS:
        return XMULT_DENIAL_BONUS
    if joker_name in _RETRIGGER_JOKERS:
        return RETRIGGER_DENIAL_BONUS
    if joker_name in _COND_MULT_JOKERS:
        return COND_MULT_DENIAL_BONUS
    return 0.0


def _joker_score(
    joker_name: str,
    joker_obj,
    my_hand: List[Card],
    my_jokers: list,
    opp_hand: List[Card],
    opp_jokers: list,
    my_baseline: int,
    opp_baseline: int,
) -> float:
    """
    Combined score for drafting this joker:

        own_gain * OWN_WEIGHT
      - (opp_gain + denial_bonus) * DENIAL_WEIGHT

    own_gain  = how much my best hand improves if I take it.
    opp_gain  = how much their best hand would improve if they took it instead
                (i.e. the threat we're neutralising).

    We negate the denial term because a high opp_gain / bonus means we *want*
    to take it away from them, which increases our overall heuristic score.
    """
    # What we gain
    own_gain = _best_score(my_hand, my_jokers + [joker_obj]) - my_baseline

    # What they would gain
    opp_gain = _best_score(opp_hand, opp_jokers + [joker_obj]) - opp_baseline

    bonus = _denial_bonus(joker_name)

    return own_gain * OWN_WEIGHT - (opp_gain + bonus) * DENIAL_WEIGHT


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class DenialBot:
    """
    Asymmetric denial bot.

    Each draft pick evaluates every available joker with the combined
    own_gain / denial heuristic and chooses the one with the highest score.
    Hand selection is exact brute-force (identical to greedy/minimax).
    """

    def pick_joker(self, state: GameState) -> int:
        my_turn = state.current_turn
        if my_turn not in (PlayerTurn.PLAYER1, PlayerTurn.PLAYER2):
            return 0

        opp_turn = (
            PlayerTurn.PLAYER2
            if my_turn == PlayerTurn.PLAYER1
            else PlayerTurn.PLAYER1
        )

        my_hand = _hand_for(state, my_turn)
        opp_hand = _hand_for(state, opp_turn)
        my_jokers = _jokers_for(state, my_turn)
        opp_jokers = _jokers_for(state, opp_turn)

        # Baselines — what each side scores right now, without any new joker.
        my_baseline = _best_score(my_hand, my_jokers)
        opp_baseline = _best_score(opp_hand, opp_jokers)

        best_score_val = -math.inf
        best_idx = 0

        for i, joker_model in enumerate(state.joker_pool):
            name = joker_model.name
            joker_obj = _joker_from_model(joker_model)

            score = _joker_score(
                name,
                joker_obj,
                my_hand,
                my_jokers,
                opp_hand,
                opp_jokers,
                my_baseline,
                opp_baseline,
            )

            if score > best_score_val:
                best_score_val = score
                best_idx = i

        return best_idx

    def pick_hand(self, state: GameState) -> List[int]:
        my_turn = state.current_turn or PlayerTurn.PLAYER1
        hand = _hand_for(state, my_turn)
        jokers = _jokers_for(state, my_turn)
        return _normalize(_best_indices(hand, jokers), len(hand))


# Standard alias
Bot = DenialBot