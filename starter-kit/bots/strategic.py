"""
Strategic bot — maximises win rate rather than raw score.

Draft phase:
  For each candidate joker, score my best hand WITH that joker, then also
  estimate how much that joker would benefit the opponent. Pick the joker
  that maximises (my_gain + denial_weight * opponent_gain), i.e. combined
  swing value. Denial weight is higher early in the draft (when denying
  an xMult joker is devastating) and tapers off late.

Play phase:
  Enumerate all C(10,5)=252 subsets exactly.
  - If we are likely AHEAD of the opponent's estimated score, play the
    highest-scoring hand (we already win, just maximise margin).
  - If we are likely BEHIND, play the subset with the highest score
    (best chance of flipping the result).
  - Either way, always play the highest-scoring hand — win rate is
    maximised by maximising our score, not by "safe" sub-ceiling picks.
"""

from copy import deepcopy
from itertools import combinations
from typing import List, Tuple

from stellatro_common import GameState, PlayerTurn
from stellatro_game import Card, Suit, evaluate_hand, PLAYER_CARDS
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JOKER_NAME_TO_CLASS = {cls.name: cls for cls in ALL_JOKER_CLASSES}

# xMult jokers — taking these away from the opponent hard-caps their ceiling.
# We value denying these more heavily, especially early in the draft.
_XMULT_JOKER_NAMES = frozenset({
    "The Duo", "The Trio", "The Tribe", "The Order",
    "UC Socially Dead", "Flower Pot", "Sun God", "Plasma",
    "Blackjack", "Boiling Point", "Supernova",
})


def _to_cards(card_models) -> List[Card]:
    cards = []
    for c in card_models:
        # suits may be a set or list — convert safely
        suits = list(c.suits)
        if not suits:
            continue
        card = Card(c.rank, Suit(suits[0]))
        for s in suits[1:]:
            card.add_suit(Suit(s))
        card.scored = c.scored
        card.num_triggers = c.num_triggers
        cards.append(card)
    return cards


def _to_joker_instances(joker_models):
    return [_JOKER_NAME_TO_CLASS.get(j.name, RegularJoker)() for j in joker_models]


def _to_joker_classes(joker_models):
    return [_JOKER_NAME_TO_CLASS.get(j.name, RegularJoker) for j in joker_models]


def _score_combo(cards: List[Card], joker_instances, combo: Tuple[int, ...]) -> int:
    try:
        return evaluate_hand(
            [deepcopy(cards[i]) for i in combo],
            deepcopy(joker_instances),
        )
    except Exception:
        return 0


def _best_hand(cards: List[Card], joker_instances) -> Tuple[int, List[int]]:
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return 0, list(range(n))
    best_score = -1
    best_indices = list(range(5))
    for combo in combinations(range(n), 5):
        score = _score_combo(cards, joker_instances, combo)
        if score > best_score:
            best_score = score
            best_indices = list(combo)
    return max(0, best_score), best_indices


def _denial_weight(pick_num: int, joker_name: str) -> float:
    """
    How much to weight opponent denial vs self-gain.

    Early picks: denying an xMult joker is catastrophic for the opponent,
    so we weight it heavily. Late picks: pool is thin, just take value.

    pick_num is how many jokers we've already drafted (0-indexed).
    """
    is_xmult = joker_name in _XMULT_JOKER_NAMES

    if is_xmult:
        # xMult denial weights by pick number
        weights = [1.8, 1.4, 1.0, 0.6, 0.4]
    else:
        # Standard jokers — moderate denial early, low late
        weights = [0.8, 0.6, 0.4, 0.3, 0.2]

    idx = min(pick_num, len(weights) - 1)
    return weights[idx]


# ---------------------------------------------------------------------------
# Strategic bot
# ---------------------------------------------------------------------------

class StrategicBot:

    def pick_joker(self, state: GameState) -> int:
        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand       = _to_cards(state.player1_hand    if is_p1 else state.player2_hand)
        my_jokers_raw = state.player1_jokers             if is_p1 else state.player2_jokers
        opp_hand      = _to_cards(state.player2_hand    if is_p1 else state.player1_hand)
        opp_jokers_raw= state.player2_jokers             if is_p1 else state.player1_jokers

        my_jokers  = _to_joker_instances(my_jokers_raw)
        opp_jokers = _to_joker_instances(opp_jokers_raw)

        # How many jokers have I drafted so far?
        pick_num = len(my_jokers_raw)

        # Baselines without any new joker
        my_baseline,  _ = _best_hand(my_hand,  my_jokers)
        opp_baseline, _ = _best_hand(opp_hand, opp_jokers)

        best_swing = -float("inf")
        best_idx = 0

        for i, joker_model in enumerate(state.joker_pool):
            joker_name = joker_model.name
            candidate_cls = _JOKER_NAME_TO_CLASS.get(joker_name, RegularJoker)

            # How much does this joker help me?
            my_score_with, _ = _best_hand(my_hand, my_jokers + [candidate_cls()])
            my_gain = my_score_with - my_baseline

            # How much would this joker help the opponent?
            opp_score_with, _ = _best_hand(opp_hand, opp_jokers + [candidate_cls()])
            opp_gain = max(0, opp_score_with - opp_baseline)

            w = _denial_weight(pick_num, joker_name)
            swing = my_gain + w * opp_gain

            if swing > best_swing:
                best_swing = swing
                best_idx = i

        return best_idx

    def pick_hand(self, state: GameState) -> List[int]:
        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand    = _to_cards(state.player1_hand    if is_p1 else state.player2_hand)
        my_jokers  = _to_joker_instances(state.player1_jokers if is_p1 else state.player2_jokers)
        opp_hand   = _to_cards(state.player2_hand    if is_p1 else state.player1_hand)
        opp_jokers = _to_joker_instances(state.player2_jokers if is_p1 else state.player1_jokers)

        # Always play our best possible hand.
        # Since scoring is win/draw/loss (not margin-based), the best strategy
        # is always to maximise our score — this maximises the probability of
        # being above whatever the opponent scores, regardless of whether we
        # think we're ahead or behind. "Safe" sub-ceiling plays only help when
        # we know the exact opponent score, which we don't at play time.
        _, indices = _best_hand(my_hand, my_jokers)

        # Validate and fill if needed
        n = min(PLAYER_CARDS, len(my_hand))
        unique: List[int] = []
        for idx in indices:
            if 0 <= idx < n and idx not in unique:
                unique.append(idx)
        for idx in range(n):
            if len(unique) == 5:
                break
            if idx not in unique:
                unique.append(idx)

        return unique[:5]


Bot = StrategicBot