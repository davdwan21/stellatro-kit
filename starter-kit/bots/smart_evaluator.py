"""
smart_evaluator.py
Drop-in replacement for GreedyBot.pick_joker with three improvements:
  1. Stella synergy bonus  — rewards completing a generator/consumer pair
  2. Opponent denial       — penalises leaving a joker the opponent wants
  3. Future pool discount  — discounts a weak pick when better jokers remain

Usage: copy this file next to greedy_bot.py and use SmartBot instead.
"""

from copy import deepcopy
from itertools import combinations
from typing import List, Tuple

from stellatro_common import GameState, PlayerTurn
from stellatro_game import Card, Suit, evaluate_hand, PLAYER_CARDS
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

# ---------------------------------------------------------------------------
# Joker name → class lookup (same as greedy_bot)
# ---------------------------------------------------------------------------

_JOKER_NAME_TO_CLASS = {cls.name: cls for cls in ALL_JOKER_CLASSES}

# ---------------------------------------------------------------------------
# Stella taxonomy
# Generators add stella to cards; consumers turn stella into score.
# A generator alone or consumer alone is weak; both together is strong.
# ---------------------------------------------------------------------------

STELLA_GENERATORS = {
    "Star Plasma",       # x2 stella on every played card
    "Binary Star",       # even cards gain 2 stella
    "Pips",              # cards gain stella = rank (but lose base chips)
    "Wish Upon a Star",  # lowest card gains 8 stella
    "Report Card",       # each ace gives first card 11 stella
    "Cache Coherence",   # same-suit cards share max stella
    "Starjack",          # first face card gains 10 stella
    "Stargazing",        # each stella gives a retrigger (generator + consumer hybrid)
    "Thrice Twice",      # full house → each card gains 3 stella
    "Fallen Star",       # swaps stella between lowest/highest scored cards
    "Star Fish",         # pairs/trips/quads gain stella
    "Branch Out",        # each card carries half previous card's stella
}

STELLA_CONSUMERS = {
    "Boiling Point",     # total stella > 12 → x3 Mult
    "Galaxy",            # +0.25x Mult per stella
    "Popcorn",           # +30 Mult, -5 per stella (careful: can go negative)
    "Starcorn",          # each card gives (rank * stella) Mult
    "Supernova",         # x(1.1)^stella per card
    "Snowball",          # +40 Chips per stella
    "Constellation",     # +8 chips +3 mult per stella on scored cards
}

# ---------------------------------------------------------------------------
# Helpers (mirrors greedy_bot helpers)
# ---------------------------------------------------------------------------

def _to_cards(card_models) -> List[Card]:
    cards = []
    for c in card_models:
        card = Card(c.rank, Suit(c.suits[0]))
        for s in c.suits[1:]:
            card.add_suit(Suit(s))
        card.scored = c.scored
        card.num_triggers = c.num_triggers
        cards.append(card)
    return cards


def _to_jokers(joker_models):
    return [_JOKER_NAME_TO_CLASS.get(j.name, RegularJoker)() for j in joker_models]


def _best_hand(cards: List[Card], jokers) -> Tuple[int, List[int]]:
    """Return (best_score, best_indices) across all C(n,5) combos."""
    best_score = -1
    best_indices = list(range(5))
    n = min(PLAYER_CARDS, len(cards))
    for combo in combinations(range(n), 5):
        try:
            score = evaluate_hand(
                [deepcopy(cards[i]) for i in combo],
                deepcopy(jokers),
            )
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_indices = list(combo)
    return best_score, best_indices


# ---------------------------------------------------------------------------
# Stella synergy bonus
# ---------------------------------------------------------------------------

def _joker_names(joker_models) -> set:
    return {j.name for j in joker_models}


def _stella_synergy_bonus(
    candidate_name: str,
    my_joker_names: set,
    pool_names: set,
) -> float:
    """
    Return a score bonus to add when the candidate completes or extends a
    stella engine.

    Logic:
      - If candidate is a GENERATOR and we already own a CONSUMER → big bonus
      - If candidate is a CONSUMER and we already own a GENERATOR → big bonus
      - If candidate is a GENERATOR/CONSUMER but the other half isn't owned
        AND isn't in the pool → penalty (we'll never complete the engine)
      - If candidate is a GENERATOR/CONSUMER and the other half IS still in
        the pool → small bonus (we might draft the other half next)
    """
    COMPLETE_ENGINE_BONUS = 8_000   # completing the pair is very valuable
    HALF_ENGINE_IN_POOL   = 2_000   # other half still draftable
    DEAD_HALF_PENALTY     = -1_000  # no way to complete the engine

    is_gen  = candidate_name in STELLA_GENERATORS
    is_con  = candidate_name in STELLA_CONSUMERS

    if not is_gen and not is_con:
        return 0.0

    if is_gen:
        already_have_consumer = bool(my_joker_names & STELLA_CONSUMERS)
        consumer_in_pool      = bool(pool_names & STELLA_CONSUMERS)
        if already_have_consumer:
            return COMPLETE_ENGINE_BONUS
        elif consumer_in_pool:
            return HALF_ENGINE_IN_POOL
        else:
            return DEAD_HALF_PENALTY

    if is_con:
        already_have_generator = bool(my_joker_names & STELLA_GENERATORS)
        generator_in_pool      = bool(pool_names & STELLA_GENERATORS)
        if already_have_generator:
            return COMPLETE_ENGINE_BONUS
        elif generator_in_pool:
            return HALF_ENGINE_IN_POOL
        else:
            return DEAD_HALF_PENALTY

    return 0.0


# ---------------------------------------------------------------------------
# Opponent denial score
# ---------------------------------------------------------------------------

def _opponent_denial_score(
    candidate_name: str,
    opp_hand: List[Card],
    opp_jokers,
    candidate_joker,
) -> float:
    """
    How much would this joker improve the opponent's score?
    We use a fraction of that as a denial bonus for taking it away.

    Denial is worth less than self-improvement (0.4 weight), because
    we only care about the score *gap*, not raw opponent damage.
    """
    DENIAL_WEIGHT = 0.4

    opp_baseline, _ = _best_hand(opp_hand, opp_jokers)
    opp_with_joker, _ = _best_hand(opp_hand, opp_jokers + [candidate_joker])
    opp_gain = max(0, opp_with_joker - opp_baseline)
    return DENIAL_WEIGHT * opp_gain


# ---------------------------------------------------------------------------
# Future pool discount
# ---------------------------------------------------------------------------

def _future_pool_best(
    pool_names: set,
    candidate_name: str,
    my_hand: List[Card],
    my_jokers,
) -> float:
    """
    What's the best score gain available from the *remaining* pool after
    taking this candidate?  If there are much better jokers still available,
    this pick should be discounted — but note the opponent gets next pick,
    so we weight remaining pool value lightly.
    """
    FUTURE_DISCOUNT = 0.15   # low weight: opponent picks next, pool shrinks

    remaining = pool_names - {candidate_name}
    if not remaining:
        return 0.0

    baseline, _ = _best_hand(my_hand, my_jokers)
    best_future_gain = 0
    for name in remaining:
        cls = _JOKER_NAME_TO_CLASS.get(name, RegularJoker)
        joker = cls()
        score, _ = _best_hand(my_hand, my_jokers + [joker])
        gain = max(0, score - baseline)
        best_future_gain = max(best_future_gain, gain)

    return FUTURE_DISCOUNT * best_future_gain


# ---------------------------------------------------------------------------
# Main composite score
# ---------------------------------------------------------------------------

def _evaluate_joker(
    candidate_idx: int,
    state: GameState,
    my_hand: List[Card],
    my_jokers,
    opp_hand: List[Card],
    opp_jokers,
) -> float:
    """
    Composite joker value = self_gain + stella_synergy + denial - future_discount
    """
    joker_model = state.joker_pool[candidate_idx]
    candidate_name = joker_model.name
    cls = _JOKER_NAME_TO_CLASS.get(candidate_name, RegularJoker)
    candidate_joker = cls()

    my_joker_names  = {j.name for j in (state.player1_jokers
                       if state.current_turn == PlayerTurn.PLAYER1
                       else state.player2_jokers)}
    pool_names      = {j.name for j in state.joker_pool}

    # 1. Raw self-improvement
    baseline, _    = _best_hand(my_hand, my_jokers)
    with_joker, _  = _best_hand(my_hand, my_jokers + [candidate_joker])
    self_gain      = with_joker - baseline

    # 2. Stella synergy adjustment
    synergy = _stella_synergy_bonus(candidate_name, my_joker_names, pool_names)

    # 3. Opponent denial
    denial = _opponent_denial_score(
        candidate_name, opp_hand, opp_jokers, deepcopy(candidate_joker)
    )

    # 4. Future pool discount (subtract: if better jokers remain, wait)
    future_val = _future_pool_best(pool_names, candidate_name, my_hand, my_jokers)

    return self_gain + synergy + denial - future_val


# ---------------------------------------------------------------------------
# SmartBot — drop-in replacement for GreedyBot
# ---------------------------------------------------------------------------

class SmartBot:
    """
    Identical to GreedyBot except pick_joker uses the composite evaluator.
    pick_hand is unchanged — exhaustive search is already optimal.
    """

    def pick_joker(self, state: GameState) -> int:
        is_p1     = state.current_turn == PlayerTurn.PLAYER1
        my_hand   = _to_cards(state.player1_hand   if is_p1 else state.player2_hand)
        my_jokers = _to_jokers(state.player1_jokers if is_p1 else state.player2_jokers)
        opp_hand  = _to_cards(state.player2_hand   if is_p1 else state.player1_hand)
        opp_jokers = _to_jokers(state.player2_jokers if is_p1 else state.player1_jokers)

        best_score = float("-inf")
        best_idx   = 0

        for i in range(len(state.joker_pool)):
            score = _evaluate_joker(
                i, state, my_hand, my_jokers, opp_hand, opp_jokers
            )
            if score > best_score:
                best_score = score
                best_idx   = i

        return best_idx

    def pick_hand(self, state: GameState) -> List[int]:
        is_p1     = state.current_turn == PlayerTurn.PLAYER1
        my_hand   = _to_cards(state.player1_hand   if is_p1 else state.player2_hand)
        my_jokers = _to_jokers(state.player1_jokers if is_p1 else state.player2_jokers)
        _, indices = _best_hand(my_hand, my_jokers)
        return indices


Bot = SmartBot