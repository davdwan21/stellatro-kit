"""
CompetitionBot — Advanced Stellatro bot combining:
  - Exhaustive 252-combo hand evaluation with joker-aware scoring
  - Minimax with alpha-beta pruning for the joker draft phase
  - Heuristic joker valuation tuned to hand synergies
"""

from itertools import combinations
from typing import Any, Dict, List, Optional

from bots.bot_interface import BotInterface
from stellatro_common import GameState, PlayerTurn


# ---------------------------------------------------------------------------
# Hand rank constants (ascending)
# ---------------------------------------------------------------------------
HAND_RANKS = {
    "High Card": 0,
    "Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Four of a Kind": 7,
    "Straight Flush": 8,
}

BASE_CHIPS = {
    "High Card": 10,
    "Pair": 20,
    "Two Pair": 30,
    "Three of a Kind": 40,
    "Straight": 60,
    "Flush": 70,
    "Full House": 90,
    "Four of a Kind": 120,
    "Straight Flush": 160,
}

BASE_MULT = {
    "High Card": 1,
    "Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 2,
    "Straight": 3,
    "Flush": 3,
    "Full House": 4,
    "Four of a Kind": 5,
    "Straight Flush": 6,
}

RANK_CHIPS = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10, "A": 11,
}

RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]


# ---------------------------------------------------------------------------
# Pure hand classification (no engine dependency)
# ---------------------------------------------------------------------------

def card_rank(card) -> str:
    """Extract the rank string from a card object."""
    if hasattr(card, "rank"):
        return str(card.rank)
    if hasattr(card, "value"):
        v = str(card.value)
        face = {"11": "J", "12": "Q", "13": "K", "14": "A"}
        return face.get(v, v)
    return str(card)


def card_suit(card) -> str:
    """Extract the suit string from a card object."""
    if hasattr(card, "suit"):
        return str(card.suit)
    return "?"


def rank_index(r: str) -> int:
    try:
        return RANK_ORDER.index(r)
    except ValueError:
        return -1


def classify_five(cards) -> tuple[str, List]:
    """
    Classify exactly 5 cards. Returns (hand_name, scored_cards).
    scored_cards is the subset that contributes chips (all 5 for most hands).
    """
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    idxs = sorted([rank_index(r) for r in ranks], reverse=True)
    rank_counts: Dict[str, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts = sorted(rank_counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    is_straight = (
        len(set(idxs)) == 5 and (max(idxs) - min(idxs) == 4)
    ) or set(idxs) == {0, 1, 2, 3, 12}  # A-2-3-4-5 wheel

    scored = list(cards)  # default: all cards score

    if is_straight and is_flush:
        return "Straight Flush", scored
    if counts[0] == 4:
        # Scored = the four-of-a-kind cards
        quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
        scored = [c for c in cards if card_rank(c) == quad_rank]
        return "Four of a Kind", scored
    if counts[0] == 3 and counts[1] == 2:
        return "Full House", scored
    if is_flush:
        return "Flush", scored
    if is_straight:
        return "Straight", scored
    if counts[0] == 3:
        triple_rank = [r for r, c in rank_counts.items() if c == 3][0]
        scored = [c for c in cards if card_rank(c) == triple_rank]
        return "Three of a Kind", scored
    if counts[0] == 2 and counts[1] == 2:
        pair_ranks = sorted(
            [r for r, c in rank_counts.items() if c == 2],
            key=rank_index, reverse=True
        )
        scored = [c for c in cards if card_rank(c) in pair_ranks]
        return "Two Pair", scored
    if counts[0] == 2:
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        scored = [c for c in cards if card_rank(c) == pair_rank]
        return "Pair", scored

    # High card: only the highest card scores
    best = sorted(cards, key=lambda c: rank_index(card_rank(c)), reverse=True)
    return "High Card", [best[0]]


# ---------------------------------------------------------------------------
# Joker effect simulation
# ---------------------------------------------------------------------------

def joker_name(joker) -> str:
    if hasattr(joker, "name"):
        return str(joker.name)
    return str(joker)


def apply_joker_effects(hand_name: str, scored_cards, jokers, chips: float, mult: float) -> tuple[float, float]:
    """
    Apply joker effects heuristically. We model the most common joker patterns
    described in jokers.md. For jokers we can't simulate exactly we use
    conservative estimates so we don't over-value them.
    """
    for j in jokers:
        name = joker_name(j).lower()

        # --- Mult adders ---
        if "pair" in name and "mult" in name and hand_name == "Pair":
            mult += 4
        if "two pair" in name and "mult" in name and hand_name == "Two Pair":
            mult += 4
        if "three" in name and "mult" in name and "kind" in name and hand_name == "Three of a Kind":
            mult += 4
        if "flush" in name and "mult" in name and "straight" not in name and hand_name == "Flush":
            mult += 4
        if "straight" in name and "mult" in name and "flush" not in name and hand_name == "Straight":
            mult += 4
        if "full" in name and "mult" in name and hand_name == "Full House":
            mult += 4
        if "four" in name and "mult" in name and hand_name == "Four of a Kind":
            mult += 4
        if "straight flush" in name and "mult" in name and hand_name == "Straight Flush":
            mult += 6

        # --- Chip adders ---
        if "gold" in name or "treasure" in name:
            chips += 20
        if "ruby" in name or "diamond" in name:
            chips += 15

        # --- Mult multipliers ---
        if "double" in name and "mult" in name:
            mult *= 2
        if "xmult" in name or "x mult" in name:
            mult *= 1.5  # conservative

        # --- Per-card bonuses ---
        if "face" in name:
            face_cards = [c for c in scored_cards if card_rank(c) in ("J", "Q", "K")]
            chips += len(face_cards) * 10
        if "heart" in name:
            hearts = [c for c in scored_cards if "heart" in card_suit(c).lower()]
            mult += len(hearts)
        if "spade" in name:
            spades = [c for c in scored_cards if "spade" in card_suit(c).lower()]
            chips += len(spades) * 5
        if "club" in name:
            clubs = [c for c in scored_cards if "club" in card_suit(c).lower()]
            chips += len(clubs) * 5
        if "high card" in name and hand_name == "High Card":
            chips += 30

    return chips, mult


def score_hand(cards, jokers) -> float:
    """Score a 5-card hand with the given jokers. Returns chips * mult."""
    hand_name, scored = classify_five(cards)
    chips = float(BASE_CHIPS[hand_name])
    mult = float(BASE_MULT[hand_name])

    for c in scored:
        chips += RANK_CHIPS.get(card_rank(c), 0)

    chips, mult = apply_joker_effects(hand_name, scored, jokers, chips, mult)
    return chips * mult


def best_hand_score(hand, jokers) -> tuple[float, List[int]]:
    """
    Evaluate all C(10,5)=252 subsets of a 10-card hand.
    Returns (best_score, best_indices).
    """
    best_score = -1.0
    best_indices: List[int] = list(range(5))

    for combo in combinations(range(len(hand)), 5):
        cards = [hand[i] for i in combo]
        s = score_hand(cards, jokers)
        if s > best_score:
            best_score = s
            best_indices = list(combo)

    return best_score, best_indices


# ---------------------------------------------------------------------------
# Joker value heuristic (draft evaluation)
# ---------------------------------------------------------------------------

def joker_value_for_hand(joker, hand, existing_jokers) -> float:
    """
    Estimate the marginal value of drafting a joker given our current hand
    and already-drafted jokers.
    """
    name = joker_name(joker).lower()

    # Analyse hand structure
    ranks = [card_rank(c) for c in hand]
    suits = [card_suit(c) for c in hand]
    rank_counts: Dict[str, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)

    suit_counts: Dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values()) if suit_counts else 0

    rank_idxs = sorted([rank_index(r) for r in ranks], reverse=True)
    consecutive = sum(
        1 for a, b in zip(rank_idxs, rank_idxs[1:]) if a - b == 1
    )

    has_pair = counts[0] >= 2
    has_two_pair = counts[0] >= 2 and counts[1] >= 2
    has_trips = counts[0] >= 3
    has_flush_draw = max_suit >= 7
    has_straight_draw = consecutive >= 4
    has_quads = counts[0] >= 4
    has_full_house = counts[0] >= 3 and counts[1] >= 2

    value = 10.0  # base value for any joker

    # Hand-type synergies
    if "pair" in name and not "two" in name:
        value += 20 if has_pair else 5
    if "two pair" in name:
        value += 20 if has_two_pair else 5
    if "three" in name and "kind" in name:
        value += 25 if has_trips else 5
    if "flush" in name and "straight" not in name:
        value += 25 if has_flush_draw else 8
    if "straight" in name and "flush" not in name:
        value += 20 if has_straight_draw else 5
    if "full" in name and "house" in name:
        value += 30 if has_full_house else 5
    if "four" in name and "kind" in name:
        value += 35 if has_quads else 5
    if "straight flush" in name:
        value += 35 if (has_flush_draw and has_straight_draw) else 5

    # Generic power jokers
    if "double" in name and "mult" in name:
        value += 30
    if "xmult" in name or "x mult" in name:
        value += 25
    if "gold" in name or "treasure" in name:
        value += 15
    if "face" in name:
        face_count = sum(1 for r in ranks if r in ("J", "Q", "K"))
        value += face_count * 4

    # Suit-based jokers
    for suit in ("heart", "spade", "club", "diamond"):
        if suit in name:
            count = sum(1 for s in suits if suit in s.lower())
            value += count * 5

    # Penalise jokers that duplicate effects we already have
    for existing in existing_jokers:
        en = joker_name(existing).lower()
        if en == name:
            value -= 15  # duplicate joker

    return value


# ---------------------------------------------------------------------------
# Minimax for the draft phase
# ---------------------------------------------------------------------------

class DraftState:
    """Lightweight state for minimax search during drafting."""
    __slots__ = ("pool", "my_jokers", "opp_jokers", "my_hand", "opp_hand",
                 "picks_left", "is_my_turn")

    def __init__(self, pool, my_jokers, opp_jokers, my_hand, opp_hand,
                 picks_left, is_my_turn):
        self.pool = pool
        self.my_jokers = my_jokers
        self.opp_jokers = opp_jokers
        self.my_hand = my_hand
        self.opp_hand = opp_hand
        self.picks_left = picks_left
        self.is_my_turn = is_my_turn


def draft_heuristic(state: DraftState) -> float:
    """
    Terminal or depth-limit evaluation: expected score delta (us - opponent).
    """
    my_score, _ = best_hand_score(state.my_hand, state.my_jokers)
    opp_score, _ = best_hand_score(state.opp_hand, state.opp_jokers)
    return my_score - opp_score


def minimax(state: DraftState, depth: int, alpha: float, beta: float) -> float:
    if state.picks_left == 0 or depth == 0:
        return draft_heuristic(state)

    if state.is_my_turn:
        best = float("-inf")
        for i, joker in enumerate(state.pool):
            new_pool = state.pool[:i] + state.pool[i+1:]
            new_my = state.my_jokers + [joker]
            child = DraftState(new_pool, new_my, state.opp_jokers,
                               state.my_hand, state.opp_hand,
                               state.picks_left - 1, False)
            val = minimax(child, depth - 1, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        worst = float("inf")
        for i, joker in enumerate(state.pool):
            new_pool = state.pool[:i] + state.pool[i+1:]
            new_opp = state.opp_jokers + [joker]
            child = DraftState(new_pool, state.my_jokers, new_opp,
                               state.my_hand, state.opp_hand,
                               state.picks_left - 1, True)
            val = minimax(child, depth - 1, alpha, beta)
            worst = min(worst, val)
            beta = min(beta, worst)
            if beta <= alpha:
                break
        return worst


def best_draft_pick(pool, my_jokers, opp_jokers, my_hand, opp_hand,
                    picks_left, depth: int = 4) -> int:
    """
    Run minimax and return the pool index of the best joker to draft.
    Falls back to heuristic ranking if pool is large (prunes search).
    """
    # If pool is large, pre-filter to top candidates by heuristic to keep
    # the search tractable. We search the top 6 jokers by heuristic value.
    scored = [
        (joker_value_for_hand(j, my_hand, my_jokers), i, j)
        for i, j in enumerate(pool)
    ]
    scored.sort(reverse=True)

    # Limit branching factor
    candidates = scored[:min(6, len(scored))]

    best_val = float("-inf")
    best_pool_idx = scored[0][1]  # fallback

    for _, pool_idx, joker in candidates:
        new_pool = pool[:pool_idx] + pool[pool_idx+1:]
        new_my = my_jokers + [joker]
        child = DraftState(new_pool, new_my, opp_jokers,
                           my_hand, opp_hand,
                           picks_left - 1, False)
        val = minimax(child, depth - 1, float("-inf"), float("inf"))
        if val > best_val:
            best_val = val
            best_pool_idx = pool_idx

    return best_pool_idx


# ---------------------------------------------------------------------------
# The Bot
# ---------------------------------------------------------------------------

class CompetitionBot(BotInterface):
    """
    A strong Stellatro bot:
      - Draft phase: minimax with alpha-beta (depth 4, branching factor 6)
      - Play phase: exhaustive C(10,5) hand evaluation with joker simulation
    """

    MINIMAX_DEPTH = 4

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)

    def _my_hand(self, gs: GameState):
        if gs.current_turn == PlayerTurn.PLAYER1:
            return gs.player1_hand
        return gs.player2_hand

    def _opp_hand(self, gs: GameState):
        if gs.current_turn == PlayerTurn.PLAYER1:
            return gs.player2_hand
        return gs.player1_hand

    def _my_jokers(self, gs: GameState):
        if gs.current_turn == PlayerTurn.PLAYER1:
            return list(gs.player1_jokers)
        return list(gs.player2_jokers)

    def _opp_jokers(self, gs: GameState):
        if gs.current_turn == PlayerTurn.PLAYER1:
            return list(gs.player2_jokers)
        return list(gs.player1_jokers)

    def pick_joker(self, gs: GameState) -> int:
        pool = list(gs.joker_pool)
        if not pool:
            return 0

        my_hand = self._my_hand(gs)
        opp_hand = self._opp_hand(gs)
        my_jokers = self._my_jokers(gs)
        opp_jokers = self._opp_jokers(gs)

        # How many total picks remain (both players)?
        picks_remaining = len(pool)  # each pick removes one joker

        # Use minimax when the pool is small enough to be useful
        # and we have enough picks left to bother looking ahead.
        if len(pool) <= 12 and picks_remaining > 1:
            idx = best_draft_pick(
                pool, my_jokers, opp_jokers, my_hand, opp_hand,
                picks_remaining, depth=self.MINIMAX_DEPTH
            )
        else:
            # Pool too large for deep search — use heuristic ranking
            scored = [
                (joker_value_for_hand(j, my_hand, my_jokers), i)
                for i, j in enumerate(pool)
            ]
            scored.sort(reverse=True)
            idx = scored[0][1]

        return idx

    def pick_hand(self, gs: GameState) -> List[int]:
        my_hand = self._my_hand(gs)
        my_jokers = self._my_jokers(gs)

        if len(my_hand) < 5:
            return list(range(len(my_hand)))

        _, best_indices = best_hand_score(my_hand, my_jokers)
        return best_indices