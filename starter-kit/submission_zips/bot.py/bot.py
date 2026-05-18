"""
Archetype-aware bot — identifies which strategy archetype is being built
(by self and opponent) and drafts/denies accordingly.

Core ideas:
  1. Every joker belongs to one or more archetypes.
  2. After each pick, classify how committed each player is to each archetype.
  3. During drafting:
       a. Score each candidate joker by how much it advances MY leading archetype.
       b. Add a denial bonus if the joker also advances the OPPONENT'S leading archetype.
       c. If my hand doesn't support my leading archetype, switch to the next best.
  4. Play phase is identical to strategic_bot: ahead → floor, behind → ceiling.
"""

from copy import deepcopy
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from stellatro_common import GameState, PlayerTurn
from stellatro_game import Card, Suit, evaluate_hand, PLAYER_CARDS
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

# ---------------------------------------------------------------------------
# Joker → archetype membership
# Each joker can belong to multiple archetypes (list of archetype names).
# Weights > 1.0 mark "engine" pieces that are especially critical.
# ---------------------------------------------------------------------------

ARCHETYPES: Dict[str, Dict[str, float]] = {
    # ------------------------------------------------------------------
    # STELLA ENGINE
    # Goal: accumulate stella on cards, then cash out via multipliers.
    # Key enablers: Pips, Binary Star, Report Card, Branch Out, Wish Upon a Star
    # Key payoffs: Supernova, Starcorn, Boiling Point, Galaxy, Snowball, Constellation
    # Bridges (give stella AND score): Stargazing, Cache Coherence, Fallen Star,
    #         Star Fish, Starjack, Star Plasma, Thrice Twice
    # ------------------------------------------------------------------
    "stella": {
        # Enablers (give stella)
        "Pips":              2.0,
        "Binary Star":       1.5,
        "Report Card":       1.5,
        "Branch Out":        1.5,
        "Wish Upon a Star":  1.0,
        "Star Plasma":       1.5,
        "Starjack":          1.0,
        "Thrice Twice":      1.0,
        "Fallen Star":       1.0,
        "Star Fish":         1.0,
        "Cache Coherence":   1.0,
        # Bridges
        "Stargazing":        2.0,   # stella → retriggers; critical piece
        # Payoffs
        "Supernova":         2.0,
        "Starcorn":          2.0,
        "Boiling Point":     1.5,
        "Galaxy":            1.5,
        "Snowball":          1.5,
        "Constellation":     1.5,
        "Popcorn":          -1.0,   # anti-stella; negative membership
    },

    # ------------------------------------------------------------------
    # HAND TYPE MULTIPLIER
    # Goal: play a specific hand type and stack jokers that reward it.
    # ------------------------------------------------------------------
    "hand_type": {
        # Pair
        "Jolly Joker":   1.0,
        "Sly Joker":     0.8,
        "The Duo":       1.5,
        # Two Pair
        "Cheeky Joker":  1.0,
        "Jovial Joker":  0.8,
        # Three of a Kind
        "Zany Joker":    1.0,
        "Merry Joker":   0.8,
        "The Trio":      1.5,
        # Straight
        "Witty Joker":   1.0,
        "Lively Joker":  0.8,
        "The Order":     1.5,
        # Flush
        "Daring Joker":  1.0,
        "Vibrant Joker": 0.8,
        "The Tribe":     1.5,
        # Full House
        "Thrice Twice":  0.5,   # also stella
        # Generic hand-type boost
        "Half Joker":    1.0,
        "Six Seven":     0.8,
        "Blackjack":     1.0,
    },

    # ------------------------------------------------------------------
    # FACE CARD RETRIGGER
    # Goal: load face cards, retrigger them, multiply via face-specific jokers.
    # ------------------------------------------------------------------
    "face": {
        "Sock and Buskin":    2.0,   # retrigger all face cards
        "PhotoGraph Joker":   1.5,
        "Spotlight":          1.5,
        "Scary Face Joker":   1.0,
        "Mirror":             1.0,
        "Bit Byte":           0.8,
        "Starjack":           0.5,   # also stella
        "Last Lecture":       0.8,
        "Jam Session":        1.0,   # rewards retriggers
    },

    # ------------------------------------------------------------------
    # SUIT SYNERGY
    # Goal: concentrate on one or all suits for bonus chips/mult.
    # ------------------------------------------------------------------
    "suit": {
        "Flower Pot":     2.0,   # x3 if all 4 suits
        "Diamond Joker":  1.0,
        "Heart Joker":    1.0,
        "Club Joker":     1.0,
        "Spade Joker":    1.0,
        "Arrowhead":      1.0,
        "Sun God":        1.5,
        "Color Theory":   1.5,
        "Encore":         0.8,
    },

    # ------------------------------------------------------------------
    # NICHE / HIGH-CARD GOTCHA
    # Goal: unusual win conditions that most bots don't account for.
    # ------------------------------------------------------------------
    "niche": {
        "UC Socially Dead": 2.0,
        "Student ID":       1.5,
        "Blackjack":        1.5,
        "Six Seven":        1.5,
        "Loss Cut":         1.0,
        "Lock In":          2.0,
        "Plasma":           1.0,
    },

    # ------------------------------------------------------------------
    # LOW-RANK ENGINE
    # Goal: play low cards, retrigger them (Seltzer), boost chips.
    # ------------------------------------------------------------------
    "low_rank": {
        "Seltzer":              2.0,
        "Dining Hall Prices":   1.5,
        "Group Project":        1.5,
        "Walkie Talkie":        1.0,
        "Fibonacci Joker":      1.0,
        "Eight College":        1.0,
        "Half Joker":           0.8,
        "Study Group":          0.8,
    },
}

# Flat lookup: joker_name → {archetype: weight}
_JOKER_ARCHETYPE: Dict[str, Dict[str, float]] = defaultdict(dict)
for _arch, _members in ARCHETYPES.items():
    for _name, _w in _members.items():
        _JOKER_ARCHETYPE[_name][_arch] = _w


# ---------------------------------------------------------------------------
# Shared card/joker conversion helpers
# ---------------------------------------------------------------------------

_JOKER_NAME_TO_CLASS = {cls.name: cls for cls in ALL_JOKER_CLASSES}


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


def _score_combo(cards: List[Card], jokers, combo: Tuple[int, ...]) -> int:
    try:
        return evaluate_hand(
            [deepcopy(cards[i]) for i in combo],
            deepcopy(jokers),
        )
    except Exception:
        return 0


def _all_scores(cards: List[Card], jokers) -> List[Tuple[int, List[int]]]:
    n = min(PLAYER_CARDS, len(cards))
    results = []
    for combo in combinations(range(n), 5):
        score = _score_combo(cards, jokers, combo)
        results.append((score, list(combo)))
    return results


def _best_hand(cards: List[Card], jokers) -> Tuple[int, List[int]]:
    results = _all_scores(cards, jokers)
    return max(results, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Archetype scoring helpers
# ---------------------------------------------------------------------------

def _archetype_scores(joker_models) -> Dict[str, float]:
    """Return total archetype commitment score for a list of joker models."""
    scores: Dict[str, float] = defaultdict(float)
    for j in joker_models:
        for arch, w in _JOKER_ARCHETYPE.get(j.name, {}).items():
            scores[arch] += w
    return scores


def _leading_archetype(joker_models) -> Optional[str]:
    """Return the archetype with the highest commitment score, or None."""
    scores = _archetype_scores(joker_models)
    if not scores:
        return None
    return max(scores, key=lambda a: scores[a])


def _hand_supports_archetype(cards: List[Card], archetype: str) -> bool:
    """
    Quick heuristic: does the hand have the raw material for this archetype?
    """
    ranks = [c.rank for c in cards]
    suits = [next(iter(c.suits)) if hasattr(c, "suits") and c.suits else c.suit for c in cards]

    if archetype == "stella":
        # High-rank cards generate more stella via Pips/Starcorn
        high_rank_count = sum(1 for r in ranks if r >= 8)
        return high_rank_count >= 3

    if archetype == "face":
        face_count = sum(1 for r in ranks if r in (11, 12, 13))  # J Q K
        return face_count >= 3

    if archetype == "suit":
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        # Either 4+ of one suit (flush potential) or all 4 suits present
        return max(suit_counts.values()) >= 4 or len(suit_counts) >= 4

    if archetype == "low_rank":
        low_count = sum(1 for r in ranks if r <= 8)
        return low_count >= 4

    if archetype == "hand_type":
        # Almost always supportable — just need pairs/straights/flushes
        return True

    if archetype == "niche":
        return True

    return True


# ---------------------------------------------------------------------------
# Archetype-aware bot
# ---------------------------------------------------------------------------

class ArchetypeBot:
    # Weight on denying the opponent's archetype jokers (on top of raw EV swing)
    _DENIAL_WEIGHT  = 1.0
    # Bonus multiplier applied when a joker directly advances our committed archetype
    _ARCHETYPE_BONUS = 0.5   # fraction of raw EV gain added as bonus

    def pick_joker(self, state: GameState) -> int:
        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand       = _to_cards(state.player1_hand    if is_p1 else state.player2_hand)
        my_jokers_raw = state.player1_jokers             if is_p1 else state.player2_jokers
        my_jokers     = _to_jokers(my_jokers_raw)
        opp_hand      = _to_cards(state.player2_hand    if is_p1 else state.player1_hand)
        opp_jokers_raw= state.player2_jokers             if is_p1 else state.player1_jokers
        opp_jokers    = _to_jokers(opp_jokers_raw)

        # Identify each player's leading archetype
        my_archetype  = _leading_archetype(my_jokers_raw)
        opp_archetype = _leading_archetype(opp_jokers_raw)

        # If my leading archetype doesn't suit my hand, pick the next best one
        if my_archetype and not _hand_supports_archetype(my_hand, my_archetype):
            my_arch_scores = _archetype_scores(my_jokers_raw)
            sorted_archs = sorted(my_arch_scores, key=lambda a: my_arch_scores[a], reverse=True)
            for arch in sorted_archs:
                if _hand_supports_archetype(my_hand, arch):
                    my_archetype = arch
                    break

        my_baseline,  _ = _best_hand(my_hand,  my_jokers)
        opp_baseline, _ = _best_hand(opp_hand, opp_jokers)

        best_value = -float("inf")
        best_idx   = 0

        for i, joker_model in enumerate(state.joker_pool):
            candidate     = _JOKER_NAME_TO_CLASS.get(joker_model.name, RegularJoker)()
            candidate_name = joker_model.name

            # --- Raw EV gain for me ---
            my_score_with, _ = _best_hand(my_hand, my_jokers + [candidate])
            my_gain = my_score_with - my_baseline

            # --- Archetype synergy bonus ---
            # If this joker advances my committed archetype, boost its value.
            arch_bonus = 0.0
            if my_archetype:
                joker_arch_weights = _JOKER_ARCHETYPE.get(candidate_name, {})
                arch_weight = joker_arch_weights.get(my_archetype, 0.0)
                if arch_weight > 0:
                    arch_bonus = my_gain * self._ARCHETYPE_BONUS * arch_weight

            # --- Denial value ---
            # How much would this joker advance the OPPONENT's archetype?
            opp_score_with, _ = _best_hand(opp_hand, opp_jokers + [candidate])
            opp_raw_gain = opp_score_with - opp_baseline

            # Extra denial bonus if the joker is a key piece for their archetype
            opp_arch_denial = 0.0
            if opp_archetype:
                opp_arch_weights = _JOKER_ARCHETYPE.get(candidate_name, {})
                opp_arch_weight  = opp_arch_weights.get(opp_archetype, 0.0)
                if opp_arch_weight > 0:
                    # Deny jokers that are critical engine pieces more aggressively
                    opp_arch_denial = opp_raw_gain * self._ARCHETYPE_BONUS * opp_arch_weight

            total_value = (
                my_gain
                + arch_bonus
                + self._DENIAL_WEIGHT * (opp_raw_gain + opp_arch_denial)
            )

            if total_value > best_value:
                best_value = total_value
                best_idx   = i

        return best_idx

    def pick_hand(self, state: GameState) -> List[int]:
        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand   = _to_cards(state.player1_hand    if is_p1 else state.player2_hand)
        my_jokers = _to_jokers(state.player1_jokers if is_p1 else state.player2_jokers)
        opp_hand  = _to_cards(state.player2_hand    if is_p1 else state.player1_hand)
        opp_jokers= _to_jokers(state.player2_jokers if is_p1 else state.player1_jokers)

        opp_best_score, _ = _best_hand(opp_hand, opp_jokers)

        all_results = _all_scores(my_hand, my_jokers)
        all_results.sort(key=lambda x: x[0])

        best_score = all_results[-1][0]

        if best_score > opp_best_score:
            # AHEAD: lock in the win with highest score
            _, indices = all_results[-1]
        else:
            # BEHIND: find the lowest score that still beats them (safest winning hand)
            beating = [(s, idx) for s, idx in all_results if s > opp_best_score]
            if beating:
                _, indices = min(beating, key=lambda x: x[0])
            else:
                # Can't win — maximise score anyway
                _, indices = all_results[-1]

        return indices


Bot = ArchetypeBot