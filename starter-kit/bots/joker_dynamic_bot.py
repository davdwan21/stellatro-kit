"""
HybridWinrateBot

Goal:
Win more rounds, not produce the biggest possible score.

Combines:
1. Immediate self-gain from SmartBot
2. Conditional Stella synergy
3. Opponent denial
4. Future pool discount
5. Standalone joker strength
6. Anti-dead-engine penalties
7. Small current-hand compatibility bias
"""

from copy import deepcopy
from itertools import combinations
from collections import Counter
from typing import List, Tuple

from stellatro_common import GameState, PlayerTurn
from stellatro_game import Card, Suit, evaluate_hand, PLAYER_CARDS
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker


_JOKER_NAME_TO_CLASS = {cls.name: cls for cls in ALL_JOKER_CLASSES}


# =========================================================
# Persistent standalone strength
# Small scale on purpose.
# This should guide picks, not overpower real score gain.
# =========================================================

STANDALONE_POWER = {
    "Mirror": 650,
    "Sock and Buskin": 625,
    "PhotoGraph Joker": 600,
    "Galaxy": 575,
    "Sun God": 550,
    "Constellation": 540,
    "Seltzer": 530,
    "Jam Session": 500,
    "Scary Face Joker": 475,
    "The Tribe": 460,
    "The Order": 450,
    "The Duo": 430,
    "The Trio": 425,
    "Half Joker": 420,
    "Lock In": 410,

    # powerful, but more setup-dependent
    "Stargazing": 375,
    "Starcorn": 360,
    "Supernova": 350,
    "Snowball": 340,
    "Cache Coherence": 330,
    "Wish Upon a Star": 320,
    "Report Card": 300,
    "Fallen Star": 290,
    "Star Fish": 285,
    "Encore": 275,

    # medium/filler
    "Color Theory": 250,
    "Flower Pot": 240,
    "Binary Star": 225,
    "Pips": 215,
    "Star Plasma": 210,
    "Boiling Point": 205,
    "Anya": 200,
    "Spotlight": 195,
    "Blackjack": 190,
    "Fibonacci Joker": 170,
    "Group Project": 160,
    "Study Group": 150,
    "Walkie Talkie": 145,
    "Arrowhead": 140,

    "Regular Joker": 0,
}


STELLA_GENERATORS = {
    "Star Plasma",
    "Binary Star",
    "Pips",
    "Wish Upon a Star",
    "Report Card",
    "Cache Coherence",
    "Starjack",
    "Stargazing",      # hybrid
    "Thrice Twice",
    "Fallen Star",
    "Star Fish",
    "Branch Out",
}

STELLA_CONSUMERS = {
    "Boiling Point",
    "Galaxy",
    "Popcorn",
    "Starcorn",
    "Supernova",
    "Snowball",
    "Constellation",
    "Stargazing",      # hybrid
}

RETRIGGERS = {
    "Sock and Buskin",
    "Seltzer",
    "Encore",
    "Last Lecture",
    "Stargazing",
}

FACE_JOKERS = {
    "Mirror",
    "PhotoGraph Joker",
    "Sock and Buskin",
    "Scary Face Joker",
    "Spotlight",
}

LOW_CARD_JOKERS = {
    "Seltzer",
    "Binary Star",
    "Fibonacci Joker",
    "Group Project",
    "Six Seven",
}

FLUSH_JOKERS = {
    "Sun God",
    "The Tribe",
    "Color Theory",
    "Flower Pot",
    "Arrowhead",
}


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


def _best_hand(cards: List[Card], jokers) -> Tuple[float, List[int]]:
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


def _hand_features(cards):
    ranks = [c.rank for c in cards]
    suits = []

    for c in cards:
        suits.extend(list(c.suits))

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    return {
        "faces": sum(1 for r in ranks if 11 <= r <= 13),
        "low": sum(1 for r in ranks if r <= 8),
        "aces": sum(1 for r in ranks if r == 14),
        "pairs": sum(1 for v in rank_counts.values() if v >= 2),
        "max_suit": max(suit_counts.values(), default=0),
    }


def _safe_gain(new_score, base_score):
    return max(0.0, float(new_score) - float(base_score))


class HybridWinrateBot:

    def _conditional_synergy(
        self,
        candidate_name,
        owned_names,
        pool_names,
        hand_info,
        num_owned,
    ):
        bonus = 0.0

        has_gen = bool(owned_names & STELLA_GENERATORS)
        has_con = bool(owned_names & STELLA_CONSUMERS)

        gen_in_pool = bool(pool_names & STELLA_GENERATORS)
        con_in_pool = bool(pool_names & STELLA_CONSUMERS)

        is_gen = candidate_name in STELLA_GENERATORS
        is_con = candidate_name in STELLA_CONSUMERS

        early = num_owned <= 1

        # ----------------------------
        # Stella engine logic
        # ----------------------------

        if is_gen and has_con:
            bonus += 3500

        elif is_gen and con_in_pool:
            bonus += 700

        elif is_gen:
            bonus -= 600

        if is_con and has_gen:
            bonus += 3500

        elif is_con and gen_in_pool:
            bonus += 700

        elif is_con:
            bonus -= 600

        # Stargazing is insane only with Stella already online
        if candidate_name == "Stargazing":
            if has_gen:
                bonus += 2500
            else:
                bonus -= 1800

        # Cache Coherence needs real Stella support
        if candidate_name == "Cache Coherence":
            if has_gen:
                bonus += 1200
            else:
                bonus -= 1000

        # Starcorn/Supernova are bad if no Stella generation
        if candidate_name in {"Starcorn", "Supernova", "Snowball"}:
            if not has_gen:
                bonus -= 1200

        # Early game: avoid speculative combo pieces
        if early and candidate_name in {
            "Stargazing",
            "Starcorn",
            "Supernova",
            "Cache Coherence",
            "Boiling Point",
        }:
            bonus -= 700

        # ----------------------------
        # Retrigger logic
        # ----------------------------

        owned_retriggers = len(owned_names & RETRIGGERS)

        if candidate_name in RETRIGGERS:
            bonus += 500 * owned_retriggers

        if candidate_name == "Jam Session" and owned_retriggers:
            bonus += 1600

        # ----------------------------
        # Current-hand compatibility
        # intentionally small
        # ----------------------------

        if candidate_name in FACE_JOKERS:
            bonus += min(900, hand_info["faces"] * 250)

        if candidate_name in LOW_CARD_JOKERS:
            bonus += min(800, hand_info["low"] * 150)

        if candidate_name in FLUSH_JOKERS and hand_info["max_suit"] >= 4:
            bonus += 700

        if candidate_name in {"The Duo", "The Trio", "Star Fish", "Thrice Twice"}:
            bonus += 300 * hand_info["pairs"]

        if candidate_name in {"Report Card", "Student ID"}:
            bonus += 300 * hand_info["aces"]

        return bonus

    def _opponent_denial(
        self,
        candidate_joker,
        opp_hand,
        opp_jokers,
    ):
        opp_base, _ = _best_hand(opp_hand, opp_jokers)
        opp_with, _ = _best_hand(opp_hand, opp_jokers + [deepcopy(candidate_joker)])
        opp_gain = _safe_gain(opp_with, opp_base)

        # denial matters, but should not dominate self improvement
        return 0.35 * opp_gain

    def _future_pool_discount(
        self,
        candidate_name,
        pool_names,
        my_hand,
        my_jokers,
        my_baseline,
    ):
        remaining = pool_names - {candidate_name}
        if not remaining:
            return 0.0

        best_future_gain = 0.0

        for name in remaining:
            cls = _JOKER_NAME_TO_CLASS.get(name, RegularJoker)
            joker = cls()

            score, _ = _best_hand(my_hand, my_jokers + [joker])
            gain = _safe_gain(score, my_baseline)
            best_future_gain = max(best_future_gain, gain)

        # low because opponent also drafts from the pool
        return 0.12 * best_future_gain

    def _evaluate_candidate(
        self,
        idx,
        state,
        my_hand,
        my_jokers,
        opp_hand,
        opp_jokers,
        hand_info,
    ):
        joker_model = state.joker_pool[idx]
        name = joker_model.name

        cls = _JOKER_NAME_TO_CLASS.get(name, RegularJoker)
        candidate = cls()

        owned_names = {j.name for j in my_jokers}
        pool_names = {j.name for j in state.joker_pool}

        my_baseline, _ = _best_hand(my_hand, my_jokers)
        my_with, _ = _best_hand(my_hand, my_jokers + [candidate])

        self_gain = _safe_gain(my_with, my_baseline)

        standalone = STANDALONE_POWER.get(name, 100)

        synergy = self._conditional_synergy(
            name,
            owned_names,
            pool_names,
            hand_info,
            len(my_jokers),
        )

        denial = self._opponent_denial(
            deepcopy(candidate),
            opp_hand,
            opp_jokers,
        )

        future_discount = self._future_pool_discount(
            name,
            pool_names,
            my_hand,
            my_jokers,
            my_baseline,
        )

        # Final formula:
        # self_gain is still the anchor.
        # standalone/synergy guide consistent drafting.
        value = (
            self_gain
            + standalone
            + synergy
            + denial
            - future_discount
        )

        return value

    def pick_joker(self, state: GameState) -> int:
        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand = _to_cards(state.player1_hand if is_p1 else state.player2_hand)
        my_jokers = _to_jokers(state.player1_jokers if is_p1 else state.player2_jokers)

        opp_hand = _to_cards(state.player2_hand if is_p1 else state.player1_hand)
        opp_jokers = _to_jokers(state.player2_jokers if is_p1 else state.player1_jokers)

        hand_info = _hand_features(my_hand)

        best_idx = 0
        best_value = float("-inf")

        for i in range(len(state.joker_pool)):
            value = self._evaluate_candidate(
                i,
                state,
                my_hand,
                my_jokers,
                opp_hand,
                opp_jokers,
                hand_info,
            )

            if value > best_value:
                best_value = value
                best_idx = i

        return best_idx

    def pick_hand(self, state: GameState) -> List[int]:
        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand = _to_cards(state.player1_hand if is_p1 else state.player2_hand)
        my_jokers = _to_jokers(state.player1_jokers if is_p1 else state.player2_jokers)

        _, indices = _best_hand(my_hand, my_jokers)
        return indices


Bot = HybridWinrateBot