from copy import deepcopy
from itertools import combinations
from typing import List

from stellatro_common import GameState, PlayerTurn
from stellatro_game import Card, Suit, evaluate_hand, PLAYER_CARDS
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

_JOKER_NAME_TO_CLASS = {cls.name: cls for cls in ALL_JOKER_CLASSES}

# ---------- STATIC META RANKING ----------
JOKER_PRIORITY = {
    "Stargazing": 10000,
    "Starcorn": 9800,
    "Supernova": 9600,
    "Sock and Buskin": 9400,
    "Seltzer": 9300,
    "Galaxy": 9000,
    "Constellation": 8900,
    "Snowball": 8800,
    "Cache Coherence": 8700,
    "Wish Upon a Star": 8600,
    "Mirror": 8500,
    "Sun God": 8400,
    "PhotoGraph Joker": 8350,
    "Star Plasma": 8300,
    "Report Card": 8000,
    "Fallen Star": 7900,
    "Star Fish": 7800,
    "Encore": 7700,
    "Jam Session": 7600,
    "Lock In": 7500,
}

# ---------- SYNERGY GROUPS ----------
STELLA_GENERATORS = {
    "Wish Upon a Star",
    "Binary Star",
    "Pips",
    "Report Card",
    "Starjack",
    "Thrice Twice",
    "Fallen Star",
    "Star Fish",
}

STELLA_PAYOFFS = {
    "Stargazing",
    "Galaxy",
    "Constellation",
    "Snowball",
    "Starcorn",
    "Supernova",
    "Boiling Point",
}

RETRIGGER_JOKERS = {
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


def _best_hand(cards: List[Card], jokers):
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


class MetaGreedyBot:

    def _synergy_bonus(self, owned_names, candidate_name):

        bonus = 0

        # Stella engine stacking
        if candidate_name in STELLA_GENERATORS:
            bonus += 1200 * len(owned_names & STELLA_PAYOFFS)

        if candidate_name in STELLA_PAYOFFS:
            bonus += 1400 * len(owned_names & STELLA_GENERATORS)

        # Retrigger stacking
        if candidate_name in RETRIGGER_JOKERS:
            bonus += 800 * len(owned_names & RETRIGGER_JOKERS)

        # Face synergy
        if candidate_name in FACE_JOKERS:
            bonus += 600 * len(owned_names & FACE_JOKERS)

        # Cache Coherence is insane with Stella
        if candidate_name == "Cache Coherence":
            bonus += 2000 * len(owned_names & STELLA_GENERATORS)

        # Stargazing is the real win condition
        if candidate_name == "Stargazing":
            bonus += 3000 * len(owned_names & STELLA_GENERATORS)

        return bonus

    def pick_joker(self, state: GameState) -> int:

        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand = _to_cards(
            state.player1_hand if is_p1 else state.player2_hand
        )

        my_jokers = _to_jokers(
            state.player1_jokers if is_p1 else state.player2_jokers
        )

        owned_names = {j.name for j in my_jokers}

        best_value = -1
        best_idx = 0

        for i, joker_model in enumerate(state.joker_pool):

            name = joker_model.name

            candidate = _JOKER_NAME_TO_CLASS.get(
                name,
                RegularJoker
            )()

            # brute-force score
            hand_score, _ = _best_hand(
                my_hand,
                my_jokers + [candidate]
            )

            # meta ranking
            priority = JOKER_PRIORITY.get(name, 1000)

            # synergy scaling
            synergy = self._synergy_bonus(
                owned_names,
                name
            )

            total_value = (
                hand_score
                + priority
                + synergy
            )

            if total_value > best_value:
                best_value = total_value
                best_idx = i

        return best_idx

    def pick_hand(self, state: GameState) -> List[int]:

        is_p1 = state.current_turn == PlayerTurn.PLAYER1

        my_hand = _to_cards(
            state.player1_hand if is_p1 else state.player2_hand
        )

        my_jokers = _to_jokers(
            state.player1_jokers if is_p1 else state.player2_jokers
        )

        _, indices = _best_hand(
            my_hand,
            my_jokers
        )

        return indices


Bot = MetaGreedyBot