"""
Starter-bot utilities for working with `GameState`.

These helpers do three common jobs:
1. Convert serialized `GameState` models into live engine objects.
2. Answer "what is my best current hand?" questions exactly.
3. Measure how much a hypothetical new joker or card would improve that hand.
"""

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence

from stellatro_common import CardModel, GameState, JokerModel, PlayerTurn
from stellatro_game import Card, Checker, HandType, Joker, PLAYER_CARDS, Suit, evaluate_hand
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker


Hand = List[Card]
JokerList = List[Joker]
_JOKER_NAME_TO_CLASS = {joker_cls.name: joker_cls for joker_cls in ALL_JOKER_CLASSES}


@dataclass(frozen=True)
class HandSummary:
    """Exact summary of the best 5-card hand currently available."""

    indices: tuple[int, ...]
    cards: tuple[Card, ...]
    hand_type: HandType
    score: int


def resolved_player_turn(
    state: GameState,
    player_turn: PlayerTurn | None = None,
) -> PlayerTurn:
    """
    Pick a concrete player turn.

    If the caller does not specify one, use `state.current_turn`. If the state
    has no active turn, default to Player 1 for convenience.
    """

    if player_turn is not None:
        return player_turn
    return state.current_turn or PlayerTurn.PLAYER1


def opponent_of(player_turn: PlayerTurn) -> PlayerTurn:
    """Return the opposing player."""

    if player_turn == PlayerTurn.PLAYER1:
        return PlayerTurn.PLAYER2
    return PlayerTurn.PLAYER1


def card_from_model(card_model: CardModel) -> Card:
    """Convert a serialized card model into the engine's mutable `Card`."""

    suits = [Suit(suit) for suit in card_model.suits]
    if not suits:
        raise ValueError("CardModel must include at least one suit.")

    card = Card(card_model.rank, suits[0])
    for suit in suits[1:]:
        card.add_suit(suit)
    card.scored = card_model.scored
    card.num_triggers = card_model.num_triggers
    return card


def joker_from_model(joker_model: JokerModel) -> Joker:
    """Convert a serialized joker model into the engine's `Joker` object."""

    joker_cls = _JOKER_NAME_TO_CLASS.get(joker_model.name, RegularJoker)
    return joker_cls()


def cards_from_models(card_models: Sequence[CardModel]) -> Hand:
    """Convert a list of `CardModel` objects into engine `Card` objects."""

    return [card_from_model(card_model) for card_model in card_models]


def jokers_from_models(joker_models: Sequence[JokerModel]) -> JokerList:
    """Convert a list of `JokerModel` objects into engine `Joker` objects."""

    return [joker_from_model(joker_model) for joker_model in joker_models]


def hand_for_player(
    state: GameState,
    player_turn: PlayerTurn | None = None,
) -> Hand:
    """Return one player's current hand as engine `Card` objects."""

    player_turn = resolved_player_turn(state, player_turn)
    if player_turn == PlayerTurn.PLAYER1:
        return cards_from_models(state.player1_hand)
    return cards_from_models(state.player2_hand)


def jokers_for_player(
    state: GameState,
    player_turn: PlayerTurn | None = None,
) -> JokerList:
    """Return one player's drafted jokers as engine `Joker` objects."""

    player_turn = resolved_player_turn(state, player_turn)
    if player_turn == PlayerTurn.PLAYER1:
        return jokers_from_models(state.player1_jokers)
    return jokers_from_models(state.player2_jokers)


def current_player_hand(state: GameState) -> Hand:
    """Shortcut for the active player's hand."""

    return hand_for_player(state, state.current_turn)


def current_player_jokers(state: GameState) -> JokerList:
    """Shortcut for the active player's drafted jokers."""

    return jokers_for_player(state, state.current_turn)


def iter_5_card_indices(hand: Sequence[Card]) -> Iterable[tuple[int, ...]]:
    """Yield every legal 5-card subset by index."""

    playable_cards = min(PLAYER_CARDS, len(hand))
    if playable_cards < 5:
        return
    yield from combinations(range(playable_cards), 5)


def hand_type_for_cards(cards: Sequence[Card]) -> HandType:
    """Classify a 5-card hand without joker scoring."""

    card_copies = [deepcopy(card) for card in cards]
    return Checker(card_copies).check()


def score_hand(cards: Sequence[Card], jokers: Sequence[Joker]) -> int:
    """Score a specific 5-card hand with the supplied jokers."""

    return evaluate_hand(
        [deepcopy(card) for card in cards],
        deepcopy(list(jokers)),
    )


def best_hand_summary(
    hand: Sequence[Card],
    jokers: Sequence[Joker],
) -> HandSummary:
    """
    Return the exact best 5-card hand available from the given cards and jokers.
    """

    best_indices: tuple[int, ...] | None = None
    best_cards: tuple[Card, ...] = ()
    best_type = HandType.HIGH_CARD
    best_score = -1

    for indices in iter_5_card_indices(hand):
        chosen_cards = tuple(hand[index] for index in indices)
        hand_score = score_hand(chosen_cards, jokers)
        if hand_score > best_score:
            best_indices = indices
            best_cards = chosen_cards
            best_type = hand_type_for_cards(chosen_cards)
            best_score = hand_score

    if best_indices is None:
        return HandSummary(indices=(), cards=(), hand_type=HandType.HIGH_CARD, score=0)

    return HandSummary(
        indices=best_indices,
        cards=best_cards,
        hand_type=best_type,
        score=int(best_score),
    )


def best_hand_for_player(
    state: GameState,
    player_turn: PlayerTurn | None = None,
) -> HandSummary:
    """Return the exact best current hand for a player in this state."""

    player_turn = resolved_player_turn(state, player_turn)
    return best_hand_summary(
        hand_for_player(state, player_turn),
        jokers_for_player(state, player_turn),
    )


def best_current_hand(state: GameState) -> HandSummary:
    """Shortcut for the active player's best current hand."""

    return best_hand_for_player(state, state.current_turn)


def score_delta_if_pick_joker(
    state: GameState,
    joker_index: int,
    player_turn: PlayerTurn | None = None,
) -> int:
    """
    Return the exact increase in best-hand score if this player drafts a joker.

    This compares:
    - current best achievable hand score
    - best achievable hand score after adding `state.joker_pool[joker_index]`
    """

    player_turn = resolved_player_turn(state, player_turn)
    hand = hand_for_player(state, player_turn)
    jokers = jokers_for_player(state, player_turn)

    current_score = best_hand_summary(hand, jokers).score
    candidate_joker = joker_from_model(state.joker_pool[joker_index])
    improved_score = best_hand_summary(hand, jokers + [candidate_joker]).score
    return improved_score - current_score


def score_delta_if_add_card(
    state: GameState,
    card_model: CardModel,
    player_turn: PlayerTurn | None = None,
) -> int:
    """
    Return the exact increase in best-hand score if this player gained a card.

    This is useful for "what if I could add this card to my hand?" analysis,
    even though the current draft phase is about jokers rather than cards.
    """

    player_turn = resolved_player_turn(state, player_turn)
    hand = hand_for_player(state, player_turn)
    jokers = jokers_for_player(state, player_turn)

    current_score = best_hand_summary(hand, jokers).score
    improved_score = best_hand_summary(hand + [card_from_model(card_model)], jokers).score
    return improved_score - current_score

def print_card_list(cards: List[Card]) -> None:
    for card in cards:
        print(card, end=" ")
    print()  # newline at the end


def print_jokers(jokers: List) -> None:
    for joker in jokers:
        print(joker, end=" ")
    print()  # newline at the end


__all__ = [
    "HandSummary",
    "best_current_hand",
    "best_hand_for_player",
    "best_hand_summary",
    "card_from_model",
    "cards_from_models",
    "current_player_hand",
    "current_player_jokers",
    "hand_for_player",
    "hand_type_for_cards",
    "iter_5_card_indices",
    "joker_from_model",
    "jokers_for_player",
    "jokers_from_models",
    "opponent_of",
    "resolved_player_turn",
    "score_delta_if_add_card",
    "score_delta_if_pick_joker",
    "score_hand",
    "print_card_list",
    "print_jokers"
]
