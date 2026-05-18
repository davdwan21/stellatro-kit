import math
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence

from stellatro_common import CardModel, GameState, JokerModel, PlayerTurn
from stellatro_game import Card, JOKER_HAND_SIZE, Joker, PLAYER_CARDS, Suit, evaluate_hand
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

Hand = List[Card]
JokerList = List[Joker]

_JOKER_NAME_TO_CLASS = {joker_cls.name: joker_cls for joker_cls in ALL_JOKER_CLASSES}
DEFAULT_DRAFT_SUBSET_LIMIT = 3


def _card_from_model(card_model: CardModel) -> Card:
    suits = [Suit(suit) for suit in card_model.suits]
    if not suits:
        raise ValueError("CardModel must include at least one suit.")

    card = Card(card_model.rank, suits[0])
    for suit in suits[1:]:
        card.add_suit(suit)
    card.scored = card_model.scored
    card.num_triggers = card_model.num_triggers
    return card


def _joker_from_model(joker_model: JokerModel) -> Joker:
    joker_cls = _JOKER_NAME_TO_CLASS.get(joker_model.name, RegularJoker)
    return joker_cls()


def _hand_for_player(state: GameState, player_turn: PlayerTurn) -> Hand:
    if player_turn == PlayerTurn.PLAYER1:
        return [_card_from_model(card) for card in state.player1_hand]
    return [_card_from_model(card) for card in state.player2_hand]


def _jokers_for_player(state: GameState, player_turn: PlayerTurn) -> JokerList:
    if player_turn == PlayerTurn.PLAYER1:
        return [_joker_from_model(joker) for joker in state.player1_jokers]
    return [_joker_from_model(joker) for joker in state.player2_jokers]


def _normalize_play_indices(indices: Sequence[int], hand_size: int) -> List[int]:
    playable_cards = min(PLAYER_CARDS, hand_size)
    unique_in_range: List[int] = []

    for index in indices:
        if 0 <= index < playable_cards and index not in unique_in_range:
            unique_in_range.append(index)
        if len(unique_in_range) == 5:
            return unique_in_range

    for index in range(playable_cards):
        if index not in unique_in_range:
            unique_in_range.append(index)
        if len(unique_in_range) == 5:
            break

    return unique_in_range[:5]


def _all_5_card_subsets(hand: Sequence[Card]) -> Iterable[tuple[int, ...]]:
    if len(hand) < 5:
        return
    yield from combinations(range(len(hand)), 5)


def _iter_play_values(
    hand: Sequence[Card], jokers: Sequence[Joker]
) -> Iterable[tuple[tuple[int, ...], int]]:
    for indices in _all_5_card_subsets(hand):
        subset = [deepcopy(hand[index]) for index in indices]
        try:
            yield indices, evaluate_hand(subset, deepcopy(list(jokers)))
        except Exception:
            continue


def _play_value_for_indices(
    hand: Sequence[Card], jokers: Sequence[Joker], indices: Sequence[int]
) -> int:
    subset = [deepcopy(hand[index]) for index in indices]
    try:
        return evaluate_hand(subset, deepcopy(list(jokers)))
    except Exception:
        return 0


def _candidate_play_subsets(
    hand: Hand, jokers: JokerList, limit: int = DEFAULT_DRAFT_SUBSET_LIMIT
) -> tuple[tuple[int, ...], ...]:
    scored_subsets = sorted(
        _iter_play_values(hand, jokers),
        key=lambda item: item[1],
        reverse=True,
    )
    if limit <= 0:
        return tuple(indices for indices, _ in scored_subsets)
    return tuple(indices for indices, _ in scored_subsets[:limit])


def _approximate_best_play_value(
    hand: Hand,
    jokers: JokerList,
    candidate_subsets: Sequence[tuple[int, ...]],
) -> int:
    if not candidate_subsets:
        # exact search fallback if subset is not cached
        best_value = -math.inf
        for _, value in _iter_play_values(hand, jokers):
            if value > best_value:
                best_value = value
        return 0 if best_value == -math.inf else int(best_value)
    return max(_play_value_for_indices(hand, jokers, indices) for indices in candidate_subsets)


def _best_play_indices(hand: Hand, jokers: JokerList) -> List[int]:
    best_value = -math.inf
    best_indices: tuple[int, ...] | None = None

    for indices, value in _iter_play_values(hand, jokers):
        if value > best_value:
            best_value = value
            best_indices = indices

    if best_indices is None:
        return []
    return list(best_indices)


@dataclass(frozen=True)
class DraftState:
    remaining: tuple[Joker, ...]
    my_picks: tuple[Joker, ...]
    opp_picks: tuple[Joker, ...]
    my_turn: bool


class Bot:
    def __init__(
        self,
        max_depth: int = 3,
        draft_subset_limit: int = DEFAULT_DRAFT_SUBSET_LIMIT,
        time_limit: float = 0.065,  # 65 ms threshold to comply with 0.1s timeout
    ) -> None:
        self.max_depth = max_depth
        self.draft_subset_limit = draft_subset_limit
        self.time_limit = time_limit
        self.search_start_time = 0.0

    def pick_joker(self, state: GameState) -> int:
        start_time = time.perf_counter()
        player_turn = state.current_turn
        if player_turn not in (PlayerTurn.PLAYER1, PlayerTurn.PLAYER2):
            return 0

        # Reconstruct engine objects
        my_hand = _hand_for_player(state, player_turn)
        opp_turn = (
            PlayerTurn.PLAYER2
            if player_turn == PlayerTurn.PLAYER1
            else PlayerTurn.PLAYER1
        )
        opp_hand = _hand_for_player(state, opp_turn)
        my_picks = tuple(_jokers_for_player(state, player_turn))
        opp_picks = tuple(_jokers_for_player(state, opp_turn))
        pool = tuple(_joker_from_model(joker) for joker in state.joker_pool)

        # Precompute the top promising card play subsets to speed up search evaluations
        my_candidate_subsets = _candidate_play_subsets(
            my_hand, list(my_picks), self.draft_subset_limit
        )
        opp_candidate_subsets = _candidate_play_subsets(
            opp_hand, list(opp_picks), self.draft_subset_limit
        )

        # 1. Greedy Heuristic / Move Ordering
        greedy_choices = []
        for index, joker in enumerate(pool):
            my_score = _approximate_best_play_value(
                my_hand, list(my_picks) + [joker], my_candidate_subsets
            )
            opp_score = _approximate_best_play_value(
                opp_hand, list(opp_picks) + [joker], opp_candidate_subsets
            )
            greedy_choices.append((index, joker, my_score, opp_score))

        # Default fallback is the best immediate greedy choice (maximizing our own advantage)
        best_index = 0
        best_val = -math.inf
        for index, _, my_score, opp_score in greedy_choices:
            val = my_score - opp_score
            if val > best_val:
                best_val = val
                best_index = index

        if len(pool) <= 1:
            return 0

        self.search_start_time = start_time
        best_idx_found = best_index

        # 2. Iterative Deepening Minimax with Move Ordering and hard Timeout check
        try:
            for target_depth in range(1, self.max_depth + 1):
                best_value = -math.inf
                best_index_for_depth = best_index

                # Sort available jokers greedily (best for us first)
                sorted_choices = sorted(
                    greedy_choices,
                    key=lambda x: (x[2], -x[3]),
                    reverse=True,
                )

                for index, joker, _, _ in sorted_choices:
                    if time.perf_counter() - start_time > self.time_limit:
                        raise TimeoutError()

                    next_state = DraftState(
                        remaining=pool[:index] + pool[index + 1 :],
                        my_picks=my_picks + (joker,),
                        opp_picks=opp_picks,
                        my_turn=False,
                    )
                    value = self._minimax(
                        my_hand,
                        opp_hand,
                        next_state,
                        -math.inf,
                        math.inf,
                        depth=1,
                        target_depth=target_depth,
                        my_candidate_subsets=my_candidate_subsets,
                        opp_candidate_subsets=opp_candidate_subsets,
                    )

                    if value > best_value:
                        best_value = value
                        best_index_for_depth = index

                # Complete iteration fully -> update choice
                best_idx_found = best_index_for_depth
        except TimeoutError:
            pass

        return best_idx_found

    def pick_hand(self, state: GameState) -> List[int]:
        player_turn = state.current_turn or PlayerTurn.PLAYER1
        hand = _hand_for_player(state, player_turn)
        jokers = _jokers_for_player(state, player_turn)
        return _normalize_play_indices(_best_play_indices(hand, jokers), len(hand))

    def _leaf_value(
        self,
        my_hand: Hand,
        opp_hand: Hand,
        state: DraftState,
        my_candidate_subsets: Sequence[tuple[int, ...]],
        opp_candidate_subsets: Sequence[tuple[int, ...]],
    ) -> float:
        my_best = _approximate_best_play_value(
            my_hand, list(state.my_picks), my_candidate_subsets
        )
        opp_best = _approximate_best_play_value(
            opp_hand, list(state.opp_picks), opp_candidate_subsets
        )
        return float(my_best - opp_best)

    def _heuristic_value(
        self,
        my_hand: Hand,
        opp_hand: Hand,
        state: DraftState,
        my_candidate_subsets: Sequence[tuple[int, ...]],
        opp_candidate_subsets: Sequence[tuple[int, ...]],
    ) -> float:
        my_now = _approximate_best_play_value(
            my_hand, list(state.my_picks), my_candidate_subsets
        )
        opp_now = _approximate_best_play_value(
            opp_hand, list(state.opp_picks), opp_candidate_subsets
        )
        return float(my_now - opp_now)

    def _minimax(
        self,
        my_hand: Hand,
        opp_hand: Hand,
        state: DraftState,
        alpha: float,
        beta: float,
        depth: int,
        target_depth: int,
        my_candidate_subsets: Sequence[tuple[int, ...]],
        opp_candidate_subsets: Sequence[tuple[int, ...]],
    ) -> float:
        if time.perf_counter() - self.search_start_time > self.time_limit:
            raise TimeoutError()

        if (
            len(state.my_picks) == JOKER_HAND_SIZE
            and len(state.opp_picks) == JOKER_HAND_SIZE
        ):
            return self._leaf_value(
                my_hand,
                opp_hand,
                state,
                my_candidate_subsets,
                opp_candidate_subsets,
            )

        if depth >= target_depth:
            return self._heuristic_value(
                my_hand,
                opp_hand,
                state,
                my_candidate_subsets,
                opp_candidate_subsets,
            )

        # Move Ordering inside search layers
        candidates = []
        for index, joker in enumerate(state.remaining):
            if state.my_turn:
                score = _approximate_best_play_value(
                    my_hand, list(state.my_picks) + [joker], my_candidate_subsets
                )
                candidates.append((index, joker, score))
            else:
                score = _approximate_best_play_value(
                    opp_hand, list(state.opp_picks) + [joker], opp_candidate_subsets
                )
                candidates.append((index, joker, score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        if state.my_turn:
            best_value = -math.inf
            for index, joker, _ in candidates:
                next_state = DraftState(
                    remaining=state.remaining[:index] + state.remaining[index + 1 :],
                    my_picks=state.my_picks + (joker,),
                    opp_picks=state.opp_picks,
                    my_turn=False,
                )
                value = self._minimax(
                    my_hand,
                    opp_hand,
                    next_state,
                    alpha,
                    beta,
                    depth + 1,
                    target_depth,
                    my_candidate_subsets,
                    opp_candidate_subsets,
                )
                if value > best_value:
                    best_value = value
                if best_value > alpha:
                    alpha = best_value
                if alpha >= beta:
                    break
            return best_value

        best_value = math.inf
        for index, joker, _ in candidates:
            next_state = DraftState(
                remaining=state.remaining[:index] + state.remaining[index + 1 :],
                my_picks=state.my_picks,
                opp_picks=state.opp_picks + (joker,),
                my_turn=True,
            )
            value = self._minimax(
                my_hand,
                opp_hand,
                next_state,
                alpha,
                beta,
                depth + 1,
                target_depth,
                my_candidate_subsets,
                opp_candidate_subsets,
            )
            if value < best_value:
                best_value = value
            if best_value < beta:
                beta = best_value
            if alpha >= beta:
                break
        return best_value
    

Bot = Bot