import math
import time
from copy import deepcopy
from itertools import combinations
from typing import List, Tuple

from stellatro_common import CardModel, GameState, JokerModel, PlayerTurn
from stellatro_game import Card, JOKER_HAND_SIZE, Joker, PLAYER_CARDS, Suit, evaluate_hand
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

Hand = List[Card]

_JOKER_NAME_TO_CLASS = {joker_cls.name: joker_cls for joker_cls in ALL_JOKER_CLASSES}

# xMult jokers: the highest-impact denial targets. Taking these off the board
# hard-caps the opponent's ceiling regardless of what else they draft.
_XMULT_JOKER_NAMES = frozenset({
    "The Duo",
    "The Trio",
    "The Tribe",
    "The Order",
    "UC Socially Dead",
    "Flower Pot",
    "Sun God",
    "Plasma",
})


def _pick_number(state: GameState, player_turn: PlayerTurn) -> int:
    """Return which pick this is for us (0-indexed: 0 = our first pick)."""
    if player_turn == PlayerTurn.PLAYER1:
        return len(state.player1_jokers)
    return len(state.player2_jokers)


def _is_player1(state: GameState, player_turn: PlayerTurn) -> bool:
    return player_turn == PlayerTurn.PLAYER1


def _denial_weight(pick_num: int, is_p1: bool, joker_name: str) -> float:
    """
    Compute the denial weight for the greedy scoring formula:
        val = my_score + denial_weight * opp_gain

    Strategy:
      P1 pick 0: take the best thing for yourself — you have first access to
                 every xMult joker, so denial is unnecessary. Pure value.
      P1 pick 1: P2 has taken one joker; if it was threatening, consider denial.
                 Moderate weight.
      P1 picks 2-4: standard balanced play.

      P2 pick 0: P1 already took something good. Aggressively deny the next
                 best xMult joker — this is the most important denial pick in
                 the game. Very high weight, especially for xMult jokers.
      P2 pick 1: still early, still worth denying if an xMult is available.
      P2 picks 2-4: standard balanced play, denial fades as pool thins.
    """
    is_xmult = joker_name in _XMULT_JOKER_NAMES

    if is_p1:
        # P1 schedule: low → moderate → standard
        weights = [0.10, 0.25, 0.35, 0.35, 0.30]
    else:
        # P2 schedule: very high early (especially for xMult) → standard
        if is_xmult:
            weights = [1.80, 1.20, 0.50, 0.35, 0.30]
        else:
            weights = [0.80, 0.50, 0.35, 0.35, 0.30]

    idx = min(pick_num, len(weights) - 1)
    return weights[idx]


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


def _joker_cls_from_model(joker_model: JokerModel) -> type:
    return _JOKER_NAME_TO_CLASS.get(joker_model.name, RegularJoker)


def _hand_for_player(state: GameState, player_turn: PlayerTurn) -> Hand:
    if player_turn == PlayerTurn.PLAYER1:
        return [_card_from_model(card) for card in state.player1_hand]
    return [_card_from_model(card) for card in state.player2_hand]


def _joker_classes_for_player(state: GameState, player_turn: PlayerTurn) -> List[type]:
    if player_turn == PlayerTurn.PLAYER1:
        return [_joker_cls_from_model(joker) for joker in state.player1_jokers]
    return [_joker_cls_from_model(joker) for joker in state.player2_jokers]


def _get_diverse_candidates(cards: List[Card]) -> List[List[Card]]:
    """
    Precompute strategically diverse 5-card candidates.

    The old version mostly kept the top base-score hands, plus a couple of
    flush/straight hands. That can miss joker-dependent hands such as pairs,
    two pair, trips, full house, quads, and rank/suit/stella setups. This
    version still stays small, but forces coverage across poker archetypes.
    """
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return []

    def combo_key(subset: List[Card]) -> tuple:
        return tuple(
            sorted((c.rank, tuple(sorted(s.value for s in c.suits))) for c in subset)
        )

    def common_suit_count(subset: List[Card]) -> int:
        common = set(subset[0].suits)
        for c in subset[1:]:
            common &= set(c.suits)
        return len(common)

    def is_flush(subset: List[Card]) -> bool:
        return common_suit_count(subset) > 0

    def is_straight(subset: List[Card]) -> bool:
        ranks = sorted(c.rank for c in subset)
        if len(set(ranks)) == 5 and ranks[-1] - ranks[0] == 4:
            return True
        return ranks == [2, 3, 4, 5, 14]

    def rank_counts(subset: List[Card]) -> List[int]:
        counts = {}
        for c in subset:
            counts[c.rank] = counts.get(c.rank, 0) + 1
        return sorted(counts.values(), reverse=True)

    def category(subset: List[Card]) -> str:
        counts = rank_counts(subset)
        flush = is_flush(subset)
        straight = is_straight(subset)
        if straight and flush:
            return "straight_flush"
        if counts == [4, 1]:
            return "quads"
        if counts == [3, 2]:
            return "full_house"
        if flush:
            return "flush"
        if straight:
            return "straight"
        if counts == [3, 1, 1]:
            return "trips"
        if counts == [2, 2, 1]:
            return "two_pair"
        if counts == [2, 1, 1, 1]:
            return "pair"
        return "high_card"

    scored_combos: List[Tuple[int, str, List[Card]]] = []
    for combo in combinations(range(n), 5):
        subset = [cards[i] for i in combo]
        copied_subset = [deepcopy(c) for c in subset]
        try:
            base_score = evaluate_hand(copied_subset, [])
        except Exception:
            base_score = 0
        scored_combos.append((base_score, category(subset), subset))

    scored_combos.sort(key=lambda x: x[0], reverse=True)

    candidates: List[List[Card]] = []
    seen = set()

    def add(subset: List[Card]) -> None:
        key = combo_key(subset)
        if key not in seen:
            seen.add(key)
            candidates.append(subset)

    # 1. Keep the strongest raw hands.
    for _, _, subset in scored_combos[:10]:
        add(subset)

    # 2. Force coverage of every important poker archetype.
    archetype_order = [
        "straight_flush",
        "quads",
        "full_house",
        "flush",
        "straight",
        "trips",
        "two_pair",
        "pair",
        "high_card",
    ]
    for archetype in archetype_order:
        added_for_type = 0
        for _, cat, subset in scored_combos:
            if cat == archetype:
                add(subset)
                added_for_type += 1
            if added_for_type >= 2:
                break

    # 3. Add rank-dense hands that are often useful for stella/retrigger/copy effects.
    rank_dense = sorted(
        scored_combos,
        key=lambda x: (max(rank_counts(x[2])), sum(c.rank for c in x[2]), x[0]),
        reverse=True,
    )
    for _, _, subset in rank_dense[:8]:
        add(subset)

    # 4. Fill to a modest cap. C(8,5)=56, so 24 is still cheap but much safer.
    for _, _, subset in scored_combos:
        if len(candidates) >= 24:
            break
        add(subset)

    return candidates

def _best_hand_from_candidates(candidates: List[List[Card]], joker_classes: List[type]) -> int:
    """Return the best score across precomputed candidates in under 0.1ms."""
    best_score = 0
    for subset in candidates:
        copied_subset = [deepcopy(c) for c in subset]
        fresh_jokers = [cls() for cls in joker_classes]
        try:
            score = evaluate_hand(copied_subset, fresh_jokers)
        except Exception:
            continue
        if score > best_score:
            best_score = score
    return best_score


def _best_hand_exact(cards: List[Card], joker_classes: List[type]) -> tuple[int, List[int]]:
    """Brute force exact card play for play phase (only called once per turn)."""
    best_score = -1
    best_indices = list(range(5))
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return 0, []

    for combo in combinations(range(n), 5):
        copied_subset = [deepcopy(cards[i]) for i in combo]
        fresh_jokers = [cls() for cls in joker_classes]
        try:
            score = evaluate_hand(copied_subset, fresh_jokers)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_indices = list(combo)

    return max(0, best_score), best_indices


class Bot:
    def __init__(self, time_limit: float = 0.150) -> None:
        self.time_limit = time_limit
        self.search_start_time = 0.0
        self._score_cache: dict[tuple[int, tuple[str, ...]], int] = {}

    @staticmethod
    def _joker_name(joker_cls: type) -> str:
        return getattr(joker_cls, "name", joker_cls.__name__)

    def _best_score(self, candidates: List[List[Card]], joker_classes: List[type]) -> int:
        """Cached best score over candidate hands for this exact joker set/order."""
        key = (id(candidates), tuple(self._joker_name(cls) for cls in joker_classes))
        if key not in self._score_cache:
            self._score_cache[key] = _best_hand_from_candidates(candidates, joker_classes)
        return self._score_cache[key]

    def pick_joker(self, state: GameState) -> int:
        start_time = time.perf_counter()
        self.search_start_time = start_time
        self._score_cache.clear()

        player_turn = state.current_turn
        if player_turn not in (PlayerTurn.PLAYER1, PlayerTurn.PLAYER2):
            return 0

        my_hand = _hand_for_player(state, player_turn)
        opp_turn = (
            PlayerTurn.PLAYER2
            if player_turn == PlayerTurn.PLAYER1
            else PlayerTurn.PLAYER1
        )
        opp_hand = _hand_for_player(state, opp_turn)

        my_picks = _joker_classes_for_player(state, player_turn)
        opp_picks = _joker_classes_for_player(state, opp_turn)
        pool = [_joker_cls_from_model(joker) for joker in state.joker_pool]

        if not pool:
            return 0
        if len(pool) == 1:
            return 0

        # IMPORTANT: keep original pool indices attached to each joker.
        # Recursive minimax removes cards from the pool, so raw local indices are
        # not stable. Returning root_index prevents wrong-joker picks.
        pool_entries: List[Tuple[int, type]] = list(enumerate(pool))

        my_candidates = _get_diverse_candidates(my_hand)
        opp_candidates = _get_diverse_candidates(opp_hand)

        pick_num = _pick_number(state, player_turn)
        is_p1 = _is_player1(state, player_turn)

        greedy_choices = []
        opp_base = self._best_score(opp_candidates, opp_picks)
        for root_index, joker_cls in pool_entries:
            joker_name = getattr(state.joker_pool[root_index], "name", self._joker_name(joker_cls))
            my_score = self._best_score(my_candidates, my_picks + [joker_cls])
            opp_score = self._best_score(opp_candidates, opp_picks + [joker_cls])
            opp_gain = max(0, opp_score - opp_base)
            w = _denial_weight(pick_num, is_p1, joker_name)
            val = my_score + w * opp_gain
            greedy_choices.append((root_index, val, joker_cls))

        greedy_choices.sort(key=lambda x: x[1], reverse=True)
        best_index = greedy_choices[0][0]

        if time.perf_counter() - start_time > self.time_limit:
            return best_index

        best_idx_found = best_index
        try:
            for target_depth in (1, 2):
                _, idx = self._minimax(
                    my_candidates,
                    opp_candidates,
                    pool_entries,
                    my_picks,
                    opp_picks,
                    my_turn=True,
                    depth=0,
                    target_depth=target_depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    pick_num=pick_num,
                    is_p1=is_p1,
                )
                if idx is not None and 0 <= idx < len(state.joker_pool):
                    best_idx_found = idx
        except TimeoutError:
            pass

        return best_idx_found

    def pick_hand(self, state: GameState) -> List[int]:
        player_turn = state.current_turn or PlayerTurn.PLAYER1
        hand = _hand_for_player(state, player_turn)
        joker_classes = _joker_classes_for_player(state, player_turn)
        _, indices = _best_hand_exact(hand, joker_classes)
        playable_cards = min(PLAYER_CARDS, len(hand))
        unique_in_range = []
        for idx in indices:
            if 0 <= idx < playable_cards and idx not in unique_in_range:
                unique_in_range.append(idx)
        for idx in range(playable_cards):
            if len(unique_in_range) == 5:
                break
            if idx not in unique_in_range:
                unique_in_range.append(idx)
        return unique_in_range[:5]

    def _rank_moves(
        self,
        my_candidates: List[List[Card]],
        opp_candidates: List[List[Card]],
        pool_entries: List[Tuple[int, type]],
        my_picks: List[type],
        opp_picks: List[type],
        my_turn: bool,
        current_pick: int,
        is_p1: bool,
    ) -> List[Tuple[int, float, type]]:
        """Return moves as (original_root_index, heuristic_value, joker_cls)."""
        candidates: List[Tuple[int, float, type]] = []
        if my_turn:
            opp_base = self._best_score(opp_candidates, opp_picks)
            for root_index, joker_cls in pool_entries:
                joker_name = self._joker_name(joker_cls)
                score = self._best_score(my_candidates, my_picks + [joker_cls])
                opp_score = self._best_score(opp_candidates, opp_picks + [joker_cls])
                opp_gain = max(0, opp_score - opp_base)
                w = _denial_weight(current_pick, is_p1, joker_name)
                candidates.append((root_index, score + w * opp_gain, joker_cls))
        else:
            my_base = self._best_score(my_candidates, my_picks)
            for root_index, joker_cls in pool_entries:
                joker_name = self._joker_name(joker_cls)
                score = self._best_score(opp_candidates, opp_picks + [joker_cls])
                my_score = self._best_score(my_candidates, my_picks + [joker_cls])
                my_gain = max(0, my_score - my_base)
                w = _denial_weight(current_pick, not is_p1, joker_name)
                candidates.append((root_index, score + w * my_gain, joker_cls))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _minimax(
        self,
        my_candidates: List[List[Card]],
        opp_candidates: List[List[Card]],
        pool_entries: List[Tuple[int, type]],
        my_picks: List[type],
        opp_picks: List[type],
        my_turn: bool,
        depth: int,
        target_depth: int,
        alpha: float,
        beta: float,
        pick_num: int = 0,
        is_p1: bool = True,
    ) -> tuple[float, int | None]:
        if time.perf_counter() - self.search_start_time > self.time_limit:
            raise TimeoutError()

        if len(my_picks) == JOKER_HAND_SIZE and len(opp_picks) == JOKER_HAND_SIZE:
            my_score = self._best_score(my_candidates, my_picks)
            opp_score = self._best_score(opp_candidates, opp_picks)
            return float(my_score - opp_score), None

        if depth >= target_depth or not pool_entries:
            my_score = self._best_score(my_candidates, my_picks)
            opp_score = self._best_score(opp_candidates, opp_picks)
            return float(my_score - opp_score), None

        # Slightly wider beam early, because combo jokers may look weak at first.
        K = 5 if depth == 0 else 4
        current_pick = pick_num + depth
        top_candidates = self._rank_moves(
            my_candidates,
            opp_candidates,
            pool_entries,
            my_picks,
            opp_picks,
            my_turn,
            current_pick,
            is_p1,
        )[:K]

        best_root_idx = None
        if my_turn:
            best_val = -math.inf
            for root_index, _, joker_cls in top_candidates:
                next_pool = [(i, cls) for i, cls in pool_entries if i != root_index]
                val, child_root = self._minimax(
                    my_candidates,
                    opp_candidates,
                    next_pool,
                    my_picks + [joker_cls],
                    opp_picks,
                    my_turn=False,
                    depth=depth + 1,
                    target_depth=target_depth,
                    alpha=alpha,
                    beta=beta,
                    pick_num=pick_num,
                    is_p1=is_p1,
                )
                if val > best_val:
                    best_val = val
                    # At the root, return the actual original joker index we are choosing.
                    # Below the root, propagate any descendant root index if present,
                    # otherwise keep this move's original index.
                    best_root_idx = root_index if depth == 0 else (child_root if child_root is not None else root_index)
                alpha = max(alpha, best_val)
                if alpha >= beta:
                    break
            return best_val, best_root_idx

        best_val = math.inf
        for root_index, _, joker_cls in top_candidates:
            next_pool = [(i, cls) for i, cls in pool_entries if i != root_index]
            val, child_root = self._minimax(
                my_candidates,
                opp_candidates,
                next_pool,
                my_picks,
                opp_picks + [joker_cls],
                my_turn=True,
                depth=depth + 1,
                target_depth=target_depth,
                alpha=alpha,
                beta=beta,
                pick_num=pick_num,
                is_p1=is_p1,
            )
            if val < best_val:
                best_val = val
                best_root_idx = child_root
            beta = min(beta, best_val)
            if alpha >= beta:
                break
        return best_val, best_root_idx


ParticipantBot = Bot
