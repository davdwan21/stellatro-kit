import math
import time
from copy import deepcopy
from itertools import combinations
from typing import List

from stellatro_common import CardModel, GameState, JokerModel, PlayerTurn
from stellatro_game import Card, JOKER_HAND_SIZE, Joker, PLAYER_CARDS, Suit, evaluate_hand
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

Hand = List[Card]

_JOKER_NAME_TO_CLASS = {joker_cls.name: joker_cls for joker_cls in ALL_JOKER_CLASSES}

# xMult jokers: taking these off the board hard-caps the opponent's ceiling.
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

# Jokers that reward flush hands.
_FLUSH_JOKER_NAMES = frozenset({
    "Flower Pot",
    "Smeared Joker",
    "Suit Joker",
})

# Jokers that reward straight hands.
_STRAIGHT_JOKER_NAMES = frozenset({
    "The Order",
    "Straight Shooter",
})

# Jokers that reward pair / multi-pair hands.
_PAIR_JOKER_NAMES = frozenset({
    "The Duo",
    "The Trio",
    "The Tribe",
})

# Flat-chip consistency jokers that Prakar-style bots exploit.
# We treat them similarly to xMult for denial purposes.
_CHIP_JOKER_NAMES = frozenset({
    "Spare Trousers",
    "Bootstrapper",
    "Egg",
    "Throwback",
    "Runner",
    "Supernova",
    "Dusk",
})


def _pick_number(state: GameState, player_turn: PlayerTurn) -> int:
    if player_turn == PlayerTurn.PLAYER1:
        return len(state.player1_jokers)
    return len(state.player2_jokers)


def _is_player1(state: GameState, player_turn: PlayerTurn) -> bool:
    return player_turn == PlayerTurn.PLAYER1


def _denial_weight(pick_num: int, is_p1: bool, joker_name: str) -> float:
    """
    Greedy scoring: val = my_score + denial_weight * opp_gain

    Changes from v1:
    - P2 pick-0 xMult weight reduced 1.80 → 1.35 (was too aggressive, crowding
      out real value and causing the P2 seat-gap observed in benchmarks).
    - Chip jokers now use a moderate denial path instead of the flat default.
    - Late picks (idx 3-4) kept at 0.30 — pool is thin, just take value.
    """
    is_xmult = joker_name in _XMULT_JOKER_NAMES
    is_chip  = joker_name in _CHIP_JOKER_NAMES

    if is_p1:
        weights = [0.10, 0.25, 0.35, 0.35, 0.30]
    else:
        if is_xmult:
            weights = [1.35, 1.00, 0.50, 0.35, 0.30]  # was [1.80, 1.20, ...]
        elif is_chip:
            weights = [0.60, 0.45, 0.35, 0.30, 0.25]
        else:
            weights = [0.70, 0.45, 0.35, 0.35, 0.30]  # was [0.80, 0.50, ...]

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


# ---------------------------------------------------------------------------
# Hand-type detection helpers (cheap, no evaluate_hand call needed)
# ---------------------------------------------------------------------------

def _flush_potential(cards: List[Card]) -> float:
    """Fraction of cards that share the most common suit."""
    if not cards:
        return 0.0
    suit_counts: dict = {}
    for c in cards:
        for s in c.suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
    return max(suit_counts.values()) / len(cards)


def _straight_potential(cards: List[Card]) -> float:
    """How close the hand is to a straight (0–1)."""
    if len(cards) < 2:
        return 0.0
    ranks = sorted(set(c.rank for c in cards))
    # sliding window of 5
    best = 0
    for i in range(len(ranks)):
        window = [r for r in ranks if ranks[i] <= r <= ranks[i] + 4]
        best = max(best, len(window))
    # Ace-low straight (A-2-3-4-5)
    if 14 in ranks:
        low_ranks = [r for r in ranks if r <= 5] + [1]
        window = [r for r in low_ranks if 1 <= r <= 5]
        best = max(best, len(window))
    return best / 5.0


def _pair_potential(cards: List[Card]) -> int:
    """Number of ranks that appear 2+ times."""
    rank_counts: dict = {}
    for c in cards:
        rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1
    return sum(1 for v in rank_counts.values() if v >= 2)


def _synergy_bonus(joker_cls: type, cards: List[Card]) -> float:
    """
    Lightweight bonus added to the greedy val score when a joker matches the
    hand's natural tendencies.  Intentionally conservative — just enough to
    break ties and avoid drafting flush jokers into a pair-heavy hand.

    Returns a score in the same units as evaluate_hand output.
    """
    name = getattr(joker_cls, "name", "")
    bonus = 0.0

    if name in _FLUSH_JOKER_NAMES:
        fp = _flush_potential(cards)
        # Strong bonus only when ≥3/N cards share a suit
        if fp >= 0.6:
            bonus += 800.0 * fp
        elif fp >= 0.4:
            bonus += 300.0 * fp

    if name in _STRAIGHT_JOKER_NAMES:
        sp = _straight_potential(cards)
        if sp >= 0.8:
            bonus += 800.0 * sp
        elif sp >= 0.6:
            bonus += 300.0 * sp

    if name in _PAIR_JOKER_NAMES:
        pp = _pair_potential(cards)
        if pp >= 2:
            bonus += 600.0
        elif pp >= 1:
            bonus += 200.0

    return bonus


# ---------------------------------------------------------------------------
# Candidate generation — now joker-aware
# ---------------------------------------------------------------------------

def _get_diverse_candidates(
    cards: List[Card],
    joker_classes: List[type] = None,
    max_candidates: int = 20,          # increased from 12
) -> List[List[Card]]:
    """
    Return up to `max_candidates` strategically diverse 5-card combos.

    Changes from v1:
    - Pool size increased 12 → 20 to reduce synergy-miss collapses.
    - joker_classes parameter: when provided, the flush/straight/pair slot
      budgets are scaled up to match the joker portfolio, ensuring the
      candidate pool actually covers the hand types the jokers reward.
    - Base-score slots reduced from 8 → 6 to make room for typed candidates.
    """
    joker_classes = joker_classes or []
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return []

    # How many jokers of each type do we hold?
    flush_jokers    = sum(1 for j in joker_classes if getattr(j, "name", "") in _FLUSH_JOKER_NAMES)
    straight_jokers = sum(1 for j in joker_classes if getattr(j, "name", "") in _STRAIGHT_JOKER_NAMES)
    pair_jokers     = sum(1 for j in joker_classes if getattr(j, "name", "") in _PAIR_JOKER_NAMES)

    # Slot budgets — scale with how many synergy jokers we hold
    base_slots     = 6
    flush_slots    = 2 + flush_jokers * 2      # 2 baseline + 2 per flush joker
    straight_slots = 2 + straight_jokers * 2
    pair_slots     = 1 + pair_jokers            # pairs usually covered by base score

    scored_combos = []
    for combo in combinations(range(n), 5):
        subset = [cards[i] for i in combo]
        copied = [deepcopy(c) for c in subset]
        try:
            score = evaluate_hand(copied, [])
        except Exception:
            score = 0
        scored_combos.append((score, subset))

    scored_combos.sort(key=lambda x: x[0], reverse=True)

    candidates = []
    seen_keys: set = set()

    def combo_key(subset):
        return tuple(sorted(
            (c.rank, tuple(sorted(s.value for s in c.suits)))
            for c in subset
        ))

    def is_flush(subset):
        common = set(subset[0].suits)
        for c in subset[1:]:
            common &= set(c.suits)
            if not common:
                return False
        return True

    def is_straight(subset):
        ranks = sorted(c.rank for c in subset)
        if len(set(ranks)) == 5 and ranks[-1] - ranks[0] == 4:
            return True
        return ranks == [2, 3, 4, 5, 14]

    def is_pair_heavy(subset):
        rc: dict = {}
        for c in subset:
            rc[c.rank] = rc.get(c.rank, 0) + 1
        return sum(1 for v in rc.values() if v >= 2) >= 2

    def add(subset):
        k = combo_key(subset)
        if k not in seen_keys:
            seen_keys.add(k)
            candidates.append(subset)
            return True
        return False

    # 1. Top base-score combos
    for score, subset in scored_combos[:base_slots]:
        add(subset)

    # 2. Flush candidates
    flush_added = 0
    for _, subset in scored_combos:
        if flush_added >= flush_slots:
            break
        if is_flush(subset) and add(subset):
            flush_added += 1

    # 3. Straight candidates
    straight_added = 0
    for _, subset in scored_combos:
        if straight_added >= straight_slots:
            break
        if is_straight(subset) and add(subset):
            straight_added += 1

    # 4. Pair-heavy candidates (for pair jokers)
    pair_added = 0
    for _, subset in scored_combos:
        if pair_added >= pair_slots:
            break
        if is_pair_heavy(subset) and add(subset):
            pair_added += 1

    # 5. Fill remaining slots with best unseen combos
    for _, subset in scored_combos:
        if len(candidates) >= max_candidates:
            break
        add(subset)

    return candidates


def _best_hand_from_candidates(
    candidates: List[List[Card]],
    joker_classes: List[type],
) -> int:
    best = 0
    for subset in candidates:
        copied = [deepcopy(c) for c in subset]
        fresh_jokers = [cls() for cls in joker_classes]
        try:
            score = evaluate_hand(copied, fresh_jokers)
        except Exception:
            continue
        if score > best:
            best = score
    return best


def _best_hand_exact(
    cards: List[Card],
    joker_classes: List[type],
) -> tuple[int, List[int]]:
    """Brute-force exact best hand for play phase."""
    best_score = -1
    best_indices = list(range(5))
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return 0, []

    for combo in combinations(range(n), 5):
        copied = [deepcopy(cards[i]) for i in combo]
        fresh_jokers = [cls() for cls in joker_classes]
        try:
            score = evaluate_hand(copied, fresh_jokers)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_indices = list(combo)

    return max(0, best_score), best_indices


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self, time_limit: float = 0.175) -> None:
        self.time_limit = time_limit
        self.search_start_time = 0.0

    def pick_joker(self, state: GameState) -> int:
        start_time = time.perf_counter()
        self.search_start_time = start_time
        player_turn = state.current_turn
        if player_turn not in (PlayerTurn.PLAYER1, PlayerTurn.PLAYER2):
            return 0

        opp_turn = (
            PlayerTurn.PLAYER2 if player_turn == PlayerTurn.PLAYER1
            else PlayerTurn.PLAYER1
        )

        my_hand  = _hand_for_player(state, player_turn)
        opp_hand = _hand_for_player(state, opp_turn)

        my_picks  = _joker_classes_for_player(state, player_turn)
        opp_picks = _joker_classes_for_player(state, opp_turn)
        pool = [_joker_cls_from_model(joker) for joker in state.joker_pool]

        if not pool:
            return 0
        if len(pool) == 1:
            return 0

        pick_num = _pick_number(state, player_turn)
        is_p1    = _is_player1(state, player_turn)

        # Joker-aware candidate generation: include the jokers we're
        # *considering* picking when building the candidate pool for scoring.
        # This prevents synergy misses where the best hand type for a joker
        # isn't represented in the top-12 base-score candidates.
        opp_candidates = _get_diverse_candidates(opp_hand, opp_picks)

        # Phase 1: greedy pass with synergy bonus
        # val = my_score + synergy_bonus + denial_weight * opp_gain
        greedy_choices = []
        opp_base = _best_hand_from_candidates(opp_candidates, opp_picks)

        for index, joker_model in enumerate(state.joker_pool):
            joker_cls  = pool[index]
            joker_name = joker_model.name

            # Build my candidates conditioned on this candidate joker
            my_candidates_j = _get_diverse_candidates(my_hand, my_picks + [joker_cls])
            my_score  = _best_hand_from_candidates(my_candidates_j, my_picks + [joker_cls])

            opp_score = _best_hand_from_candidates(opp_candidates, opp_picks + [joker_cls])
            opp_gain  = max(0, opp_score - opp_base)

            w       = _denial_weight(pick_num, is_p1, joker_name)
            syn     = _synergy_bonus(joker_cls, my_hand)
            val     = my_score + syn + w * opp_gain

            greedy_choices.append((index, val, joker_cls, my_candidates_j))

        greedy_choices.sort(key=lambda x: x[1], reverse=True)
        best_index = greedy_choices[0][0]

        if time.perf_counter() - start_time > self.time_limit:
            return best_index

        # Phase 2: pruned minimax with iterative deepening.
        # Beam width: K=5 at depth 1, K=3 at depth 2.
        # Pre-cache my candidates for each greedy-top joker to avoid
        # recomputing them inside the recursive search.
        cached_my_candidates = {
            entry[0]: entry[3] for entry in greedy_choices
        }

        best_idx_found = best_index
        try:
            for target_depth, beam_k in ((1, 5), (2, 3)):
                val, idx = self._minimax(
                    my_hand=my_hand,
                    opp_candidates=opp_candidates,
                    pool=pool,
                    pool_models=state.joker_pool,
                    my_picks=my_picks,
                    opp_picks=opp_picks,
                    my_turn=True,
                    depth=0,
                    target_depth=target_depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    pick_num=pick_num,
                    is_p1=is_p1,
                    beam_k=beam_k,
                    cached_my_candidates=cached_my_candidates,
                )
                if idx is not None:
                    best_idx_found = idx
        except TimeoutError:
            pass

        return best_idx_found

    def pick_hand(self, state: GameState) -> List[int]:
        player_turn  = state.current_turn or PlayerTurn.PLAYER1
        hand         = _hand_for_player(state, player_turn)
        joker_classes = _joker_classes_for_player(state, player_turn)
        _, indices   = _best_hand_exact(hand, joker_classes)
        playable     = min(PLAYER_CARDS, len(hand))
        unique: List[int] = []
        for idx in indices:
            if 0 <= idx < playable and idx not in unique:
                unique.append(idx)
        for idx in range(playable):
            if len(unique) == 5:
                break
            if idx not in unique:
                unique.append(idx)
        return unique[:5]

    def _minimax(
        self,
        my_hand: List[Card],
        opp_candidates: List[List[Card]],
        pool: List[type],
        pool_models,
        my_picks: List[type],
        opp_picks: List[type],
        my_turn: bool,
        depth: int,
        target_depth: int,
        alpha: float,
        beta: float,
        pick_num: int,
        is_p1: bool,
        beam_k: int,
        cached_my_candidates: dict,
    ) -> tuple[float, int | None]:
        if time.perf_counter() - self.search_start_time > self.time_limit:
            raise TimeoutError()

        # Terminal: both hands full
        if len(my_picks) == JOKER_HAND_SIZE and len(opp_picks) == JOKER_HAND_SIZE:
            my_cands  = _get_diverse_candidates(my_hand, my_picks)
            my_score  = _best_hand_from_candidates(my_cands, my_picks)
            opp_score = _best_hand_from_candidates(opp_candidates, opp_picks)
            return float(my_score - opp_score), None

        if depth >= target_depth or not pool:
            my_cands  = _get_diverse_candidates(my_hand, my_picks)
            my_score  = _best_hand_from_candidates(my_cands, my_picks)
            opp_score = _best_hand_from_candidates(opp_candidates, opp_picks)
            return float(my_score - opp_score), None

        # pick_num advances by 1 per ply so denial weights stay on schedule.
        # Fix from v1: track whose ply this is so current_pick reflects the
        # actual draft position of the player picking, not a shared counter.
        current_pick = pick_num + depth

        node_candidates = []
        opp_base = _best_hand_from_candidates(opp_candidates, opp_picks)

        for index, joker_cls in enumerate(pool):
            joker_name = getattr(pool_models[index] if pool_models else joker_cls, "name", "")
            if my_turn:
                # Use cached candidates when available (depth 0 greedy pre-cached them)
                if depth == 0 and index in cached_my_candidates:
                    my_cands_j = cached_my_candidates[index]
                else:
                    my_cands_j = _get_diverse_candidates(my_hand, my_picks + [joker_cls])

                my_score_j  = _best_hand_from_candidates(my_cands_j, my_picks + [joker_cls])
                opp_score_j = _best_hand_from_candidates(opp_candidates, opp_picks + [joker_cls])
                opp_gain    = max(0, opp_score_j - opp_base)
                syn         = _synergy_bonus(joker_cls, my_hand)
                w           = _denial_weight(current_pick, is_p1, joker_name)
                val         = my_score_j + syn + w * opp_gain
            else:
                # Opponent's turn: they pick for themselves, we model them symmetrically
                my_cands_base  = _get_diverse_candidates(my_hand, my_picks)
                my_base        = _best_hand_from_candidates(my_cands_base, my_picks)
                opp_score_j    = _best_hand_from_candidates(opp_candidates, opp_picks + [joker_cls])
                my_score_j     = _best_hand_from_candidates(my_cands_base, my_picks + [joker_cls])
                my_gain        = max(0, my_score_j - my_base)
                # Opponent is modelled as seat-swapped (not is_p1)
                w              = _denial_weight(current_pick, not is_p1, joker_name)
                val            = opp_score_j + w * my_gain

            node_candidates.append((index, val, joker_cls))

        node_candidates.sort(key=lambda x: x[1], reverse=True)
        top = node_candidates[:beam_k]

        best_idx = None
        # Pool models need to be indexed correctly after pool slicing.
        # Pass None for pool_models at depth > 0 (joker_name falls back to cls.name).
        if my_turn:
            best_val = -math.inf
            for index, _, joker_cls in top:
                next_pool    = pool[:index] + pool[index + 1:]
                val, _       = self._minimax(
                    my_hand=my_hand,
                    opp_candidates=opp_candidates,
                    pool=next_pool,
                    pool_models=None,
                    my_picks=my_picks + [joker_cls],
                    opp_picks=opp_picks,
                    my_turn=False,
                    depth=depth + 1,
                    target_depth=target_depth,
                    alpha=alpha,
                    beta=beta,
                    pick_num=pick_num,
                    is_p1=is_p1,
                    beam_k=beam_k,
                    cached_my_candidates={},
                )
                if val > best_val:
                    best_val = val
                    best_idx = index
                alpha = max(alpha, best_val)
                if alpha >= beta:
                    break
            return best_val, best_idx
        else:
            best_val = math.inf
            for index, _, joker_cls in top:
                next_pool = pool[:index] + pool[index + 1:]
                val, _    = self._minimax(
                    my_hand=my_hand,
                    opp_candidates=opp_candidates,
                    pool=next_pool,
                    pool_models=None,
                    my_picks=my_picks,
                    opp_picks=opp_picks + [joker_cls],
                    my_turn=True,
                    depth=depth + 1,
                    target_depth=target_depth,
                    alpha=alpha,
                    beta=beta,
                    pick_num=pick_num,
                    is_p1=is_p1,
                    beam_k=beam_k,
                    cached_my_candidates={},
                )
                if val < best_val:
                    best_val = val
                    best_idx = index
                beta = min(beta, best_val)
                if alpha >= beta:
                    break
            return best_val, best_idx


ParticipantBot = Bot