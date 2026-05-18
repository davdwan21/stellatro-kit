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

# ---------------------------------------------------------------------------
# Hard time cap — must return before 200 ms or the move is rejected.
# ---------------------------------------------------------------------------
_TIME_LIMIT = 0.160

# ---------------------------------------------------------------------------
# Hand-type joker sets
# ---------------------------------------------------------------------------

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

_FLUSH_JOKER_NAMES = frozenset({
    "The Tribe",
    "Flower Pot",
    "Smeared Joker",
    "Suit Joker",
    "Daring Joker",
    "Vibrant Joker",
    "Sun God",
    "Arrowhead",
    "Spade Joker",
    "Heart Joker",
    "Diamond Joker",
    "Club Joker",
})

_STRAIGHT_JOKER_NAMES = frozenset({
    "The Order",
    "Straight Shooter",
    "Witty Joker",
    "Lively Joker",
})

_PAIR_JOKER_NAMES = frozenset({
    "The Duo",
    "The Trio",
    "The Tribe",
    "The Family",
    "Jolly Joker",
    "Sly Joker",
    "Zany Joker",
    "Merry Joker",
    "Cheeky Joker",
    "Jovial Joker",
    "Star Fish",
    "Thrice Twice",
})

_FACE_JOKER_NAMES = frozenset({
    "Sock and Buskin",
    "PhotoGraph Joker",
    "Scary Face Joker",
    "Mirror",
    "Spotlight",
    "Starjack",
})

_CHIP_JOKER_NAMES = frozenset({
    "Spare Trousers",
    "Bootstrapper",
    "Egg",
    "Throwback",
    "Runner",
    "Supernova",
    "Dusk",
})

# ---------------------------------------------------------------------------
# Joker floor values.
# Standalone baseline value for known strong jokers. Prevents bot from
# drafting a worthless denial pick when evaluate_hand scores are similar.
# Benchmarking showed 40% of fixed_synergy's values is the right level —
# strong enough to steer marginal picks, not so strong it overrides evaluate_hand.
# ---------------------------------------------------------------------------
_JOKER_FLOOR: dict[str, float] = {
    "Pips": 360,
    "Stargazing": 340,
    "Starcorn": 320,
    "Galaxy": 304,
    "Sock and Buskin": 288,
    "Jam Session": 272,
    "Supernova": 232,
    "Snowball": 208,
    "Lock In": 168,
    "Six Seven": 140,
    "Blackjack": 128,
}

# ---------------------------------------------------------------------------
# Pair-synergy table.
# Bonus for completing a known two-joker engine. Values scaled up 50% from
# the original conservative priors after benchmarking showed underweighting
# was losing marginal 2k–5k games to fixed_synergy's stronger priors.
# ---------------------------------------------------------------------------
_PAIR_SYNERGY: dict[tuple[str, str], float] = {
    ("Pips", "Stargazing"): 2850,
    ("Pips", "Starcorn"): 2400,
    ("Pips", "Galaxy"): 2100,
    ("Pips", "Snowball"): 1875,
    ("Galaxy", "Stargazing"): 1800,
    ("Supernova", "Stargazing"): 1875,
    ("Sock and Buskin", "Jam Session"): 2025,
    ("Seltzer", "Jam Session"): 1425,
    ("Encore", "Jam Session"): 1125,
    ("Lock In", "Pips"): 975,
    ("Starjack", "Starcorn"): 1050,
    ("Fallen Star", "Starcorn"): 1200,
    ("Star Fish", "Galaxy"): 1050,
}


def _joker_floor(joker_cls: type) -> float:
    return _JOKER_FLOOR.get(getattr(joker_cls, "name", ""), 0.0) * 1.000000, 0.0)


def _pair_synergy_bonus(existing: List[type], new_cls: type) -> float:
    existing_names = {getattr(j, "name", "") for j in existing}
    new_name = getattr(new_cls, "name", "")
    total = 0.0
    for (a, b), val in _PAIR_SYNERGY.items():
        if (new_name == a and b in existing_names) or (new_name == b and a in existing_names):
            total += val * 1.000000
    return total


# ---------------------------------------------------------------------------
# Denial weights
# ---------------------------------------------------------------------------

def _denial_weight(pick_num: int, is_p1: bool, joker_name: str) -> float:
    is_xmult = joker_name in _XMULT_JOKER_NAMES
    is_chip  = joker_name in _CHIP_JOKER_NAMES

    if is_p1:
        if is_xmult:
            weights = [0.3000, 0.4000, 0.3800, 0.3200, 0.2500]
        else:
            weights = [0.1000, 0.2500, 0.3500, 0.3500, 0.3000]
    else:
        if is_xmult:
            weights = [0.8500, 0.9000, 0.5500, 0.3800, 0.2800]
        elif is_chip:
            weights = [0.6000, 0.4500, 0.3500, 0.3000, 0.2500]
        else:
            weights = [0.7000, 0.4500, 0.3500, 0.3500, 0.3000]

    return weights[min(pick_num, len(weights) - 1)]


# ---------------------------------------------------------------------------
# Card / joker model helpers
# ---------------------------------------------------------------------------

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


def _pick_number(state: GameState, player_turn: PlayerTurn) -> int:
    if player_turn == PlayerTurn.PLAYER1:
        return len(state.player1_jokers)
    return len(state.player2_jokers)


def _is_player1(state: GameState, player_turn: PlayerTurn) -> bool:
    return player_turn == PlayerTurn.PLAYER1


# ---------------------------------------------------------------------------
# Hand-shape helpers
# ---------------------------------------------------------------------------

def _flush_potential(cards: List[Card]) -> float:
    if not cards:
        return 0.0
    counts: dict = {}
    for c in cards:
        for s in c.suits:
            counts[s] = counts.get(s, 0) + 1
    return max(counts.values()) / len(cards)


def _straight_potential(cards: List[Card]) -> float:
    if len(cards) < 2:
        return 0.0
    ranks = sorted(set(c.rank for c in cards))
    best = 0
    for i in range(len(ranks)):
        window = [r for r in ranks if ranks[i] <= r <= ranks[i] + 4]
        best = max(best, len(window))
    if 14 in ranks:
        low_ranks = sorted(set((1 if r == 14 else r) for r in ranks if r <= 5 or r == 14))
        best = max(best, len([r for r in low_ranks if 1 <= r <= 5]))
    return best / 5.0


def _pair_potential(cards: List[Card]) -> int:
    counts: dict = {}
    for c in cards:
        counts[c.rank] = counts.get(c.rank, 0) + 1
    return sum(1 for v in counts.values() if v >= 2)


def _face_count(cards: List[Card]) -> int:
    return sum(1 for c in cards if c.rank in {11, 12, 13})


def _synergy_bonus(joker_cls: type, cards: List[Card]) -> float:
    name = getattr(joker_cls, "name", "")
    bonus = 0.0

    if name in _FLUSH_JOKER_NAMES:
        fp = _flush_potential(cards)
        bonus += (800.0 * fp if fp >= 0.60 else 300.0 * fp if fp >= 0.40 else 0.0)

    if name in _STRAIGHT_JOKER_NAMES:
        sp = _straight_potential(cards)
        bonus += (800.0 * sp if sp >= 0.80 else 300.0 * sp if sp >= 0.60 else 0.0)

    if name in _PAIR_JOKER_NAMES:
        pp = _pair_potential(cards)
        bonus += (600.0 if pp >= 2 else 200.0 if pp >= 1 else 0.0)

    if name in _FACE_JOKER_NAMES:
        fc = _face_count(cards)
        bonus += (550.0 if fc >= 3 else 220.0 if fc >= 2 else 0.0)

    return bonus


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _get_diverse_candidates(
    cards: List[Card],
    joker_classes: List[type] = None,
    max_candidates: int = 20,
) -> List[List[Card]]:
    joker_classes = joker_classes or []
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return []

    names = {getattr(j, "name", "") for j in joker_classes}
    flush_jokers    = len(names & _FLUSH_JOKER_NAMES)
    straight_jokers = len(names & _STRAIGHT_JOKER_NAMES)
    pair_jokers     = len(names & _PAIR_JOKER_NAMES)
    face_jokers     = len(names & _FACE_JOKER_NAMES)

    base_slots     = 6
    flush_slots    = 2 + flush_jokers * 2
    straight_slots = 2 + straight_jokers * 2
    pair_slots     = 1 + pair_jokers
    face_slots     = 1 + face_jokers

    scored_combos = []
    for combo in combinations(range(n), 5):
        subset = [cards[i] for i in combo]
        try:
            score = evaluate_hand([deepcopy(c) for c in subset], [])
        except Exception:
            score = 0
        scored_combos.append((score, subset))

    scored_combos.sort(key=lambda x: x[0], reverse=True)

    candidates: List[List[Card]] = []
    seen: set = set()

    def combo_key(subset):
        return tuple(sorted(
            (c.rank, tuple(sorted(s.value for s in c.suits))) for c in subset
        ))

    def add(subset) -> bool:
        k = combo_key(subset)
        if k in seen:
            return False
        seen.add(k)
        candidates.append(subset)
        return True

    def is_flush(subset):
        common = set(subset[0].suits)
        for c in subset[1:]:
            common &= set(c.suits)
            if not common:
                return False
        return True

    def is_straight(subset):
        ranks = sorted(c.rank for c in subset)
        return (len(set(ranks)) == 5 and ranks[-1] - ranks[0] == 4) or ranks == [2, 3, 4, 5, 14]

    def is_pair_heavy(subset):
        rc: dict = {}
        for c in subset:
            rc[c.rank] = rc.get(c.rank, 0) + 1
        return sum(1 for v in rc.values() if v >= 2) >= 2

    def is_face_heavy(subset):
        return sum(1 for c in subset if c.rank in {11, 12, 13}) >= 2

    for _, subset in scored_combos[:base_slots]:
        add(subset)

    added = 0
    for _, subset in scored_combos:
        if added >= flush_slots:
            break
        if is_flush(subset) and add(subset):
            added += 1

    added = 0
    for _, subset in scored_combos:
        if added >= straight_slots:
            break
        if is_straight(subset) and add(subset):
            added += 1

    added = 0
    for _, subset in scored_combos:
        if added >= pair_slots:
            break
        if is_pair_heavy(subset) and add(subset):
            added += 1

    added = 0
    for _, subset in scored_combos:
        if added >= face_slots:
            break
        if is_face_heavy(subset) and add(subset):
            added += 1

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
        try:
            score = evaluate_hand([deepcopy(c) for c in subset], [cls() for cls in joker_classes])
            if score > best:
                best = score
        except Exception:
            continue
    return best


def _best_hand_exact(
    cards: List[Card],
    joker_classes: List[type],
) -> tuple[int, List[int]]:
    best_score = -1
    best_indices = list(range(5))
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return 0, []

    for combo in combinations(range(n), 5):
        try:
            score = evaluate_hand(
                [deepcopy(cards[i]) for i in combo],
                [cls() for cls in joker_classes],
            )
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
    def __init__(self, time_limit: float = _TIME_LIMIT) -> None:
        self.time_limit = time_limit
        self.search_start_time = 0.0
        self._cand_cache: dict = {}

    def _elapsed(self) -> float:
        return time.perf_counter() - self.search_start_time

    def _timeout(self) -> None:
        if self._elapsed() > self.time_limit:
            raise TimeoutError()

    def _cached_candidates(
        self, cards: List[Card], joker_classes: List[type]
    ) -> List[List[Card]]:
        key = (
            tuple((c.rank, tuple(sorted(s.value for s in c.suits))) for c in cards),
            tuple(sorted(getattr(j, "name", "") for j in joker_classes)),
        )
        if key not in self._cand_cache:
            self._cand_cache[key] = _get_diverse_candidates(cards, joker_classes)
        return self._cand_cache[key]

    def pick_joker(self, state: GameState) -> int:
        self.search_start_time = time.perf_counter()
        self._cand_cache = {}

        player_turn = state.current_turn
        if player_turn not in (PlayerTurn.PLAYER1, PlayerTurn.PLAYER2):
            return 0

        opp_turn = (
            PlayerTurn.PLAYER2 if player_turn == PlayerTurn.PLAYER1 else PlayerTurn.PLAYER1
        )

        my_hand  = _hand_for_player(state, player_turn)
        opp_hand = _hand_for_player(state, opp_turn)

        my_picks  = _joker_classes_for_player(state, player_turn)
        opp_picks = _joker_classes_for_player(state, opp_turn)
        pool      = [_joker_cls_from_model(joker) for joker in state.joker_pool]

        if len(pool) <= 1:
            return 0

        pick_num = _pick_number(state, player_turn)
        is_p1    = _is_player1(state, player_turn)

        opp_candidates = self._cached_candidates(opp_hand, opp_picks)
        opp_base       = _best_hand_from_candidates(opp_candidates, opp_picks)

        # ------------------------------------------------------------------
        # Phase 1: greedy pass — always completes for a safe fallback.
        # val = evaluate_hand score
        #     + hand-shape synergy bonus
        #     + engine-completion bonus (our pairs)
        #     + standalone floor value
        #     + denial weight × opponent gain
        #     + opponent engine-completion denial
        # ------------------------------------------------------------------
        greedy_choices = []
        for index, joker_model in enumerate(state.joker_pool):
            joker_cls  = pool[index]
            joker_name = joker_model.name

            my_picks_j      = my_picks + [joker_cls]
            my_candidates_j = self._cached_candidates(my_hand, my_picks_j)
            my_score        = _best_hand_from_candidates(my_candidates_j, my_picks_j)

            opp_score_j = _best_hand_from_candidates(opp_candidates, opp_picks + [joker_cls])
            opp_gain    = max(0, opp_score_j - opp_base)

            val = (
                my_score
                + _synergy_bonus(joker_cls, my_hand)
                + _pair_synergy_bonus(my_picks, joker_cls)
                + _joker_floor(joker_cls)
                + _denial_weight(pick_num, is_p1, joker_name) * opp_gain
                + 0.35 * _pair_synergy_bonus(opp_picks, joker_cls)
            )
            greedy_choices.append((index, val, joker_cls, my_candidates_j))

        greedy_choices.sort(key=lambda x: x[1], reverse=True)
        best_index = greedy_choices[0][0]

        if self._elapsed() > self.time_limit:
            return best_index

        # ------------------------------------------------------------------
        # Phase 2: iterative-deepening minimax with alpha-beta pruning.
        # ------------------------------------------------------------------
        cached_my_candidates = {entry[0]: entry[3] for entry in greedy_choices}

        best_idx_found = best_index
        try:
            for target_depth, beam_k in ((1, 5), (2, 3)):
                self._timeout()
                val, idx = self._minimax(
                    my_hand=my_hand,
                    opp_hand=opp_hand,
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
        player_turn   = state.current_turn or PlayerTurn.PLAYER1
        hand          = _hand_for_player(state, player_turn)
        joker_classes = _joker_classes_for_player(state, player_turn)
        _, indices    = _best_hand_exact(hand, joker_classes)

        playable = min(PLAYER_CARDS, len(hand))
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
        opp_hand: List[Card],
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
        self._timeout()

        if not pool or (
            len(my_picks) >= JOKER_HAND_SIZE and len(opp_picks) >= JOKER_HAND_SIZE
        ):
            my_cands  = self._cached_candidates(my_hand, my_picks)
            my_score  = _best_hand_from_candidates(my_cands, my_picks)
            opp_score = _best_hand_from_candidates(opp_candidates, opp_picks)
            return float(my_score - opp_score), None

        if depth >= target_depth:
            my_cands  = self._cached_candidates(my_hand, my_picks)
            my_score  = _best_hand_from_candidates(my_cands, my_picks)
            opp_score = _best_hand_from_candidates(opp_candidates, opp_picks)
            return float(my_score - opp_score), None

        current_pick = min(4, pick_num + depth)
        node_candidates = []

        if my_turn:
            opp_base = _best_hand_from_candidates(opp_candidates, opp_picks)

            for index, joker_cls in enumerate(pool):
                joker_name = getattr(
                    pool_models[index] if pool_models else joker_cls, "name", ""
                )

                if depth == 0 and index in cached_my_candidates:
                    my_cands_j = cached_my_candidates[index]
                else:
                    my_cands_j = self._cached_candidates(my_hand, my_picks + [joker_cls])

                my_score_j  = _best_hand_from_candidates(my_cands_j, my_picks + [joker_cls])
                opp_score_j = _best_hand_from_candidates(opp_candidates, opp_picks + [joker_cls])
                opp_gain    = max(0, opp_score_j - opp_base)

                val = (
                    my_score_j
                    + _synergy_bonus(joker_cls, my_hand)
                    + _pair_synergy_bonus(my_picks, joker_cls)
                    + _joker_floor(joker_cls)
                    + _denial_weight(current_pick, is_p1, joker_name) * opp_gain
                    + 0.35 * _pair_synergy_bonus(opp_picks, joker_cls)
                )
                node_candidates.append((index, val, joker_cls))

        else:
            my_cands_base = self._cached_candidates(my_hand, my_picks)

            for index, joker_cls in enumerate(pool):
                joker_name = getattr(
                    pool_models[index] if pool_models else joker_cls, "name", ""
                )

                opp_cands_j = self._cached_candidates(opp_hand, opp_picks + [joker_cls])
                opp_score_j = _best_hand_from_candidates(opp_cands_j, opp_picks + [joker_cls])

                val = (
                    opp_score_j
                    + _pair_synergy_bonus(opp_picks, joker_cls)
                    + 0.65 * _pair_synergy_bonus(my_picks, joker_cls)
                )
                node_candidates.append((index, val, joker_cls))

        node_candidates.sort(key=lambda x: x[1], reverse=True)
        top = node_candidates[:beam_k]

        if my_turn:
            best_val = -math.inf
            best_idx = None
            for index, _, joker_cls in top:
                next_pool = pool[:index] + pool[index + 1:]
                val, _ = self._minimax(
                    my_hand=my_hand,
                    opp_hand=opp_hand,
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
            best_idx = None
            for index, _, joker_cls in top:
                next_pool = pool[:index] + pool[index + 1:]
                opp_cands_next = self._cached_candidates(opp_hand, opp_picks + [joker_cls])
                val, _ = self._minimax(
                    my_hand=my_hand,
                    opp_hand=opp_hand,
                    opp_candidates=opp_cands_next,
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