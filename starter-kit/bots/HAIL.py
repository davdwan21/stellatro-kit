import math
import time
from copy import deepcopy
from itertools import combinations
from typing import List, Tuple

from stellatro_common import CardModel, GameState, JokerModel, PlayerTurn
from stellatro_game import Card, JOKER_HAND_SIZE, PLAYER_CARDS, Suit, evaluate_hand
from stellatro_game.jokers import ALL_JOKER_CLASSES, RegularJoker

Hand = List[Card]

_JOKER_NAME_TO_CLASS = {joker_cls.name: joker_cls for joker_cls in ALL_JOKER_CLASSES}

# ---------------------------------------------------------------------------
# Tuned meta constants
# ---------------------------------------------------------------------------

JOKER_BASE_VALUE = {
    "Pips": 900,
    "Stargazing": 850,
    "Starcorn": 800,
    "Galaxy": 760,
    "Sock and Buskin": 720,
    "Jam Session": 680,
    "Supernova": 580,
    "Snowball": 520,
    "Lock In": 420,
    "Six Seven": 350,
    "Blackjack": 320,
}

PAIR_SYNERGY = {
    ("Pips", "Stargazing"): 1900,
    ("Pips", "Starcorn"): 1600,
    ("Pips", "Galaxy"): 1400,
    ("Pips", "Snowball"): 1250,
    ("Galaxy", "Stargazing"): 1200,
    ("Supernova", "Stargazing"): 1250,
    ("Sock and Buskin", "Jam Session"): 1350,
    ("Seltzer", "Jam Session"): 950,
    ("Encore", "Jam Session"): 750,
    ("Lock In", "Pips"): 650,
    ("Starjack", "Starcorn"): 700,
    ("Fallen Star", "Starcorn"): 800,
    ("Star Fish", "Galaxy"): 700,
}

S_TIER_DENIAL = frozenset({
    "Pips",
    "Stargazing",
    "Starcorn",
    "Galaxy",
    "Sock and Buskin",
    "Jam Session",
})

A_TIER_DENIAL = frozenset({
    "Snowball",
    "Supernova",
    "Lock In",
    "Six Seven",
    "Blackjack",
})

_XMULT_JOKER_NAMES = frozenset({
    "The Duo",
    "The Trio",
    "The Tribe",
    "The Order",
    "UC Socially Dead",
    "Flower Pot",
})

_FLUSH_JOKER_NAMES = frozenset({
    "The Tribe",
    "Flower Pot",
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
    "Witty Joker",
    "Lively Joker",
})

_PAIR_JOKER_NAMES = frozenset({
    "The Duo",
    "The Trio",
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


def _pick_number(state: GameState, player_turn: PlayerTurn) -> int:
    if player_turn == PlayerTurn.PLAYER1:
        return len(state.player1_jokers)
    return len(state.player2_jokers)


def _is_player1(player_turn: PlayerTurn) -> bool:
    return player_turn == PlayerTurn.PLAYER1


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


def _names(joker_classes: List[type]) -> set:
    return {getattr(j, "name", "") for j in joker_classes}


def _pair_synergy_bonus(existing: List[type], new_cls: type) -> float:
    names = _names(existing)
    new_name = getattr(new_cls, "name", "")
    total = 0.0
    for (a, b), val in PAIR_SYNERGY.items():
        if new_name == a and b in names:
            total += val
        elif new_name == b and a in names:
            total += val
    return total


def _combo_completion_bonus(existing: List[type], new_cls: type) -> float:
    return (
        JOKER_BASE_VALUE.get(getattr(new_cls, "name", ""), 0) +
        _pair_synergy_bonus(existing, new_cls)
    )


def _denial_weight(pick_num: int, is_p1: bool, joker_name: str) -> float:
    if joker_name in S_TIER_DENIAL:
        p1_weights = [0.35, 0.45, 0.50, 0.42, 0.30]
        p2_weights = [1.70, 1.35, 0.95, 0.55, 0.35]
    elif joker_name in A_TIER_DENIAL:
        p1_weights = [0.20, 0.30, 0.36, 0.32, 0.24]
        p2_weights = [0.95, 0.70, 0.50, 0.36, 0.24]
    elif joker_name in _XMULT_JOKER_NAMES:
        p1_weights = [0.10, 0.22, 0.28, 0.26, 0.20]
        p2_weights = [0.88, 0.62, 0.44, 0.30, 0.20]
    else:
        p1_weights = [0.05, 0.12, 0.18, 0.18, 0.14]
        p2_weights = [0.52, 0.38, 0.28, 0.22, 0.16]

    weights = p1_weights if is_p1 else p2_weights
    return weights[min(pick_num, len(weights) - 1)]


def _archetype_vector(jokers: List[type]) -> dict:
    names = _names(jokers)
    return {
        "flush": len(names & _FLUSH_JOKER_NAMES),
        "straight": len(names & _STRAIGHT_JOKER_NAMES),
        "pairs": len(names & _PAIR_JOKER_NAMES),
        "face": len(names & _FACE_JOKER_NAMES),
    }


def _engine_pressure_score(existing: List[type], candidate: type) -> float:
    before = _archetype_vector(existing)
    after = _archetype_vector(existing + [candidate])
    score = 0.0
    for key in before:
        b = before[key]
        a = after[key]
        if b >= 1 and a >= 2:
            score += 420.0
        elif b == 0 and a == 1:
            score += 90.0
    return score


def _future_scaling_bonus(joker_classes: List[type]) -> float:
    names = _names(joker_classes)
    bonus = 0.0
    bonus += 120.0 * len(names & _XMULT_JOKER_NAMES)
    if "Pips" in names and "Stargazing" in names:
        bonus += 600.0
    if "Pips" in names and "Galaxy" in names:
        bonus += 420.0
    if "Pips" in names and "Starcorn" in names:
        bonus += 420.0
    if "Starcorn" in names and "Galaxy" in names:
        bonus += 320.0
    if "Sock and Buskin" in names and "Jam Session" in names:
        bonus += 450.0
    return bonus


# ---------------------------------------------------------------------------
# Hand shape helpers
# ---------------------------------------------------------------------------

def _flush_potential(cards: List[Card]) -> float:
    if not cards:
        return 0.0
    counts = {}
    for c in cards:
        for s in c.suits:
            counts[s] = counts.get(s, 0) + 1
    return max(counts.values()) / len(cards)


def _straight_potential(cards: List[Card]) -> float:
    ranks = sorted(set(c.rank for c in cards))
    if not ranks:
        return 0.0
    best = 0
    for start in range(2, 11):
        best = max(best, sum(1 for r in ranks if start <= r <= start + 4))
    if 14 in ranks:
        best = max(best, len({r for r in ranks if r in {2, 3, 4, 5}}) + 1)
    return best / 5.0


def _pair_potential(cards: List[Card]) -> int:
    counts = {}
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
        if fp >= 0.6:
            bonus += 650.0 * fp
        elif fp >= 0.4:
            bonus += 220.0 * fp
    if name in _STRAIGHT_JOKER_NAMES:
        sp = _straight_potential(cards)
        if sp >= 0.8:
            bonus += 650.0 * sp
        elif sp >= 0.6:
            bonus += 230.0 * sp
    if name in _PAIR_JOKER_NAMES:
        pp = _pair_potential(cards)
        if pp >= 2:
            bonus += 520.0
        elif pp >= 1:
            bonus += 180.0
    if name in _FACE_JOKER_NAMES:
        fc = _face_count(cards)
        if fc >= 3:
            bonus += 550.0
        elif fc >= 2:
            bonus += 220.0
    return bonus


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _get_diverse_candidates(
    cards: List[Card],
    joker_classes: List[type] = None,
    max_candidates: int = 32,
) -> List[List[Card]]:
    joker_classes = joker_classes or []
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return []

    names = _names(joker_classes)
    flush_jokers = len(names & _FLUSH_JOKER_NAMES)
    straight_jokers = len(names & _STRAIGHT_JOKER_NAMES)
    pair_jokers = len(names & _PAIR_JOKER_NAMES)
    face_jokers = len(names & _FACE_JOKER_NAMES)

    base_slots = 7
    flush_slots = 2 + 2 * flush_jokers
    straight_slots = 2 + 2 * straight_jokers
    pair_slots = 2 + pair_jokers
    face_slots = 2 + 2 * face_jokers
    high_rank_slots = 3
    stella_slots = 3

    scored = []
    for combo in combinations(range(n), 5):
        subset = [cards[i] for i in combo]
        try:
            base = evaluate_hand([deepcopy(c) for c in subset], [])
        except Exception:
            base = 0

        ranks = [c.rank for c in subset]
        suits = [next(iter(c.suits)) for c in subset]
        rank_sum = sum(ranks)
        duplicate_count = len(ranks) - len(set(ranks))
        face_count = sum(1 for r in ranks if r in {11, 12, 13})
        unique_suits = len(set(suits))

        shape = (
            7 * rank_sum
            + 70 * duplicate_count
            + (220 if unique_suits == 1 else 0)
            + 65 * face_count
            + (45 * rank_sum if names & {"Pips", "Galaxy", "Stargazing", "Starcorn"} else 0)
            + (90 * face_count if names & _FACE_JOKER_NAMES else 0)
            + (95 * duplicate_count if names & _PAIR_JOKER_NAMES else 0)
        )
        scored.append((base + shape, subset))

    scored.sort(key=lambda x: x[0], reverse=True)

    candidates = []
    seen = set()

    def key(subset):
        return tuple(sorted((c.rank, tuple(sorted(s.value for s in c.suits))) for c in subset))

    def add(subset):
        k = key(subset)
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
        counts = {}
        for c in subset:
            counts[c.rank] = counts.get(c.rank, 0) + 1
        return any(v >= 2 for v in counts.values())

    def is_face_heavy(subset):
        return sum(1 for c in subset if c.rank in {11, 12, 13}) >= 2

    def is_face_scaling(subset):
        return sum(1 for c in subset if c.rank in {11, 12, 13}) >= 3

    def high_rank_score(subset):
        return sum(c.rank for c in subset)

    def duplicate_score(subset):
        counts = {}
        for c in subset:
            counts[c.rank] = counts.get(c.rank, 0) + 1
        return sum(v * v for v in counts.values())

    for _, subset in scored[:base_slots]:
        add(subset)

    added = 0
    for _, subset in scored:
        if added >= flush_slots:
            break
        if is_flush(subset) and add(subset):
            added += 1

    added = 0
    for _, subset in scored:
        if added >= straight_slots:
            break
        if is_straight(subset) and add(subset):
            added += 1

    added = 0
    for _, subset in scored:
        if added >= pair_slots:
            break
        if is_pair_heavy(subset) and add(subset):
            added += 1

    added = 0
    for _, subset in scored:
        if added >= face_slots:
            break
        if is_face_heavy(subset) and add(subset):
            added += 1

    added = 0
    face_sorted = sorted(scored, key=lambda x: sum(1 for c in x[1] if c.rank in {11, 12, 13}), reverse=True)
    for _, subset in face_sorted:
        if added >= face_slots:
            break
        if is_face_scaling(subset) and add(subset):
            added += 1

    added = 0
    rank_sorted = sorted(scored, key=lambda x: high_rank_score(x[1]), reverse=True)
    for _, subset in rank_sorted:
        if added >= high_rank_slots:
            break
        if add(subset):
            added += 1

    added = 0
    dup_sorted = sorted(scored, key=lambda x: duplicate_score(x[1]), reverse=True)
    for _, subset in dup_sorted:
        if added >= stella_slots:
            break
        if duplicate_score(subset) >= 7 and add(subset):
            added += 1

    for _, subset in scored:
        if len(candidates) >= max_candidates:
            break
        add(subset)

    return candidates


def _best_hand_from_candidates(candidates: List[List[Card]], joker_classes: List[type]) -> int:
    best = 0
    for subset in candidates:
        try:
            score = evaluate_hand([deepcopy(c) for c in subset], [cls() for cls in joker_classes])
            if score > best:
                best = score
        except Exception:
            continue
    return best


def _best_hand_exact(cards: List[Card], joker_classes: List[type]) -> Tuple[int, List[int]]:
    """Brute-force all C(n,5) combos. Returns (best_score, best_indices)."""
    best_score = -1
    best_indices = list(range(5))
    n = min(PLAYER_CARDS, len(cards))
    if n < 5:
        return 0, list(range(n))

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


def _score_combo_exact(cards: List[Card], joker_classes: List[type], combo: Tuple[int, ...]) -> int:
    try:
        return evaluate_hand(
            [deepcopy(cards[i]) for i in combo],
            [cls() for cls in joker_classes],
        )
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Opponent-aware hand selection
# ---------------------------------------------------------------------------

def _pick_hand_vs_opponent(
    my_hand: List[Card],
    my_joker_classes: List[type],
    opp_hand: List[Card],
    opp_joker_classes: List[type],
) -> List[int]:
    """
    Win/draw/loss scoring means margin doesn't matter — only outcome.
    Strategy:
      1. Compute opponent's best possible score (they also play optimally).
      2. If our best hand already beats them, play it (maximize winning margin
         to guard against any scoring edge cases, and stay robust).
      3. If we're behind, find the minimum hand that still beats them —
         the "cheapest" winning hand, which is most likely to actually score
         above their result regardless of variance.
      4. If we can't beat them, maximize our score to chase a draw.
    """
    n = min(PLAYER_CARDS, len(my_hand))
    if n < 5:
        return list(range(n))

    # Step 1: opponent's ceiling
    opp_best, _ = _best_hand_exact(opp_hand, opp_joker_classes)

    # Step 2: our ceiling
    my_best_score, my_best_indices = _best_hand_exact(my_hand, my_joker_classes)

    if my_best_score > opp_best:
        # We already win with our best hand — play it
        return my_best_indices

    # Step 3: find all hands that beat the opponent
    beating: List[Tuple[int, List[int]]] = []
    for combo in combinations(range(n), 5):
        score = _score_combo_exact(my_hand, my_joker_classes, combo)
        if score > opp_best:
            beating.append((score, list(combo)))

    if beating:
        # Take the minimum winning hand — least variance, still wins
        _, indices = min(beating, key=lambda x: x[0])
        return indices

    # Step 4: can't win — maximize score to try to draw or minimize loss
    return my_best_indices


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self, time_limit: float = 0.185) -> None:
        self.time_limit = time_limit
        self.search_start_time = 0.0
        self._cand_cache = {}

    def _timeout(self):
        if time.perf_counter() - self.search_start_time > self.time_limit:
            raise TimeoutError()

    def _cached_candidates(self, cards: List[Card], joker_classes: List[type]) -> List[List[Card]]:
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

        opp_turn = PlayerTurn.PLAYER2 if player_turn == PlayerTurn.PLAYER1 else PlayerTurn.PLAYER1

        my_hand = _hand_for_player(state, player_turn)
        opp_hand = _hand_for_player(state, opp_turn)

        my_picks = _joker_classes_for_player(state, player_turn)
        opp_picks = _joker_classes_for_player(state, opp_turn)
        pool = [_joker_cls_from_model(joker) for joker in state.joker_pool]

        if len(pool) <= 1:
            return 0

        pick_num = _pick_number(state, player_turn)
        is_p1 = _is_player1(player_turn)

        opp_candidates = self._cached_candidates(opp_hand, opp_picks)
        opp_base = _best_hand_from_candidates(opp_candidates, opp_picks)

        greedy_choices = []
        for index, joker_cls in enumerate(pool):
            joker_name = getattr(joker_cls, "name", "")

            my_picks_j = my_picks + [joker_cls]
            opp_picks_j = opp_picks + [joker_cls]

            my_candidates_j = self._cached_candidates(my_hand, my_picks_j)
            my_score = _best_hand_from_candidates(my_candidates_j, my_picks_j)

            opp_score = _best_hand_from_candidates(opp_candidates, opp_picks_j)
            opp_gain = max(0, opp_score - opp_base)

            syn = _synergy_bonus(joker_cls, my_hand)
            combo = _combo_completion_bonus(my_picks, joker_cls)
            deny = _denial_weight(pick_num, is_p1, joker_name) * opp_gain
            opp_combo_deny = 0.55 * _pair_synergy_bonus(opp_picks, joker_cls)

            if not is_p1:
                opp_engine_gain = _engine_pressure_score(opp_picks, joker_cls)
                deny += 0.25 * opp_engine_gain
                my_future = _future_scaling_bonus(my_picks_j)
                opp_future_if_given = _future_scaling_bonus(opp_picks_j)
                val = (
                    my_score + syn + combo
                    + 0.35 * my_future
                    + deny + opp_combo_deny
                    + 0.12 * opp_future_if_given
                )
            else:
                val = (
                    my_score + syn + combo
                    + 0.22 * _future_scaling_bonus(my_picks_j)
                    + deny + opp_combo_deny
                )

            greedy_choices.append((index, val, joker_cls, my_candidates_j))

        greedy_choices.sort(key=lambda x: x[1], reverse=True)
        best_index = greedy_choices[0][0]

        try:
            for target_depth, beam_k in ((1, 9), (2, 4)):
                self._timeout()
                val, idx = self._minimax(
                    my_hand=my_hand,
                    opp_hand=opp_hand,
                    opp_candidates=opp_candidates,
                    pool=pool,
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
                )
                if idx is not None:
                    best_index = idx
        except TimeoutError:
            pass

        return best_index

    def pick_hand(self, state: GameState) -> List[int]:
        player_turn = state.current_turn or PlayerTurn.PLAYER1

        my_hand = _hand_for_player(state, player_turn)
        my_joker_classes = _joker_classes_for_player(state, player_turn)

        opp_turn = PlayerTurn.PLAYER2 if player_turn == PlayerTurn.PLAYER1 else PlayerTurn.PLAYER1
        opp_hand = _hand_for_player(state, opp_turn)
        opp_joker_classes = _joker_classes_for_player(state, opp_turn)

        indices = _pick_hand_vs_opponent(my_hand, my_joker_classes, opp_hand, opp_joker_classes)

        # Validate and fill if needed
        playable = min(PLAYER_CARDS, len(my_hand))
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

    def _position_value(self, my_hand, opp_candidates, my_picks, opp_picks):
        my_candidates = self._cached_candidates(my_hand, my_picks)
        my_score = _best_hand_from_candidates(my_candidates, my_picks)
        opp_score = _best_hand_from_candidates(opp_candidates, opp_picks)
        return float(my_score - opp_score)

    def _minimax(
        self,
        my_hand: List[Card],
        opp_hand: List[Card],
        opp_candidates: List[List[Card]],
        pool: List[type],
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
    ) -> Tuple[float, int]:
        self._timeout()

        if not pool or (len(my_picks) >= JOKER_HAND_SIZE and len(opp_picks) >= JOKER_HAND_SIZE):
            return self._position_value(my_hand, opp_candidates, my_picks, opp_picks), None

        if depth >= target_depth:
            return self._position_value(my_hand, opp_candidates, my_picks, opp_picks), None

        current_pick = min(4, pick_num + depth)
        node_candidates = []

        my_base_candidates = self._cached_candidates(my_hand, my_picks)
        my_base = _best_hand_from_candidates(my_base_candidates, my_picks)
        opp_base = _best_hand_from_candidates(opp_candidates, opp_picks)

        for index, joker_cls in enumerate(pool):
            joker_name = getattr(joker_cls, "name", "")

            if my_turn:
                my_picks_j = my_picks + [joker_cls]
                my_cands_j = self._cached_candidates(my_hand, my_picks_j)
                my_score_j = _best_hand_from_candidates(my_cands_j, my_picks_j)

                opp_score_j = _best_hand_from_candidates(opp_candidates, opp_picks + [joker_cls])
                opp_gain = max(0, opp_score_j - opp_base)

                if not is_p1:
                    val = (
                        my_score_j
                        + _synergy_bonus(joker_cls, my_hand)
                        + _combo_completion_bonus(my_picks, joker_cls)
                        + 0.25 * _future_scaling_bonus(my_picks_j)
                        + _denial_weight(current_pick, is_p1, joker_name) * opp_gain
                        + 0.18 * _engine_pressure_score(opp_picks, joker_cls)
                        + 0.55 * _pair_synergy_bonus(opp_picks, joker_cls)
                    )
                else:
                    val = (
                        my_score_j
                        + _synergy_bonus(joker_cls, my_hand)
                        + _combo_completion_bonus(my_picks, joker_cls)
                        + 0.18 * _future_scaling_bonus(my_picks_j)
                        + _denial_weight(current_pick, is_p1, joker_name) * opp_gain
                        + 0.55 * _pair_synergy_bonus(opp_picks, joker_cls)
                    )
            else:
                opp_score_j = _best_hand_from_candidates(opp_candidates, opp_picks + [joker_cls])
                my_score_if_stolen = _best_hand_from_candidates(my_base_candidates, my_picks + [joker_cls])
                my_gain = max(0, my_score_if_stolen - my_base)

                val = (
                    opp_score_j
                    + _combo_completion_bonus(opp_picks, joker_cls)
                    + 0.20 * _future_scaling_bonus(opp_picks + [joker_cls])
                    + _denial_weight(current_pick, not is_p1, joker_name) * my_gain
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
                val, _ = self._minimax(
                    my_hand=my_hand,
                    opp_hand=opp_hand,
                    opp_candidates=opp_candidates,
                    pool=next_pool,
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
                )
                if val < best_val:
                    best_val = val
                    best_idx = index
                beta = min(beta, best_val)
                if alpha >= beta:
                    break

            return best_val, best_idx


ParticipantBot = Bot