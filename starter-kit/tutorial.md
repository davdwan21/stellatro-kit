# Participant Tutorial

## What Is Stellatro?

Stellatro is a two-player drafting and hand-selection game inspired by Balatro.

In a single game:

1. Both players receive a 10-card hand.
2. A shared joker pool is revealed.
3. Players alternate drafting jokers from that pool.
4. Each player chooses exactly 5 cards to play.
5. The engine scores those 5 cards together with the drafted jokers.
6. The higher final score wins.

Your bot is the program that makes those choices for a player.

## Files You Should Care About

s
When you are starting out, focus on these files:

- `starter-kit/bots/participant_bot.py`
  This is the entry point that tells the project which bot class to use.
- `starter-kit/bots/bot_interface.py`
  This shows the methods your bot should implement.
- `starter-kit/bots/random_bot.py`
  This is the smallest example bot.
- `starter-kit/scripts/run_bot_match.py`
  This runs local bot-vs-bot matches in the terminal.
- `starter-kit/gui/README.md`
  This explains how to watch games in the GUI.
- `starter-kit/jokers.md`
  This lists joker names, descriptions, and when they take effect.

## Which Bot Should I Start From?

For a simple first bot, start with `starter-kit/bots/random_bot.py` or create a
small bot like the one in this tutorial. `starter-kit/bots/participant_bot.py`
is the local entry point you can update when you want the starter kit to use
your bot by default.

For a stronger rules-based bot, read `starter-kit/bots/greedy_bot.py`. It uses
exact hand scoring to pick useful jokers and then plays its best 5-card hand.

For a search-based bot, read `starter-kit/bots/minimax_bot.py`. It looks ahead
through the joker draft with minimax and alpha-beta pruning.

## Step 1: Set Up Python

From the repo root, create a virtual environment and install dependencies.

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```shell
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After you do this once, activate the environment whenever you work on the project.

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## Step 2: Run Something Before You Change Anything

Before writing your own code, make sure the project works on your machine.

Run:

```bash
python starter-kit/scripts/run_bot_match.py random random --rounds 3
```

Then list the built-in bots:

```bash
python starter-kit/scripts/run_bot_match.py --list-bots
```

## Step 3: Understand The Bot Interface

Open `starter-kit/bots/bot_interface.py`.

The two required methods are:

```python
def pick_joker(self, game_state) -> int:
    ...

def pick_hand(self, game_state) -> list[int]:
    ...
```

- `pick_joker(...)` chooses one joker from the current joker pool.
- `pick_hand(...)` chooses which cards to play from your hand.

The `game_state` object gives your bot information about the current game.

Some useful fields are:

- `game_state.joker_pool`
- `game_state.player1_hand`
- `game_state.player2_hand`
- `game_state.player1_jokers`
- `game_state.player2_jokers`
- `game_state.current_turn`
- `game_state.phase`

Important note: the current starter environment exposes both players' hands in
`GameState`, so bots can inspect full information.

## How The Game Actually Runs

### Round structure

At the start of a round, the engine:

1. Shuffles a standard 52-card deck
2. Deals 10 cards to Player 1
3. Deals 10 cards to Player 2
4. Generates a shared joker pool

In the canonical engine:

- `JOKER_POOL_SIZE = 15`
- `JOKER_HAND_SIZE = 5`

Players alternate picks until both players have drafted 5 jokers. After that,
the game moves to play phase.

Note: the local script `starter-kit/scripts/run_bot_match.py` defaults to a
smaller `--joker-pool-size 10` for faster testing.

### Draft phase

The game starts in `Phase.DRAFT` and Player 1 picks first.

Turn order is:

1. Player 1 drafts one joker from the shared pool
2. Player 2 drafts one joker from the updated pool
3. Repeat until both players have 5 jokers

### Play phase

After drafting, the engine switches to `Phase.PLAY`.

Then:

1. Player 1 submits exactly 5 card indices
2. The engine scores Player 1's selected hand
3. Player 2 submits exactly 5 card indices
4. The engine scores Player 2's selected hand
5. The game ends

Your play action must be a list of exactly 5 integers with no duplicates.

## How Hands Are Classified

The hand checker lives in `stellatro-game/stellatro_game/checker.py`.

Supported hand types in ascending order:

- High Card
- Pair
- Two Pair
- Three of a Kind
- Straight
- Flush
- Full House
- Four of a Kind
- Straight Flush

The checker also marks which cards are "scored" for that hand type. Only the cards in the highest hand type will score (e.g. Straight Flush > Four of a Kind).

## Base Score Table

Before joker effects, each hand type gives a base chips/multiplier pair:

| Hand Type       | Base Chips | Base Mult |
| --------------- | ---------: | --------: |
| High Card       |         10 |         1 |
| Pair            |         20 |         1 |
| Two Pair        |         30 |         2 |
| Three of a Kind |         40 |         2 |
| Straight        |         60 |         3 |
| Flush           |         70 |         3 |
| Full House      |         90 |         4 |
| Four of a Kind  |        120 |         5 |
| Straight Flush  |        160 |         6 |

## How Card Ranks Score

When a scored card is processed, it contributes chips based on rank:

- `2` through `10` score their numeric value
- `J`, `Q`, `K` score `10`
- `A` scores `11`

## Exact Scoring Order

When the engine evaluates your played 5-card hand, it does this:

- Step A: Reset per-hand card state
- Step B: Run all `pre_card_phase` joker effects
- Step C: Classify the hand
- Step D: Start from the base chips and multiplier
- Step E: Process each scored card: for each scored card,
  1. The card contributes rank-based chips.
  2. If the card has retriggers, this happens once per trigger.
  3. On each trigger, every joker's `apply_card_phase` effect runs.
- Step F: Run all `post_card_phase` joker effects
- Step G: Compute final score: `final_score = chips * mult`

## Step 4: Write A Very Small Bot

Create a new file:

`starter-kit/bots/my_first_bot.py`

Put this code in it:

```python
from typing import Any, Dict, List

from bots.bot_interface import BotInterface
from stellatro_common import GameState, PlayerTurn


class MyFirstBot(BotInterface):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)

    def pick_joker(self, game_state: GameState) -> int:
        # Always pick the first joker in the pool.
        if not game_state.joker_pool:
            return 0
        return 0

    def pick_hand(self, game_state: GameState) -> List[int]:
        # Figure out which hand belongs to this turn.
        if game_state.current_turn == PlayerTurn.PLAYER1:
            hand = game_state.player1_hand
        else:
            hand = game_state.player2_hand

        # Play the first 5 cards.
        return list(range(5))
```

## Step 5: Tell Stellatro To Use Your Bot

Now open `starter-kit/bots/participant_bot.py`.

Change it so it imports your class and sets `PARTICIPANT_BOT` to your bot:

```python
from bots.my_first_bot import MyFirstBot

PARTICIPANT_BOT = MyFirstBot
```

## Step 6: Run Your Bot

Now test your bot against the built-in simple bot (alias `random`):

```bash
python starter-kit/scripts/run_bot_match.py bots.my_first_bot:MyFirstBot random --rounds 5
```

## Step 7: Watch Games In The GUI

Sometimes text output is enough, but sometimes it helps a lot to actually watch
the game. You can run the GUI from source:

```bash
python starter-kit/gui/gui.py --p1 starter-kit/bots/my_first_bot.py --p2 starter-kit/bots/random_bot.py
```

If you are on macOS or Windows, there may also be a prebuilt app. See
`starter-kit/gui/README.md` for the full instructions.

## Step 8: Make The Bot Slightly Smarter

### First easy improvement: better joker drafting

Right now, your first bot always picks joker `0`.

A simple improvement:

- Look at the joker descriptions in `starter-kit/jokers.md`
- If your hand already has a pair, prefer jokers that reward pairs

### Second easy improvement: better hand selection

Playing the first five cards is usually not the best move.

A better idea:

1. Look at all possible 5-card subsets of your hand.
2. Estimate which subset is strongest.
3. Return the indices for that subset.

For example, you can start with a simpler rule:

- If you see a pair, play the pair plus your highest remaining cards.
- Otherwise, play your five highest cards.

Since each player has 10 cards and must choose 5, there are:

- `C(10, 5) = 252`

possible hands to consider, which is small enough that exact search is often practical.

## Suggested Learning Path

If you are not sure what to do next, use this order:

1. Make a bot that runs.
2. Make joker picking slightly better.
3. Make hand picking slightly better.
4. Test against `random`.
5. Test against stronger bots.
6. Refactor your code once it starts working.

## Where To Look Next

- `starter-kit/jokers.md`
  Use this when you want to know what a joker actually does.
- `starter-kit/bots/random_bot.py`
  Use this when you want to see the smallest working bot.
- `starter-kit/bots/minimax_bot.py`
  Use this when you want to study a stronger bot after you understand the basics.
- `starter-kit/gui/README.md`
  Use this when you want to test visually.
- `stellatro-game/stellatro_game/game.py`
  Use this when you want the exact draft and scoring flow.
- `stellatro-game/stellatro_game/checker.py`
  Use this when you want the exact hand classification rules.

## Note

When submitting, please make sure to put all of your logic in your zip!
