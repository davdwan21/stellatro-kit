# Submission Folder

Edit `bot.py` with your final bot. The tournament loader expects this file to
define a `Bot` class with:

- `pick_joker(self, state) -> int`
- `pick_hand(self, state) -> list[int]`

PLEASE MAKE SURE ALL OF YOUR BOT LOGIC IS IN THE FOLDER

Package it from the repo root.

Windows PowerShell:

```powershell
python starter-kit/scripts/zip_submission.py my_bot_name
```

macOS or Linux:

```bash
python3 starter-kit/scripts/zip_submission.py my_bot_name
```

Make sure to name the zip file with your team name.