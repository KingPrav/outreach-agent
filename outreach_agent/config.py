"""Settings, read once from the environment (.env at the project root)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LINKEDIN_NOTE_LIMIT = 300   # LinkedIn's hard cap on connection notes
EMAIL_WORD_LIMIT = 160
EMAIL_WORD_TOLERANCE = 20   # drafts slightly over the target are fine; far over are not
MAX_REVISIONS = 2           # first draft + one redraft


def _price(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resume_path: Path = ROOT / os.getenv("RESUME_PATH", "profile/resume.pdf")
    notes_path: Path = ROOT / "profile" / "notes.md"
    max_people: int = int(os.getenv("MAX_PEOPLE", "6"))
    outputs: Path = ROOT / "outputs"
    gmail_account: str = os.getenv("GMAIL_ACCOUNT", "")   # which signed-in Google account Gmail links open
    # Dollars per million tokens, for the cost estimate in run stats. Unset = no estimate.
    price_input: float | None = _price("OPENAI_PRICE_INPUT_PER_M")
    price_cached_input: float | None = _price("OPENAI_PRICE_CACHED_INPUT_PER_M")
    price_output: float | None = _price("OPENAI_PRICE_OUTPUT_PER_M")

    @property
    def cache_dir(self) -> Path:
        return self.outputs / ".cache"

    @property
    def checkpoint_db(self) -> Path:
        return self.outputs / ".state" / "checkpoints.sqlite"
