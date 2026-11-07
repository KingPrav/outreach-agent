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


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resume_path: Path = ROOT / os.getenv("RESUME_PATH", "profile/resume.pdf")
    notes_path: Path = ROOT / "profile" / "notes.md"
    max_people: int = int(os.getenv("MAX_PEOPLE", "6"))
    outputs: Path = ROOT / "outputs"

    @property
    def cache_dir(self) -> Path:
        return self.outputs / ".cache"

    @property
    def checkpoint_db(self) -> Path:
        return self.outputs / ".state" / "checkpoints.sqlite"
