"""Reading response exports.

The expected CSV columns are those of the platform's prepared export:

    response_id, anon_user_id, anon_session_id, question_id, question_type, topic, is_correct,
    time_spent_seconds, answered_at, hint_level_used, position_in_session, session_length,
    response_kind, last_opened, context_type, [user_type]

Unanswered questions follow the paper's rule (Section 4.3): an unanswered question that was the last one opened in
its session is dropped; any other unanswered question is kept as a wrong answer. If a ``user_type`` column is
present, only registered users are kept.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .records import Response

REQUIRED_COLUMNS = ("response_id", "anon_user_id", "question_id", "question_type", "topic", "is_correct",
                    "time_spent_seconds", "answered_at", "hint_level_used", "position_in_session",
                    "session_length", "response_kind", "last_opened")


def _flag(value: str) -> bool:
    return str(value).strip().upper() == "TRUE"


def _number(value: str, default: float = 0.0) -> float:
    value = (value or "").strip()
    return default if value == "" else float(value)


def read_responses_csv(path: str | Path) -> list[Response]:
    """Load a response export into Response records, applying the user filter and the unanswered rule."""
    out: list[Response] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)}")
        for row in reader:
            if "user_type" in row and row["user_type"].strip().lower() != "registered":
                continue
            if row["response_kind"] == "skipped" and _flag(row["last_opened"]):
                continue                                    # left the session on this question
            out.append(Response(
                response_id=row["response_id"],
                student_id=row["anon_user_id"],
                question_id=row["question_id"],
                topic_id=row["topic"],
                answered_at=datetime.fromisoformat(row["answered_at"]),
                is_correct=_flag(row["is_correct"]),        # unanswered rows are FALSE
                time_spent_seconds=_number(row["time_spent_seconds"]),
                hint_level_used=int(_number(row["hint_level_used"])),
                position_in_session=int(row["position_in_session"]),
                session_length=int(row["session_length"]),
                is_mcq=row["question_type"].strip().upper() == "MCQ",
                context_type=(row.get("context_type") or "practice").strip(),
            ))
    return out
