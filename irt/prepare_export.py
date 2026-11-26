"""Stage 1 of the data handling: turn raw platform exports into the prepared file the method reads.

This is the code that produced data/prepared/ from data/raw/. It uses pandas; the method itself needs only the
standard library.

Rules:

1. Registered users only (paper §4.3).
2. A row is **answered** when an option was selected and scored (`given_answer` and
   `is_correct` both set). That includes rows whose status says NOT_ATTEMPTED: the
   platform scored them, so the student did pick an option.
3. A row is **skipped** when the student opened the question but has no scored answer
   (NOT_ATTEMPTED, MARKED_FOR_REVIEW, or an option selected but never scored).
   NOT_VISITED rows are dropped: the student never saw the question.
4. Skips are placed in time by `first_visited_at`. `last_opened` marks the question the
   student opened last in the session, which §4.3's rule needs. Skips with no visit time
   are dropped (older exports have none).
5. A hint recorded on a question that offered none (`hint_tiers_provided` = 0) is reset to
   0 (bug in the export).
6. Position in session and session length come from event-time order within the session
   (the export has no position column). Event time is `answered_at` for answers and
   `first_visited_at` for skips.
7. Repeats: a later attempt by the same student at the same question.

Several files can be passed; rows repeated across files (same session and question) are
kept once.
"""

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HINT_COL = "hint_level_used (0 for no hint usage)"

REQUIRED = [
    "anon_user_id", "user_type", "anon_session_id", "exam_title", "question_id",
    "question_type", "options", "correct_answer", "given_answer", "is_correct",
    "question_status", "time_spent_seconds", "answered_at", "topic", "mock_or_practice",
]
OPTIONAL = ["question_text", "marks_obtained", "hint_tiers_provided", "first_visited_at",
            "last_visited_at", "visit_count", "answer_change_count"]

PREPARED_COLUMNS = [
    "response_id", "anon_user_id", "anon_session_id", "exam_title", "question_id",
    "question_text", "question_type", "options", "topic", "is_correct", "time_spent_seconds",
    "answered_at", "hint_level_used", "position_in_session", "session_length",
    "is_repeat_attempt", "days_since_last_attempt", "response_kind", "last_opened",
    "context_type", "question_status",
]


class SchemaError(ValueError):
    pass


def read_exports(paths: Sequence[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        if HINT_COL in df.columns:
            df = df.rename(columns={HINT_COL: "hint_level_used"})
        missing = [c for c in REQUIRED + ["hint_level_used"] if c not in df.columns]
        if missing:
            raise SchemaError(f"{path}: missing columns {missing}")
        for c in OPTIONAL:
            if c not in df.columns:
                df[c] = ""
        df["source_file"] = path
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw.drop_duplicates(["anon_session_id", "question_id"], keep="first")


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.replace("", None), format="ISO8601", errors="coerce")


def _options_text(raw: str) -> str:
    try:
        return " | ".join(o.get("text", "") for o in json.loads(raw))
    except Exception:
        return raw


def prepare(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply the rules above. Returns (prepared rows, counts of what each rule did)."""
    c: Dict[str, int] = {"raw_rows": len(raw)}
    df = raw[raw["user_type"].str.strip().str.lower() == "registered"].copy()
    c["dropped_guest_rows"] = len(raw) - len(df)

    df["status"] = df["question_status"].str.strip().str.upper()
    has_answer = df["given_answer"].str.strip() != ""
    scored = has_answer & (df["is_correct"].str.strip() != "")
    df["answered_ts"] = _ts(df["answered_at"])
    df["first_ts"] = _ts(df["first_visited_at"])
    df["last_ts"] = _ts(df["last_visited_at"])

    not_visited = df["status"] == "NOT_VISITED"
    answered = scored & df["answered_ts"].notna()
    skipped = ~answered & ~not_visited
    c["dropped_not_visited"] = int(not_visited.sum())
    c["answered"] = int(answered.sum())
    c["answered_status_not_attempted"] = int((answered & (df["status"] == "NOT_ATTEMPTED")).sum())
    c["skips_option_selected_not_scored"] = int((skipped & has_answer).sum())
    df["kind"] = np.where(answered, "answered", np.where(skipped, "skipped", "drop"))
    df = df[df["kind"] != "drop"].copy()

    df["event_ts"] = np.where(df["kind"] == "answered", df["answered_ts"], df["first_ts"])
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    no_time = (df["kind"] == "skipped") & df["event_ts"].isna()
    c["dropped_skips_without_visit_time"] = int(no_time.sum())
    df = df[~no_time].copy()

    # the question opened last in each session (§4.3 is_last_opened_in_session)
    df["opened_ts"] = df["last_ts"].fillna(df["event_ts"])
    last_idx = df.groupby("anon_session_id")["opened_ts"].idxmax()
    df["last_opened"] = df.index.isin(last_idx)
    c["skips"] = int((df["kind"] == "skipped").sum())
    c["skips_last_opened"] = int(((df["kind"] == "skipped") & df["last_opened"]).sum())

    # hints recorded where none were offered
    hint = pd.to_numeric(df["hint_level_used"], errors="coerce").fillna(0).astype(int)
    offered = pd.to_numeric(df["hint_tiers_provided"], errors="coerce")
    bad_hint = (hint > 0) & (offered == 0)
    c["hints_reset_none_offered"] = int(bad_hint.sum())
    df["hint_level_used"] = np.where(bad_hint, 0, hint)

    # chronological order; position within session by event time
    df = df.sort_values(["event_ts", "anon_session_id", "question_id"], kind="mergesort")
    df["position_in_session"] = df.groupby("anon_session_id").cumcount() + 1
    df["session_length"] = df.groupby("anon_session_id")["question_id"].transform("size")

    # repeats in replay order
    key = ["anon_user_id", "question_id"]
    df["is_repeat_attempt"] = df.duplicated(key, keep="first")
    prev = df.groupby(key)["event_ts"].shift()
    gap = (df["event_ts"] - prev).dt.total_seconds() / 86400.0
    df["days_since_last_attempt"] = gap.round(6).astype(object).where(gap.notna(), "")
    c["repeat_attempts"] = int(df["is_repeat_attempt"].sum())

    out = pd.DataFrame({
        "response_id": [f"R{i:07d}" for i in range(1, len(df) + 1)],
        "anon_user_id": df["anon_user_id"].values,
        "anon_session_id": df["anon_session_id"].values,
        "exam_title": df["exam_title"].values,
        "question_id": df["question_id"].values,
        "question_text": df["question_text"].values,
        "question_type": df["question_type"].values,
        "options": df["options"].map(_options_text).values,
        "topic": df["topic"].values,
        "is_correct": np.where(df["is_correct"].str.strip().str.lower() == "true",
                               "TRUE", "FALSE"),
        "time_spent_seconds": pd.to_numeric(df["time_spent_seconds"], errors="coerce")
                                .fillna(0).values,
        "answered_at": df["event_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").values,
        "hint_level_used": df["hint_level_used"].values,
        "position_in_session": df["position_in_session"].values,
        "session_length": df["session_length"].values,
        "is_repeat_attempt": np.where(df["is_repeat_attempt"], "TRUE", "FALSE"),
        "days_since_last_attempt": df["days_since_last_attempt"].values,
        "response_kind": df["kind"].values,
        "last_opened": np.where(df["last_opened"], "TRUE", "FALSE"),
        "context_type": np.where(df["mock_or_practice"].str.strip().str.lower()
                                 .str.startswith("mock"), "mock_test", "practice"),
        "question_status": df["status"].values,
    })
    # skipped rows are never correct
    out.loc[out["response_kind"] == "skipped", "is_correct"] = "FALSE"
    c["prepared_rows"] = len(out)
    return out[PREPARED_COLUMNS], c


def prepare_files(paths: Sequence[str], out_path: str) -> Dict[str, int]:
    prepared, counts = prepare(read_exports(paths))
    prepared.to_csv(out_path, index=False, encoding="utf-8")
    return counts


def describe_counts(counts: Dict[str, int]) -> List[str]:
    labels = [
        ("raw_rows", "Rows in the export(s)"),
        ("dropped_guest_rows", "Guest rows dropped (paper: registered users only)"),
        ("dropped_not_visited", "Never-opened rows dropped"),
        ("answered", "Answered rows (option selected and scored)"),
        ("answered_status_not_attempted", "… of which marked 'not attempted' (counted as answered)"),
        ("skips", "Skipped rows (opened, no scored answer)"),
        ("skips_option_selected_not_scored", "… of which an option was selected but not scored"),
        ("skips_last_opened", "… of which the last question opened in the session"),
        ("dropped_skips_without_visit_time", "Skips dropped for having no visit time"),
        ("hints_reset_none_offered", "Hint use reset to 0 (no hints offered)"),
        ("repeat_attempts", "Repeat attempts"),
        ("prepared_rows", "Rows in the prepared file"),
    ]
    return [f"| {label} | {counts.get(k, 0):,} |" for k, label in labels]
