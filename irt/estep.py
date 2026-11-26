"""Chronological Elo replays: the one-time bootstrap (Section 4.8) and the E-step (Section 4.4).

Student state is kept per (student, topic): a rating and the number of qualified attempts, which sets the base K.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .config import Config
from .equations import expected_outcome, fatigue_adjusted, is_qualified, k_question, k_student, observed_outcome
from .records import QuestionState, Response

StudentKey = tuple[str, str]   # (student_id, topic_id)


def run_bootstrap(ordered: Sequence[Response], questions: dict[str, QuestionState], cfg: Config
                  ) -> tuple[dict[StudentKey, float], dict[StudentKey, int]]:
    """One joint Elo pass: after each answer both the student's rating (Eq. 9) and the question's rating (Eq. 10)
    move. Stores each answer's snapshot. Returns the final student ratings and qualified-attempt counts."""
    ratings: dict[StudentKey, float] = {}
    attempts: dict[StudentKey, int] = defaultdict(int)
    q_attempts: dict[str, int] = defaultdict(int)
    for r in ordered:
        key = (r.student_id, r.topic_id)
        rs = ratings.get(key, cfg.initial_rating)
        r.snapshot = rs
        q = questions[r.question_id]
        e = fatigue_adjusted(expected_outcome(rs, q.rating, cfg), r, cfg)
        o = observed_outcome(r, q, cfg)
        ks = k_student(r, q, attempts[key], cfg)
        kq = k_question(r, q, q_attempts[r.question_id], cfg)
        ratings[key] = rs + ks * (o - e)
        q.rating = q.rating + kq * (e - o)
        if is_qualified(r, cfg):
            attempts[key] += 1
            q_attempts[r.question_id] += 1
    return ratings, dict(attempts)


def run_estep(ordered: Sequence[Response], questions: dict[str, QuestionState], cfg: Config
              ) -> tuple[dict[StudentKey, float], dict[StudentKey, int]]:
    """Replay every answer in time order updating student ratings only (Eq. 2), question ratings fixed. Stores the
    rating just before each answer as its snapshot R_s,i."""
    ratings: dict[StudentKey, float] = {}
    attempts: dict[StudentKey, int] = defaultdict(int)
    for r in ordered:
        key = (r.student_id, r.topic_id)
        rs = ratings.get(key, cfg.initial_rating)
        r.snapshot = rs
        q = questions[r.question_id]
        e = fatigue_adjusted(expected_outcome(rs, q.rating, cfg), r, cfg)
        ratings[key] = rs + k_student(r, q, attempts[key], cfg) * (observed_outcome(r, q, cfg) - e)
        if is_qualified(r, cfg):
            attempts[key] += 1
    return ratings, dict(attempts)
