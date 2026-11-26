"""Data records: a response (one student answering one question) and the state kept for each question."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Response:
    """One answer. The first block is input; the second is filled in by preprocessing and calibration."""

    response_id: str
    student_id: str
    question_id: str
    topic_id: str
    answered_at: datetime
    is_correct: bool
    time_spent_seconds: float = 0.0
    hint_level_used: int = 0
    position_in_session: int = 1
    session_length: int = 1
    is_mcq: bool = True
    context_type: str = "practice"          # "mock_test" or "practice"

    # derived
    is_repeat: bool = False                 # the student answered this question before
    days_since_last_attempt: float | None = None
    time: float = 0.0                       # winsorized response time (Eq. 23)
    is_fast: bool = False                   # faster than the question's guess threshold
    snapshot: float = 0.0                   # R_s,i: the student's rating just before this answer

    def sort_key(self) -> tuple[datetime, int, str]:
        """Chronological order; ties broken by position in the session, then by id."""
        return (self.answered_at, self.position_in_session, self.response_id)


@dataclass
class QuestionState:
    """What the method knows about one question."""

    question_id: str
    topic_id: str
    rating: float                            # R_q
    qualified_responses: int = 0
    stable: bool = False                     # has an M-step difficulty
    difficulty: float | None = None       # difficulty(q), Eq. 7
    point_biserial: float | None = None   # r_pb, Eq. 21
    discrimination: float = 1.0              # m_disc, Eq. 22
    hint_credit: dict[int, float] = field(default_factory=dict)    # hc(k, q)
    hint_penalty: dict[int, float] = field(default_factory=dict)   # hp(k, q)
