"""The calibrated model: question ratings from the last calibration plus live student ratings.

Between calibrations question ratings stay fixed; after each new answer only the student's rating moves (Algorithm 1,
line 7). The model also keeps what live answers need from the calibration: time caps, guess thresholds, each
student's qualified-attempt count, and the last attempt per (student, question) to detect repeats.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import Components, Config
from .equations import expected_outcome, fatigue_adjusted, is_qualified, k_student, observed_outcome
from .preprocessing import TimeStats
from .records import QuestionState, Response

FORMAT_VERSION = 1
StudentKey = tuple[str, str]   # (student_id, topic_id)


@dataclass
class CalibrationInfo:
    iterations: int = 0
    converged: bool = False
    max_rating_changes: list = field(default_factory=list)
    responses: int = 0
    calibrated_at: str | None = None


@dataclass
class Model:
    config: Config
    questions: dict[str, QuestionState]
    student_ratings: dict[StudentKey, float]
    student_attempts: dict[StudentKey, int]
    time_stats: TimeStats
    last_attempt: dict[tuple[str, str], datetime]
    info: CalibrationInfo = field(default_factory=CalibrationInfo)

    # ------------------------------------------------------------------ queries

    def student_rating(self, student_id: str, topic_id: str) -> float:
        return self.student_ratings.get((student_id, topic_id), self.config.initial_rating)

    def question_rating(self, question_id: str) -> float:
        q = self.questions.get(question_id)
        return self.config.initial_rating if q is None else q.rating

    def predict(self, student_id: str, question_id: str, topic_id: str) -> float:
        """P(correct) for the student's next answer to this question (Eq. 1)."""
        return expected_outcome(self.student_rating(student_id, topic_id), self.question_rating(question_id),
                                self.config)

    def select_question(self, student_id: str, topic_id: str, target: float,
                        candidates: Iterable[str] | None = None) -> str | None:
        """Section 4.9, Eq. 12: the candidate whose rating is closest to R_s + S log10((1 - target) / target).
        Candidates default to every known question of the topic."""
        if not 0.0 < target < 1.0:
            raise ValueError("target must be in (0, 1)")
        ideal = self.student_rating(student_id, topic_id) + self.config.scale * math.log10((1.0 - target) / target)
        pool = (list(candidates) if candidates is not None
                else [qid for qid, q in self.questions.items() if q.topic_id == topic_id])
        if not pool:
            return None
        return min(pool, key=lambda qid: abs(self.question_rating(qid) - ideal))

    # ------------------------------------------------------------------ live updates

    def update(self, r: Response) -> float:
        """Record a new answer: derive repeat, time and guess fields, update the student's rating (Eq. 2) and
        return the prediction made just before the answer."""
        cfg = self.config
        key = (r.student_id, r.topic_id)
        prev = self.last_attempt.get((r.student_id, r.question_id))
        r.is_repeat = prev is not None
        r.days_since_last_attempt = (r.answered_at - prev).total_seconds() / 86400.0 if prev is not None else None
        self.last_attempt[(r.student_id, r.question_id)] = r.answered_at
        self.time_stats.apply(r, cfg)

        q = self.questions.get(r.question_id)
        rs = self.student_rating(r.student_id, r.topic_id)
        r.snapshot = rs
        prediction = expected_outcome(rs, self.question_rating(r.question_id), cfg)
        e = fatigue_adjusted(prediction, r, cfg)
        attempts = self.student_attempts.get(key, 0)
        self.student_ratings[key] = rs + k_student(r, q, attempts, cfg) * (observed_outcome(r, q, cfg) - e)
        if is_qualified(r, cfg):
            self.student_attempts[key] = attempts + 1
        return prediction

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        cfg = asdict(self.config)
        return {
            "format_version": FORMAT_VERSION,
            "config": cfg,
            "info": asdict(self.info),
            "questions": [asdict(q) for q in self.questions.values()],
            "students": [{"student_id": s, "topic_id": t, "rating": v,
                          "qualified_attempts": self.student_attempts.get((s, t), 0)}
                         for (s, t), v in self.student_ratings.items()],
            "time_caps": [{"student_id": s, "topic_id": t, "cap": v} for (s, t), v in self.time_stats.caps.items()],
            "guess_thresholds": self.time_stats.thresholds,
            "last_attempt": [{"student_id": s, "question_id": q, "at": ts.isoformat()}
                             for (s, q), ts in self.last_attempt.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Model:
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format {data.get('format_version')!r}")
        raw = dict(data["config"])
        cfg = Config(**{**raw, "components": Components(**raw["components"])})
        questions = {}
        for q in data["questions"]:
            q = dict(q)
            q["hint_credit"] = {int(k): v for k, v in q["hint_credit"].items()}
            q["hint_penalty"] = {int(k): v for k, v in q["hint_penalty"].items()}
            questions[q["question_id"]] = QuestionState(**q)
        students = {(s["student_id"], s["topic_id"]): s["rating"] for s in data["students"]}
        attempts = {(s["student_id"], s["topic_id"]): s["qualified_attempts"] for s in data["students"]}
        stats = TimeStats(caps={(c["student_id"], c["topic_id"]): c["cap"] for c in data["time_caps"]},
                          thresholds=dict(data["guess_thresholds"]))
        last = {(a["student_id"], a["question_id"]): datetime.fromisoformat(a["at"]) for a in data["last_attempt"]}
        return cls(config=cfg, questions=questions, student_ratings=students, student_attempts=attempts,
                   time_stats=stats, last_attempt=last, info=CalibrationInfo(**data["info"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Model:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
