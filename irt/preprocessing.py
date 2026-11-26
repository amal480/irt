"""Preprocessing (Section 4.3): chronological order, repeat attempts, time winsorizing and guess thresholds.

Time statistics are always fitted on the calibration responses and then applied to any response, so that held-out
or live answers never influence them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .config import Config
from .records import Response


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Percentile with linear interpolation (numpy's default)."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def chronological(responses: Iterable[Response]) -> list[Response]:
    return sorted(responses, key=lambda r: r.sort_key())


def mark_repeats(ordered: Sequence[Response], last_attempt: dict[tuple[str, str], datetime] = None) -> None:
    """Set is_repeat and days_since_last_attempt from earlier answers of the same student to the same question.
    ``last_attempt`` (updated in place) carries history from before ``ordered``."""
    last = {} if last_attempt is None else last_attempt
    for r in ordered:
        key = (r.student_id, r.question_id)
        prev = last.get(key)
        r.is_repeat = prev is not None
        r.days_since_last_attempt = (r.answered_at - prev).total_seconds() / 86400.0 if prev is not None else None
        last[key] = r.answered_at


@dataclass
class TimeStats:
    """Per (student, topic) time caps (Eq. 23) and per question guess thresholds (Section 5.4)."""

    caps: dict[tuple[str, str], float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    def cap(self, r: Response, cfg: Config) -> float:
        return self.caps.get((r.student_id, r.topic_id), cfg.time_cap_seconds)

    def threshold(self, question_id: str, cfg: Config) -> float:
        return self.thresholds.get(question_id, cfg.guess_time_floor)

    def apply(self, r: Response, cfg: Config) -> None:
        """Winsorized time and the fast-answer flag for one response."""
        r.time = min(r.time_spent_seconds, self.cap(r, cfg))
        r.is_fast = bool(r.is_mcq and r.time < self.threshold(r.question_id, cfg))


def fit_time_stats(responses: Sequence[Response], cfg: Config) -> TimeStats:
    """Caps: the student's own P90 in the topic (at most T_max) when they have more than ``winsor_min_responses``
    responses there, else T_max. Thresholds: min(P10 of the question's winsorized times, T_floor)."""
    times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in responses:
        times[(r.student_id, r.topic_id)].append(r.time_spent_seconds)
    stats = TimeStats()
    for key, values in times.items():
        stats.caps[key] = (min(cfg.time_cap_seconds, percentile(sorted(values), cfg.winsor_percentile))
                           if len(values) > cfg.winsor_min_responses else cfg.time_cap_seconds)
    q_times: dict[str, list[float]] = defaultdict(list)
    for r in responses:
        q_times[r.question_id].append(min(r.time_spent_seconds, stats.cap(r, cfg)))
    stats.thresholds = {q: min(percentile(sorted(v), cfg.guess_percentile), cfg.guess_time_floor)
                        for q, v in q_times.items()}
    return stats
