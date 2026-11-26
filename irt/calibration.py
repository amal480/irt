"""Calibration (Algorithm 1): preprocessing, bootstrap, then alternate M-step and E-step until the question ratings
stop moving."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone

from .config import Config
from .estep import run_bootstrap, run_estep
from .model import CalibrationInfo, Model
from .mstep import assign_hint_effectiveness, group_by_question, run_mstep
from .preprocessing import chronological, fit_time_stats, mark_repeats
from .records import QuestionState, Response

logger = logging.getLogger(__name__)


def calibrate(responses: Sequence[Response], config: Config | None = None) -> Model:
    """Calibrate question difficulties and student ratings on historical responses.

    The responses are sorted, repeat attempts are marked from their own history, time caps and guess thresholds are
    fitted on them, and the EM loop runs from a one-time joint Elo bootstrap. The responses are modified in place
    (derived fields and snapshots)."""
    cfg = config or Config()
    if not responses:
        raise ValueError("no responses to calibrate on")
    ordered = chronological(responses)
    last_attempt: dict = {}
    mark_repeats(ordered, last_attempt)
    time_stats = fit_time_stats(ordered, cfg)
    for r in ordered:
        time_stats.apply(r, cfg)

    questions: dict[str, QuestionState] = {}
    for r in ordered:
        if r.question_id not in questions:
            questions[r.question_id] = QuestionState(question_id=r.question_id, topic_id=r.topic_id,
                                                     rating=cfg.initial_rating)
    by_question = group_by_question(ordered)
    assign_hint_effectiveness(by_question, questions, cfg)

    ratings, attempts = run_bootstrap(ordered, questions, cfg)
    info = CalibrationInfo(responses=len(ordered))
    for iteration in range(1, cfg.em_max_iterations + 1):
        deltas = run_mstep(by_question, questions, cfg)
        ratings, attempts = run_estep(ordered, questions, cfg)
        change = max(deltas.values()) if deltas else 0.0
        info.iterations = iteration
        info.max_rating_changes.append(change)
        logger.debug("EM iteration %d: largest question rating change %.3f", iteration, change)
        if change < cfg.em_tolerance:
            info.converged = True
            break
    if not info.converged:
        logger.warning("EM stopped after %d iterations without converging (last change %.3f)",
                       info.iterations, info.max_rating_changes[-1])
    info.calibrated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Model(config=cfg, questions=questions, student_ratings=ratings, student_attempts=attempts,
                 time_stats=time_stats, last_attempt=last_attempt, info=info)
