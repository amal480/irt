"""M-step (Section 4.5): question difficulty from the stored snapshots, plus hint effectiveness and discrimination."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from .config import Config
from .equations import is_guess, is_qualified, surprise_weight
from .records import QuestionState, Response


def group_by_question(responses: Sequence[Response]) -> dict[str, list[Response]]:
    by_q: dict[str, list[Response]] = defaultdict(list)
    for r in responses:
        by_q[r.question_id].append(r)
    return by_q


# --------------------------------------------------------------------------- hint effectiveness (Eqs. 15-18)


def _hint_users(rows: Sequence[Response], cfg: Config) -> list[Response]:
    return [r for r in rows if not r.is_repeat and r.hint_level_used >= 1 and not is_guess(r, cfg)]


def _effectiveness(users: Sequence[Response], cfg: Config) -> list[float]:
    """Eq. 15: eff(k) = share of hint users who answered correctly using at most k hint tiers."""
    levels = range(cfg.max_hint_level + 1)
    if not users:
        return [0.0 for _ in levels]
    return [sum(1 for r in users if r.is_correct and r.hint_level_used <= k) / len(users) for k in levels]


def assign_hint_effectiveness(by_question: dict[str, list[Response]], questions: dict[str, QuestionState],
                              cfg: Config) -> None:
    """Eqs. 16-18: hc = 1 - eff, hp = sigma * eff, with eff shrunk toward the topic's value (pseudo-count m).
    eff uses no ratings, so this is computed once per calibration."""
    by_topic: dict[str, list[Response]] = defaultdict(list)
    for rows in by_question.values():
        by_topic[rows[0].topic_id].extend(rows)
    topic_eff = {t: _effectiveness(_hint_users(rows, cfg), cfg) for t, rows in by_topic.items()}
    m = cfg.hint_prior_users
    for qid, rows in by_question.items():
        users = _hint_users(rows, cfg)
        own, prior, n = _effectiveness(users, cfg), topic_eff[rows[0].topic_id], len(users)
        eff = [(n * o + m * p) / (n + m) for o, p in zip(own, prior, strict=False)] if n + m > 0 else [0.0] * len(own)
        q = questions[qid]
        q.hint_credit = {k: 1.0 - e for k, e in enumerate(eff)}
        q.hint_penalty = {k: cfg.penalty_scale * e for k, e in enumerate(eff)}


# --------------------------------------------------------------------------- discrimination (Eqs. 21-22)


def point_biserial(rows: Sequence[Response]) -> float | None:
    """Eq. 21: ((M1 - M0) / s) * sqrt(p (1 - p)) over the snapshots, population SD. None if undefined."""
    correct = [r.snapshot for r in rows if r.is_correct]
    wrong = [r.snapshot for r in rows if not r.is_correct]
    if not correct or not wrong:
        return None
    values = [r.snapshot for r in rows]
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    if sd == 0:
        return None
    p = len(correct) / n
    return ((sum(correct) / len(correct) - sum(wrong) / len(wrong)) / sd) * math.sqrt(p * (1 - p))


def discrimination_multiplier(r_pb: float | None, cfg: Config) -> float:
    """Eq. 22: beta_d + (1 - beta_d) * min(max(r_pb, 0) / r_good, 1); 1 when undefined."""
    if r_pb is None:
        return 1.0
    return cfg.discrimination_floor + (1.0 - cfg.discrimination_floor) * min(max(r_pb, 0.0) / cfg.discrimination_good,
                                                                             1.0)


# --------------------------------------------------------------------------- difficulty (Eqs. 5-8)


def correctness_difficulty(evidence: Sequence[Response], q: QuestionState, cfg: Config) -> float | None:
    """Eq. 5: weight of the wrong answers over the weight of all answers (surprise weights, Eq. 4)."""
    pos = neg = 0.0
    for r in evidence:
        w = surprise_weight(r, q, cfg)
        if r.is_correct:
            pos += w
        else:
            neg += w
    return None if pos + neg == 0 else neg / (pos + neg)


def time_difficulty(rows: Sequence[Response], cfg: Config) -> float | None:
    """Eq. 6: t2 / (t1 + t2), mean times of wrong (t2) and correct (t1) unaided first attempts that are not guesses."""
    usable = [r for r in rows if r.hint_level_used == 0 and not r.is_repeat and not is_guess(r, cfg)
              and not (r.is_correct and r.is_fast)]
    t1 = [r.time for r in usable if r.is_correct]
    t2 = [r.time for r in usable if not r.is_correct]
    if not t1 or not t2:
        return None
    m1, m2 = sum(t1) / len(t1), sum(t2) / len(t2)
    return None if m1 + m2 == 0 else m2 / (m1 + m2)


def composite_difficulty(d_c: float, d_t: float | None, cfg: Config) -> float:
    """Eq. 7: odds(difficulty) = odds(d_c) * (t2 / t1)^w_t, with d_c clamped to [eps, 1 - eps]."""
    if not cfg.components.time_signal or d_t is None or not 0.0 < d_t < 1.0:
        return d_c
    d = min(max(d_c, cfg.difficulty_eps), 1.0 - cfg.difficulty_eps)
    odds = d / (1.0 - d) * (d_t / (1.0 - d_t)) ** cfg.time_weight
    return odds / (1.0 + odds)


def difficulty_to_rating(difficulty: float, cfg: Config) -> float:
    """Eq. 8: R_q = R0 + S log10(d / (1 - d)), d clamped to [eps, 1 - eps]."""
    d = min(max(difficulty, cfg.difficulty_eps), 1.0 - cfg.difficulty_eps)
    return cfg.initial_rating + cfg.scale * math.log10(d / (1.0 - d))


def run_mstep(by_question: dict[str, list[Response]], questions: dict[str, QuestionState],
              cfg: Config) -> dict[str, float]:
    """One M-step over every question. Only stable questions (at least min_attempts qualified responses) get a new
    rating; the others keep their current one. Returns the absolute rating change of each rewritten question."""
    deltas: dict[str, float] = {}
    for qid, rows in by_question.items():
        q = questions[qid]
        q.qualified_responses = sum(1 for r in rows if is_qualified(r, cfg))
        q.stable = q.qualified_responses >= cfg.min_attempts
        if not q.stable:
            continue
        evidence = [r for r in rows if is_qualified(r, cfg)]
        unaided = [r for r in evidence if r.hint_level_used == 0]
        q.point_biserial = point_biserial(unaided) if len(unaided) >= cfg.min_attempts else None
        q.discrimination = discrimination_multiplier(q.point_biserial, cfg)
        d_c = correctness_difficulty(evidence, q, cfg)
        if d_c is None:
            continue
        q.difficulty = composite_difficulty(d_c, time_difficulty(rows, cfg), cfg)
        new_rating = difficulty_to_rating(q.difficulty, cfg)
        deltas[qid] = abs(new_rating - q.rating)
        q.rating = new_rating
    return deltas
