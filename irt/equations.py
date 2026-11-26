"""Response-level equations of the paper (Eqs. 1, 3, 4, 13, 14, 16, 17, 19, 20, 22)."""

from __future__ import annotations

import math

from .config import Config
from .records import QuestionState, Response


def expected_outcome(student_rating: float, question_rating: float, cfg: Config) -> float:
    """Eq. 1: E = 1 / (1 + 10^((R_q - R_s) / S))."""
    return 1.0 / (1.0 + 10.0 ** ((question_rating - student_rating) / cfg.scale))


def theta_hat(student_rating: float, cfg: Config) -> float:
    """Eq. 3: the snapshot on a 0-1 scale, 1 / (1 + 10^(-(R_s - R0) / S))."""
    return 1.0 / (1.0 + 10.0 ** (-(student_rating - cfg.initial_rating) / cfg.scale))


def fatigue_adjusted(expected: float, r: Response, cfg: Config) -> float:
    """Eq. 13: E * (1 - delta * p), p = (position - 1) / max(length - 1, 1)."""
    if not cfg.components.fatigue:
        return expected
    p = (r.position_in_session - 1) / max(r.session_length - 1, 1)
    return expected * (1.0 - cfg.fatigue_delta * p)


def hint_credit(q: QuestionState | None, level: int, cfg: Config) -> float:
    """hc(k, q), Eq. 16; 1 when hints are switched off or unknown."""
    if not cfg.components.hint_adjustment:
        return 1.0
    if q is not None and level in q.hint_credit:
        return q.hint_credit[level]
    return 1.0


def hint_penalty(q: QuestionState | None, level: int, cfg: Config) -> float:
    """hp(k, q), Eq. 17; 0 when hints are switched off or unknown (hp(0, q) = 0 by definition)."""
    if not cfg.components.hint_adjustment:
        return 0.0
    if q is not None and level in q.hint_penalty:
        return q.hint_penalty[level]
    return 0.0


def observed_outcome(r: Response, q: QuestionState | None, cfg: Config) -> float:
    """Eq. 14: O = is_correct * hc(k) - (1 - is_correct) * hp(k)."""
    level = r.hint_level_used
    if r.is_correct:
        return hint_credit(q, level, cfg)
    return -hint_penalty(q, level, cfg)


def is_guess(r: Response, cfg: Config) -> bool:
    """A probable guess: an answer faster than the question's threshold (guess detection on)."""
    return cfg.components.guess_detection and r.is_fast


def is_qualified(r: Response, cfg: Config) -> bool:
    """A qualified response: a first attempt that is not a probable guess."""
    return not r.is_repeat and not is_guess(r, cfg)


def repeat_multiplier(r: Response, cfg: Config) -> float:
    """Eq. 20: beta_r + (1 - beta_r)(1 - exp(-g / tau_r)) for a repeat, g in days; 1 for a first attempt."""
    if not cfg.components.repeat_damping or not r.is_repeat or r.days_since_last_attempt is None:
        return 1.0
    g = max(r.days_since_last_attempt, 0.0)
    return cfg.repeat_floor + (1.0 - cfg.repeat_floor) * (1.0 - math.exp(-g / cfg.repeat_time_constant_days))


def k_multipliers(r: Response, q: QuestionState | None, cfg: Config) -> float:
    """m_guess * m_repeat * m_disc * m_context (Eq. 19)."""
    m_guess = cfg.guess_multiplier if (r.is_correct and is_guess(r, cfg)) else 1.0
    m_disc = q.discrimination if (q is not None and cfg.components.discrimination) else 1.0
    m_context = cfg.mock_context_weight if (cfg.components.context_weight and r.context_type == "mock_test") else 1.0
    return m_guess * repeat_multiplier(r, cfg) * m_disc * m_context


def k_student(r: Response, q: QuestionState | None, qualified_attempts: int, cfg: Config) -> float:
    """Effective student K: base K (switching after min_attempts qualified attempts) times the multipliers."""
    base = cfg.k_student_new if qualified_attempts < cfg.min_attempts else cfg.k_student_stable
    return base * k_multipliers(r, q, cfg)


def k_question(r: Response, q: QuestionState | None, qualified_responses: int, cfg: Config) -> float:
    """Bootstrap question K, analogous to the student K."""
    base = cfg.k_question_new if qualified_responses < cfg.min_attempts else cfg.k_question_stable
    return base * k_multipliers(r, q, cfg)


def surprise_weight(r: Response, q: QuestionState | None, cfg: Config) -> float:
    """Eq. 4: (1 - theta_hat) * hc(k) for a correct answer, theta_hat * (1 + hp(k)) for a wrong one."""
    th = theta_hat(r.snapshot, cfg)
    if r.is_correct:
        return (1.0 - th) * hint_credit(q, r.hint_level_used, cfg)
    return th * (1.0 + hint_penalty(q, r.hint_level_used, cfg))
