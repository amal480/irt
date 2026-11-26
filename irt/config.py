"""Configuration: every hyperparameter of the method and an on/off switch for each auxiliary component.

Defaults are the configuration used for the results in the paper: the separate-gates method (a question is
stable after ``min_attempts`` qualified responses, the M-step uses qualified responses from every student, and the
student K switches after ``min_attempts`` qualified attempts) with student K 80/40.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Components:
    """Auxiliary components (paper Section 5). Each can be switched off without touching the code."""

    fatigue: bool = True
    """Eq. 13: the expected outcome falls with the position in the session."""

    guess_detection: bool = True
    """Section 5.4: fast MCQ answers are probable guesses. Correct ones are damped in the rating update, and all of
    them are left out of the M-step and of the qualified-response counts."""

    repeat_damping: bool = True
    """Eq. 20: a repeat attempt moves the student's rating less, recovering with the days since the last attempt."""

    discrimination: bool = True
    """Eqs. 21-22: answers on questions that separate strong from weak students move the rating more."""

    time_signal: bool = True
    """Eqs. 6-7: the question's difficulty is adjusted by how much longer wrong answers take than correct ones."""

    hint_adjustment: bool = True
    """Eqs. 14-18: hint use changes the observed outcome and the M-step weights (credit and penalty)."""

    context_weight: bool = True
    """Section 5.4: answers given in mock tests move the rating more than practice answers."""


@dataclass(frozen=True)
class Config:
    """Hyperparameters. Symbols in brackets are the paper's."""

    initial_rating: float = 1500.0          # R0
    scale: float = 400.0                    # S, Elo logistic scale
    min_attempts: int = 10                  # MIN_ATTEMPTS, both gates

    # Base K (Section 5.4). The paper's printed default is 40/20 for students; the experiments use 80/40.
    k_student_new: float = 80.0
    k_student_stable: float = 40.0
    k_question_new: float = 32.0            # bootstrap only
    k_question_stable: float = 16.0

    # EM loop (Algorithm 1): stop when no question rating moves by more than em_tolerance rating points.
    em_max_iterations: int = 50
    em_tolerance: float = 1.0

    # M-step
    time_weight: float = 0.20               # w_t, Eq. 7
    difficulty_eps: float = 0.01            # clamp of difficulty(q) before Eq. 8

    # Auxiliary components
    fatigue_delta: float = 0.05             # delta, Eq. 13
    penalty_scale: float = 0.15             # sigma, Eqs. 14-17
    hint_prior_users: float = 10.0          # m, Eq. 18 (small-sample shrinkage toward the topic value)
    max_hint_level: int = 3
    guess_multiplier: float = 0.2           # m_guess for a flagged correct answer
    guess_percentile: float = 10.0          # P10 of the question's response times
    guess_time_floor: float = 3.0           # T_floor, seconds
    repeat_floor: float = 0.1               # beta_r, Eq. 20
    repeat_time_constant_days: float = 10.0  # tau_r, Eq. 20
    discrimination_floor: float = 0.1       # beta_d, Eq. 22
    discrimination_good: float = 0.30       # r_good, Eq. 22
    mock_context_weight: float = 1.5        # m_context for mock tests

    # Time winsorizing (Eq. 23)
    time_cap_seconds: float = 300.0         # T_max
    winsor_min_responses: int = 10
    winsor_percentile: float = 90.0

    components: Components = field(default_factory=Components)

    def __post_init__(self) -> None:
        positive = ("scale", "k_student_new", "k_student_stable", "k_question_new", "k_question_stable",
                    "em_tolerance", "time_cap_seconds", "repeat_time_constant_days", "discrimination_good")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_attempts < 1 or self.em_max_iterations < 1:
            raise ValueError("min_attempts and em_max_iterations must be at least 1")
        if not 0 < self.difficulty_eps < 0.5:
            raise ValueError("difficulty_eps must be in (0, 0.5)")
        if not 0 <= self.guess_percentile <= 100 or not 0 <= self.winsor_percentile <= 100:
            raise ValueError("percentiles must be in [0, 100]")
        if self.max_hint_level < 0 or self.hint_prior_users < 0:
            raise ValueError("max_hint_level and hint_prior_users must be non-negative")

    def with_components(self, **switches: bool) -> Config:
        """A copy with some components switched, e.g. ``cfg.with_components(time_signal=False)``."""
        return replace(self, components=replace(self.components, **switches))
