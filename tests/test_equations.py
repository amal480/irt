import math
from datetime import datetime

import pytest

from irt import Config, QuestionState, Response
from irt.equations import (
    expected_outcome,
    fatigue_adjusted,
    is_qualified,
    k_student,
    observed_outcome,
    repeat_multiplier,
    surprise_weight,
    theta_hat,
)
from irt.mstep import composite_difficulty, difficulty_to_rating, discrimination_multiplier

CFG = Config()


def resp(**kw) -> Response:
    base = dict(response_id="r", student_id="s", question_id="q", topic_id="t", answered_at=datetime(2026, 1, 1),
                is_correct=True)
    base.update(kw)
    return Response(**base)


def test_expected_outcome_is_elo():
    assert expected_outcome(1500, 1500, CFG) == pytest.approx(0.5)
    assert expected_outcome(1900, 1500, CFG) == pytest.approx(10 / 11)


def test_theta_hat_centred_on_initial_rating():
    assert theta_hat(1500, CFG) == pytest.approx(0.5)
    assert theta_hat(1900, CFG) == pytest.approx(10 / 11)


def test_fatigue_scales_with_position():
    assert fatigue_adjusted(0.8, resp(position_in_session=1, session_length=11), CFG) == pytest.approx(0.8)
    assert fatigue_adjusted(0.8, resp(position_in_session=11, session_length=11), CFG) == pytest.approx(0.8 * 0.95)
    off = CFG.with_components(fatigue=False)
    assert fatigue_adjusted(0.8, resp(position_in_session=11, session_length=11), off) == pytest.approx(0.8)


def test_observed_outcome_with_hints():
    q = QuestionState("q", "t", 1500, hint_credit={0: 1.0, 1: 0.6}, hint_penalty={0: 0.0, 1: 0.06})
    assert observed_outcome(resp(is_correct=True, hint_level_used=1), q, CFG) == pytest.approx(0.6)
    assert observed_outcome(resp(is_correct=False, hint_level_used=1), q, CFG) == pytest.approx(-0.06)
    no_hints = CFG.with_components(hint_adjustment=False)
    assert observed_outcome(resp(is_correct=True, hint_level_used=1), q, no_hints) == 1


def test_repeat_multiplier_recovers_with_days():
    assert repeat_multiplier(resp(), CFG) == 1.0
    assert repeat_multiplier(resp(is_repeat=True, days_since_last_attempt=0.0), CFG) == pytest.approx(0.1)
    m = repeat_multiplier(resp(is_repeat=True, days_since_last_attempt=10.0), CFG)
    assert m == pytest.approx(0.1 + 0.9 * (1 - math.exp(-1)))


def test_qualified_excludes_repeats_and_guesses():
    assert is_qualified(resp(), CFG)
    assert not is_qualified(resp(is_repeat=True), CFG)
    assert not is_qualified(resp(is_fast=True), CFG)
    assert is_qualified(resp(is_fast=True), CFG.with_components(guess_detection=False))


def test_student_k_switches_after_min_attempts_and_damps_fast_correct():
    q = QuestionState("q", "t", 1500)
    assert k_student(resp(), q, 9, CFG) == 80
    assert k_student(resp(), q, 10, CFG) == 40
    assert k_student(resp(is_fast=True), q, 10, CFG) == pytest.approx(40 * 0.2)
    assert k_student(resp(is_fast=True, is_correct=False), q, 10, CFG) == 40
    assert k_student(resp(context_type="mock_test"), q, 10, CFG) == pytest.approx(60)


def test_surprise_weight():
    q = QuestionState("q", "t", 1500)
    assert surprise_weight(resp(is_correct=True, snapshot=1500), q, CFG) == pytest.approx(0.5)
    assert surprise_weight(resp(is_correct=False, snapshot=1900), q, CFG) == pytest.approx(10 / 11)


def test_difficulty_to_rating_and_time_signal():
    assert difficulty_to_rating(0.5, CFG) == pytest.approx(1500)
    assert difficulty_to_rating(0.0, CFG) == pytest.approx(1500 + 400 * math.log10(0.01 / 0.99))
    assert composite_difficulty(0.5, 0.5, CFG) == pytest.approx(0.5)
    assert composite_difficulty(0.5, 2 / 3, CFG) > 0.5          # wrong answers take longer: harder
    assert composite_difficulty(0.5, 2 / 3, CFG.with_components(time_signal=False)) == 0.5


def test_discrimination_multiplier():
    assert discrimination_multiplier(None, CFG) == 1.0
    assert discrimination_multiplier(0.3, CFG) == pytest.approx(1.0)
    assert discrimination_multiplier(-0.2, CFG) == pytest.approx(0.1)
