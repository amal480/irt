import math

import pytest

from irt import time_split
from irt.baselines import elo, p_correct, rasch

from .conftest import make_responses


def test_p_correct_is_the_smoothed_running_share():
    rows = make_responses(n_students=40, n_questions=10, per_student=10, seed=1)
    train, test = time_split(rows)
    preds = p_correct(train, test)
    first = sorted(test, key=lambda r: r.sort_key())[0]
    right = sum(r.is_correct for r in train if r.question_id == first.question_id)
    n = sum(1 for r in train if r.question_id == first.question_id)
    g = (sum(r.is_correct for r in train) + 1) / (len(train) + 2)
    assert preds[first.response_id] == pytest.approx((right + 2 * g) / (n + 2))


def test_elo_first_prediction_is_one_half_and_stays_in_range():
    rows = make_responses(n_students=30, n_questions=10, per_student=10, seed=2)
    train, test = time_split(rows)
    preds = elo(train, test)
    assert len(preds) == len(test) and all(0 < p < 1 for p in preds.values())
    assert elo([], rows[:1])[rows[0].response_id] == pytest.approx(0.5)


def test_rasch_ranks_easy_questions_above_hard_ones():
    rows = make_responses(seed=4)
    train, test = time_split(rows)
    preds = rasch(train, test)
    assert len(preds) == len(test) and all(0 < p < 1 for p in preds.values())
    share = {}
    for r in train:
        share.setdefault(r.question_id, []).append(r.is_correct)
    # for one student, the easiest question (by share correct) must get a higher chance than the hardest one
    some = sorted(test, key=lambda r: r.sort_key())[0]
    easy = max(share, key=lambda q: sum(share[q]) / len(share[q]))
    hard = min(share, key=lambda q: sum(share[q]) / len(share[q]))
    probe = [some.__class__(**{**some.__dict__, "response_id": f"p{q}", "question_id": q}) for q in (easy, hard)]
    out = rasch(train, probe)
    assert out[f"p{easy}"] > out[f"p{hard}"]
    assert not math.isnan(out[f"p{easy}"])
