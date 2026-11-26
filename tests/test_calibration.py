import copy
import math

import pytest

from irt import Config, Model, calibrate, evaluate, time_split
from irt.evaluation import auc, ece, rmse

from .conftest import make_responses


def test_calibration_converges_and_recovers_difficulty_order(responses):
    model = calibrate(responses)
    assert model.info.converged
    # questions answered correctly more often must be rated easier
    by_q = {}
    for r in responses:
        by_q.setdefault(r.question_id, []).append(r.is_correct)
    stable = [q for q in model.questions.values() if q.stable]
    assert len(stable) >= 30
    share = {q: sum(v) / len(v) for q, v in by_q.items()}
    easiest = min(stable, key=lambda q: q.rating)
    hardest = max(stable, key=lambda q: q.rating)
    assert share[easiest.question_id] > share[hardest.question_id]


def test_calibration_is_deterministic(responses):
    a = calibrate(copy.deepcopy(responses))
    b = calibrate(copy.deepcopy(responses))
    assert {k: q.rating for k, q in a.questions.items()} == {k: q.rating for k, q in b.questions.items()}


def test_unstable_questions_keep_bootstrap_rating():
    rows = make_responses(n_students=30, n_questions=60, per_student=8, seed=3)
    model = calibrate(rows, Config(min_attempts=10))
    unstable = [q for q in model.questions.values() if not q.stable]
    assert unstable and all(q.difficulty is None for q in unstable)


def test_live_update_moves_student_and_returns_prior_prediction(responses):
    train, test = time_split(responses)
    model = calibrate(train)
    r = sorted(test, key=lambda x: x.sort_key())[0]
    before = model.student_rating(r.student_id, r.topic_id)
    p = model.update(r)
    assert p == pytest.approx(1 / (1 + 10 ** ((model.question_rating(r.question_id) - before) / 400)))
    after = model.student_rating(r.student_id, r.topic_id)
    assert (after > before) == r.is_correct


def test_repeat_is_detected_live(responses):
    model = calibrate(responses)
    r = copy.deepcopy(responses[-1])
    r.response_id = "again"
    model.update(r)
    assert r.is_repeat and r.days_since_last_attempt == 0.0


def test_model_round_trip(tmp_path, responses):
    model = calibrate(responses)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = Model.load(path)
    assert loaded.config == model.config
    for qid, q in model.questions.items():
        assert loaded.questions[qid] == q
    assert loaded.student_ratings == model.student_ratings
    assert loaded.predict("s1", "q1", "t") == model.predict("s1", "q1", "t")


def test_select_question_targets_success_rate(responses):
    model = calibrate(responses)
    qid = model.select_question("s1", "t", target=0.7)
    ideal = model.student_rating("s1", "t") + 400 * math.log10(0.3 / 0.7)
    best = min(abs(q.rating - ideal) for q in model.questions.values())
    assert abs(model.question_rating(qid) - ideal) == pytest.approx(best)


def test_components_can_be_switched_off(responses):
    on = calibrate(copy.deepcopy(responses))
    off = calibrate(copy.deepcopy(responses), Config().with_components(time_signal=False, discrimination=False))
    assert {k: q.rating for k, q in on.questions.items()} != {k: q.rating for k, q in off.questions.items()}


def test_evaluate_reports_metrics(responses):
    report = evaluate(responses)
    for s in (report.all_responses, report.first_attempts):
        assert 0 <= s.ece <= 1 and 0 <= s.rmse <= 1 and 0.5 < s.auc <= 1


def test_metrics_on_known_values():
    assert rmse([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert ece([0.7, 0.7], [1, 0]) == pytest.approx(0.2)
    assert auc([0.9, 0.1, 0.5], [1, 0, 0]) == pytest.approx(1.0)
    assert auc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)


def test_time_split_holds_out_latest_per_topic(responses):
    train, test = time_split(responses, 0.1)
    assert len(test) == round(len(responses) * 0.1)
    assert max(r.answered_at for r in train) <= min(r.answered_at for r in test)
