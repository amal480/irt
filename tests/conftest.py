import random
from datetime import datetime, timedelta

import pytest

from irt import Response


def make_responses(n_students: int = 120, n_questions: int = 40, per_student: int = 25, seed: int = 0,
                   topic: str = "t") -> list[Response]:
    """Synthetic answers from a Rasch-like world (abilities and difficulties in Elo points)."""
    rng = random.Random(seed)
    difficulty = {f"q{j}": rng.gauss(1500, 200) for j in range(n_questions)}
    t0 = datetime(2026, 1, 1)
    out, k = [], 0
    for s in range(n_students):
        ability = rng.gauss(1500, 200)
        qs = rng.sample(sorted(difficulty), per_student)
        start = t0 + timedelta(days=rng.random() * 30)
        for pos, q in enumerate(qs, 1):
            p = 1 / (1 + 10 ** ((difficulty[q] - ability) / 400))
            out.append(Response(response_id=f"r{k:06d}", student_id=f"s{s}", question_id=q, topic_id=topic,
                                answered_at=start + timedelta(minutes=pos), is_correct=rng.random() < p,
                                time_spent_seconds=rng.uniform(5, 60), position_in_session=pos,
                                session_length=per_student))
            k += 1
    return out


@pytest.fixture
def responses() -> list[Response]:
    return make_responses()
