"""Held-out evaluation (Section 6): time split, live replay of the held-out answers, and the paper's metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .calibration import calibrate
from .config import Config
from .preprocessing import chronological
from .records import Response


def ece(p: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    """Eq. 24: sum over equal-width bins of (n_b / N) * |mean prediction - observed rate|."""
    n_b, sp, sy = [0] * bins, [0.0] * bins, [0.0] * bins
    for pi, yi in zip(p, y, strict=False):
        b = min(int(pi * bins), bins - 1)
        n_b[b] += 1
        sp[b] += pi
        sy[b] += yi
    return sum(abs(sp[b] - sy[b]) for b in range(bins) if n_b[b]) / len(p)


def rmse(p: Sequence[float], y: Sequence[float]) -> float:
    """Eq. 25."""
    return (sum((pi - yi) ** 2 for pi, yi in zip(p, y, strict=False)) / len(p)) ** 0.5


def auc(p: Sequence[float], y: Sequence[float]) -> float:
    """Eq. 26: P(prediction for a correct answer > prediction for a wrong one), ties counted as half."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = sum(1 for v in y if v == 1)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise ValueError("AUC needs both correct and wrong answers")
    return (sum(r for r, v in zip(ranks, y, strict=False) if v == 1) - pos * (pos + 1) / 2.0) / (pos * neg)


def time_split(responses: Sequence[Response], fraction: float = 0.10) -> tuple[list[Response], list[Response]]:
    """The paper's split (Section 6.1): the latest ``fraction`` of responses in each topic are held out."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    by_topic: dict[str, list[Response]] = defaultdict(list)
    for r in chronological(responses):
        by_topic[r.topic_id].append(r)
    held = set()
    for rows in by_topic.values():
        n = max(1, round(len(rows) * fraction))
        held.update(r.response_id for r in rows[-n:])
    train = [r for r in responses if r.response_id not in held]
    test = [r for r in responses if r.response_id in held]
    return train, test


@dataclass
class Scores:
    responses: int
    ece: float
    rmse: float
    auc: float


@dataclass
class EvaluationReport:
    all_responses: Scores
    first_attempts: Scores
    iterations: int
    converged: bool


def _scores(p: list[float], y: list[float]) -> Scores:
    return Scores(responses=len(p), ece=ece(p, y), rmse=rmse(p, y), auc=auc(p, y))


def evaluate(responses: Sequence[Response], config: Config | None = None,
             fraction: float = 0.10) -> EvaluationReport:
    """Calibrate on the earlier responses and predict the held-out ones in time order, each before its answer."""
    train, test = time_split(responses, fraction)
    model = calibrate(train, config)
    preds, ys, first = [], [], []
    for r in chronological(test):
        preds.append(model.update(r))
        ys.append(1.0 if r.is_correct else 0.0)
        first.append(not r.is_repeat)
    fp = [p for p, f in zip(preds, first, strict=False) if f]
    fy = [v for v, f in zip(ys, first, strict=False) if f]
    return EvaluationReport(all_responses=_scores(preds, ys), first_attempts=_scores(fp, fy),
                            iterations=model.info.iterations, converged=model.info.converged)
