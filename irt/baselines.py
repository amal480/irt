"""The paper's baselines (Section 6.2): naive p-correct, standard Elo and Rasch (1PL) by marginal maximum likelihood.

Each function takes the calibration responses and the held-out responses (both after the same loading and
preprocessing as the method) and returns, for every held-out response, the predicted chance of a correct answer made
before that answer. None uses the method's adjustments (fatigue, guessing, repeats, hints, discrimination).
p-correct and Elo keep updating through the held-out responses, in time order; Rasch is fitted once.
Rasch needs numpy; the other two use only the standard library.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from .preprocessing import chronological
from .records import Response

R0, S = 1500.0, 400.0
LN10_S = math.log(10) / S


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def p_correct(train: Sequence[Response], test: Sequence[Response]) -> dict[str, float]:
    """The question's share of correct answers so far, smoothed toward the overall share so far (Laplace):
    (right + 2g) / (n + 2), g = (all right + 1) / (all answers + 2); a question with no answers yet gets g."""
    right: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    tot_right = tot = 0

    def chance(qid: str) -> float:
        g = (tot_right + 1) / (tot + 2)
        p = (right[qid] + 2 * g) / (n[qid] + 2) if n[qid] else g
        return min(max(p, 1e-4), 1 - 1e-4)

    held = _ids(test)
    out: dict[str, float] = {}
    for r in chronological(train) + chronological(test):
        if r.response_id in held:
            out[r.response_id] = chance(r.question_id)
        right[r.question_id] += int(r.is_correct)
        n[r.question_id] += 1
        tot_right += int(r.is_correct)
        tot += 1
    return out


def elo(train: Sequence[Response], test: Sequence[Response], k_student: tuple[float, float] = (80.0, 40.0),
        k_question: tuple[float, float] = (32.0, 16.0), min_attempts: int = 10) -> dict[str, float]:
    """Standard Elo (Pelanek 2016): after every answer both the student's and the question's rating move by
    K * (outcome - expected). K is the first value until the student (or question) has min_attempts answers, then the
    second. Expected outcome: 1 / (1 + 10^((R_q - R_s) / S))."""
    rs: dict[str, float] = {}
    rq: dict[str, float] = {}
    att: dict[tuple[str, str], int] = defaultdict(int)
    q_att: dict[str, int] = defaultdict(int)
    held = _ids(test)
    out: dict[str, float] = {}
    for r in chronological(train) + chronological(test):
        s, q = rs.get(r.student_id, R0), rq.get(r.question_id, R0)
        e = _sigmoid((s - q) * LN10_S)
        if r.response_id in held:
            out[r.response_id] = e
        key = (r.student_id, r.topic_id)
        ks = k_student[0] if att[key] < min_attempts else k_student[1]
        kq = k_question[0] if q_att[r.question_id] < min_attempts else k_question[1]
        d = (1.0 if r.is_correct else 0.0) - e
        rs[r.student_id] = s + ks * d
        rq[r.question_id] = q - kq * d
        att[key] += 1
        q_att[r.question_id] += 1
    return out


def rasch(train: Sequence[Response], test: Sequence[Response], n_nodes: int = 41, max_iter: int = 2000,
          tol: float = 1e-4, b_clip: float = 8.0) -> dict[str, float]:
    """Rasch (1PL) by marginal maximum likelihood (Bock and Aitkin 1981): abilities ~ N(0, sigma^2) are integrated
    out with Gauss-Hermite quadrature, and EM estimates every question's difficulty b and sigma. A student's ability
    for prediction is the posterior mean given their calibration answers (EAP); an unseen student gets 0 (the
    population mean), an unseen question the mean b. A question answered all correct or all wrong has no finite b and
    is clipped to +-b_clip logits. Fitted once on the calibration responses."""
    import numpy as np

    students = sorted({r.student_id for r in train})
    items = sorted({r.question_id for r in train})
    si = {s: i for i, s in enumerate(students)}
    qi = {q: i for i, q in enumerate(items)}
    ns, nq = len(students), len(items)
    s_idx = np.array([si[r.student_id] for r in train])
    q_idx = np.array([qi[r.question_id] for r in train])
    y = np.array([float(r.is_correct) for r in train])
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    logw = np.log(w / w.sum())
    right = np.bincount(q_idx, y, nq)
    tot = np.bincount(q_idx, minlength=nq).astype(float)
    p0 = (right + 0.5) / (tot + 1.0)
    b = np.log((1.0 - p0) / p0)
    sigma = 1.0

    def posterior(b, sigma):
        theta = sigma * x
        z = theta[None, :] - b[q_idx][:, None]
        ll = np.where(y[:, None] == 1.0, -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z))
        log_post = np.stack([np.bincount(s_idx, ll[:, k], ns) for k in range(n_nodes)], axis=1) + logw[None, :]
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        return theta, post / post.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        theta, post = posterior(b, sigma)
        weights = post[s_idx]
        b_old = b.copy()
        for _ in range(3):                                    # Newton steps on every b, posterior fixed
            pk = 1.0 / (1.0 + np.exp(-(theta[None, :] - b[q_idx][:, None])))
            f = np.bincount(q_idx, (weights * (y[:, None] - pk)).sum(axis=1), nq)
            fp = np.bincount(q_idx, (weights * pk * (1.0 - pk)).sum(axis=1), nq)
            b = np.clip(b - f / np.maximum(fp, 1e-12), -b_clip, b_clip)
        sigma_new = float(np.sqrt((post * theta[None, :] ** 2).sum(axis=1).mean()))
        done = np.max(np.abs(b - b_old)) < tol and abs(sigma_new - sigma) < tol
        sigma = sigma_new
        if done:
            break
    theta, post = posterior(b, sigma)
    eap = (post * theta[None, :]).sum(axis=1)
    b_mean = float(b.mean())
    out: dict[str, float] = {}
    for r in test:
        t = eap[si[r.student_id]] if r.student_id in si else 0.0
        bb = b[qi[r.question_id]] if r.question_id in qi else b_mean
        out[r.response_id] = float(1.0 / (1.0 + np.exp(-(t - bb))))
    return out


def _ids(responses: Sequence[Response]) -> set[str]:
    return {r.response_id for r in responses}
