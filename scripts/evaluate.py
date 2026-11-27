"""Reproduce the paper's Table 2: our method against standard Elo, Rasch and naive p-correct on held-out data.

    python scripts/evaluate.py
    python scripts/evaluate.py --disable time_signal discrimination    # our method with components switched off
    python scripts/evaluate.py --boot 0                                  # values only, no significance marks

For each subject: read data/prepared/<subject>.csv, hold out the latest 10% of responses, fit every method on the
rest, and predict the held-out responses in time order, each before its answer. Scores (ECE, RMSE, AUC) are pooled over
the four subjects, on all held-out responses and on first attempts. A mark means the difference from our method is
clear: its 95% bootstrap interval, resampling held-out students within each subject (2,000 resamples, seed 0),
excludes zero. * = clearly worse than ours, + = clearly better than ours.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from irt import Components, Config, calibrate, read_responses_csv, time_split  # noqa: E402
from irt.baselines import elo, p_correct, rasch  # noqa: E402

SUBJECTS = ("english", "general_awareness", "general_intelligence_reasoning", "quantitative_aptitude")
COMPONENTS = [f.name for f in fields(Components)]
METHODS = ("Ours", "Standard Elo", "Rasch (1PL)", "Naive p-correct")
HIGHER_IS_BETTER = {"AUC"}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    sp, sy, n = np.bincount(b, p, bins), np.bincount(b, y, bins), np.bincount(b, minlength=bins)
    return float(np.sum(np.abs(sp[n > 0] - sy[n > 0])) / len(p))


def rmse(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)   # average ranks for ties
    sums = np.bincount(inv, ranks)
    ranks = (sums / counts)[inv]
    pos = y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


METRICS = {"ECE": ece, "RMSE": rmse, "AUC": auc}


def predict_subject(subject: str, cfg: Config) -> dict:
    """Held-out predictions of every method for one subject, in time order."""
    responses = read_responses_csv(os.path.join(ROOT, "data", "prepared", f"{subject}.csv"))
    train, test = time_split(responses)
    preds = {"Standard Elo": elo(train, test), "Rasch (1PL)": rasch(train, test),
             "Naive p-correct": p_correct(train, test)}
    model = calibrate(train, cfg)
    test = sorted(test, key=lambda r: r.sort_key())
    preds["Ours"] = {r.response_id: model.update(r) for r in test}
    print(f"{subject}: {len(train):,} calibration responses, {len(test):,} held out; our EM "
          f"{model.info.iterations} iterations, converged {model.info.converged}", flush=True)
    return {"y": np.array([1.0 if r.is_correct else 0.0 for r in test]),
            "first": np.array([not r.is_repeat for r in test]),
            "student": [r.student_id for r in test],
            "p": {m: np.array([preds[m][r.response_id] for r in test]) for m in METHODS}}


def pooled_table(data: dict, answer_set: str, boot: int, rng: np.random.Generator) -> list[str]:
    y, p, groups, offset = [], defaultdict(list), [], 0
    for subject in SUBJECTS:
        d = data[subject]
        sel = d["first"] if answer_set == "first" else np.ones(len(d["y"]), bool)
        by_student: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(np.array(d["student"])[sel]):
            by_student[s].append(i)
        groups.append([np.array(g) + offset for g in by_student.values()])
        y.append(d["y"][sel])
        for m in METHODS:
            p[m].append(d["p"][m][sel])
        offset += int(sel.sum())
    y = np.concatenate(y)
    p = {m: np.concatenate(v) for m, v in p.items()}
    samples = [np.concatenate([g[j] for g in groups for j in rng.integers(0, len(g), len(g))]) for _ in range(boot)]
    rows = []
    for m in METHODS:
        cells = []
        for name, fn in METRICS.items():
            value, mark = fn(p[m], y), ""
            if m != "Ours" and boot:
                diffs = np.array([fn(p[m][ix], y[ix]) - fn(p["Ours"][ix], y[ix]) for ix in samples])
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                if not lo <= 0 <= hi:
                    better = (value > fn(p["Ours"], y)) if name in HIGHER_IS_BETTER else (value < fn(p["Ours"], y))
                    mark = "+" if better else "*"
            cells.append(f"{value:.4f}{mark}")
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--disable", nargs="*", default=[], choices=COMPONENTS, metavar="COMPONENT",
                    help=f"switch components of our method off: {', '.join(COMPONENTS)}")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the marks (0 = no marks)")
    a = ap.parse_args()
    cfg = Config().with_components(**{c: False for c in a.disable})
    data = {s: predict_subject(s, cfg) for s in SUBJECTS}
    rng = np.random.default_rng(0)
    for answer_set, title in (("all", "All held-out responses"), ("first", "First attempts")):
        n = sum(int(d["first"].sum()) if answer_set == "first" else len(d["y"]) for d in data.values())
        print(f"\n### {title} ({n:,} responses, four subjects pooled)\n")
        print("| method | ECE (lower better) | RMSE (lower better) | AUC (higher better) |\n|---|---|---|---|")
        print("\n".join(pooled_table(data, answer_set, a.boot, rng)))
    print("\n* clearly worse than ours, + clearly better than ours, no mark: no clear difference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
