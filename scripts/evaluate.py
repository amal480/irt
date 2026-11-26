"""Reproduce the paper's held-out results for our method (Table 2, row "Ours").

    python scripts/evaluate.py
    python scripts/evaluate.py --disable time_signal discrimination

For each subject: read data/prepared/<subject>.csv (stage 2 preprocessing in irt/loader.py and irt/preprocessing.py),
hold out the latest 10% of responses, calibrate on the rest, and predict the held-out responses in time order, each
before its answer. Prints ECE, RMSE and AUC per subject and with the four subjects pooled, on all held-out responses
and on first attempts.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from irt import Components, Config, calibrate, read_responses_csv, time_split  # noqa: E402
from irt.evaluation import auc, ece, rmse  # noqa: E402

SUBJECTS = ("english", "general_awareness", "general_intelligence_reasoning", "quantitative_aptitude")
COMPONENTS = [f.name for f in fields(Components)]


def _line(name: str, p: list, y: list) -> str:
    return f"| {name} | {len(p):,} | {ece(p, y):.4f} | {rmse(p, y):.4f} | {auc(p, y):.4f} |"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--disable", nargs="*", default=[], choices=COMPONENTS, metavar="COMPONENT",
                    help=f"switch components off: {', '.join(COMPONENTS)}")
    a = ap.parse_args()
    cfg = Config().with_components(**{c: False for c in a.disable})

    pooled = {"all": ([], []), "first": ([], [])}
    rows_out = {"all": [], "first": []}
    for subject in SUBJECTS:
        responses = read_responses_csv(os.path.join(ROOT, "data", "prepared", f"{subject}.csv"))
        train, test = time_split(responses)
        model = calibrate(train, cfg)
        p_all, y_all, p_fa, y_fa = [], [], [], []
        for r in sorted(test, key=lambda r: r.sort_key()):
            p, y = model.update(r), (1.0 if r.is_correct else 0.0)
            p_all.append(p)
            y_all.append(y)
            if not r.is_repeat:
                p_fa.append(p)
                y_fa.append(y)
        print(f"{subject}: {len(train):,} calibration responses, {len(test):,} held out; "
              f"EM {model.info.iterations} iterations, converged {model.info.converged}", flush=True)
        for key, (p, y) in (("all", (p_all, y_all)), ("first", (p_fa, y_fa))):
            rows_out[key].append(_line(subject, p, y))
            pooled[key][0].extend(p)
            pooled[key][1].extend(y)

    for key, title in (("all", "All held-out responses"), ("first", "First attempts")):
        print(f"\n### {title}\n\n| subject | responses | ECE | RMSE | AUC |\n|---|---|---|---|---|")
        print("\n".join(rows_out[key]))
        print(_line("**pooled**", *pooled[key]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
