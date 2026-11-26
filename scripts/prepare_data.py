"""Stage 1: raw exports (data/raw/) -> prepared files (data/prepared/).

    python scripts/prepare_data.py            # writes data/prepared/<subject>.csv
    python scripts/prepare_data.py --check    # only checks that preparing the raw files reproduces data/prepared/

Each subject's raw export is prepared on its own (irt/prepare_export.py); a summary of what each rule did is printed.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from irt.prepare_export import describe_counts, prepare, read_exports  # noqa: E402

SUBJECTS = ("english", "general_awareness", "general_intelligence_reasoning", "quantitative_aptitude")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with data/prepared/ instead of writing")
    a = ap.parse_args()
    ok = True
    for subject in SUBJECTS:
        raw = os.path.join(ROOT, "data", "raw", f"irt_paper_dataset_{subject}.csv")
        out = os.path.join(ROOT, "data", "prepared", f"{subject}.csv")
        prepared, counts = prepare(read_exports([raw]))
        print(f"\n## {subject}\n")
        print("\n".join(describe_counts(counts)))
        if a.check:
            existing = pd.read_csv(out, dtype=str, keep_default_na=False)
            mine = pd.read_csv(pd.io.common.StringIO(prepared.to_csv(index=False)), dtype=str,
                               keep_default_na=False)
            same = existing.equals(mine)
            ok &= same
            print(f"\nIdentical to data/prepared/{subject}.csv: {'yes' if same else 'NO'}")
        else:
            prepared.to_csv(out, index=False, encoding="utf-8")
            print(f"\nWrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
