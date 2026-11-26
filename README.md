# Dynamic ability snapshot calibration: implementation

This folder shows how the approach in the paper (`paper/item_response_theory_updated.pdf`) is implemented, with the
data it was evaluated on, so the implementation and the results can be checked.

**The approach in one paragraph.** Classic calibration (IRT) treats each student's ability as fixed, so on a learning
platform questions met early in students' histories look harder than they are (the *static ability bias*). Here each
answer is rated against the student's ability **at the moment of that answer**. An EM loop alternates:

- **E-step:** replay every answer in time order with Elo updates and store each student's rating just before each
  answer (the snapshot `R_s,i`);
- **M-step:** recompute each question's difficulty from its answers, weighting each answer by how surprising it was
  given that snapshot.

Between calibrations, question ratings stay fixed and only student ratings move after each new answer.

## Folder layout

```
paper/item_response_theory_updated.pdf   the paper (Sections 4-5: method, 6-7: experiments and results)
data/raw/          raw platform exports, one per subject (v3 snapshot, 17 Sep - 11 Nov 2026)
data/prepared/     the same data after stage 1 (below); the input the method reads
irt/               the implementation
scripts/           prepare the data, reproduce the paper's results
tests/             unit tests of the equations and the calibration
```

## Data handling

**Stage 1: raw export -> prepared file** (`irt/prepare_export.py`, run by `scripts/prepare_data.py`). Exactly the code
that produced `data/prepared/`:

1. registered users only;
2. a row is **answered** when an option was selected and scored; **unanswered** when the question was opened but has
   no scored answer; never-opened rows are dropped;
3. unanswered rows are timed by when the question was first opened; the **last question opened** in each session is
   marked;
4. a hint recorded on a question that offered none is reset to 0 (export bug);
5. position in session and session length come from the event-time order (the export has no position column);
6. mock vs practice test is kept as `context_type`.

**Stage 2: prepared file -> method input** (`irt/loader.py`, `irt/preprocessing.py`), as in the paper's Section 4.3:

1. unanswered rule: an unanswered question that was the **last one opened** in its session is dropped; every other
   unanswered question is kept as a **wrong** answer;
2. answers are put in time order (ties: position in session, then id) and **repeat attempts** are marked, with the
   days since the previous attempt;
3. response times are **winsorized** per student and topic (own 90th percentile, at most 300 s, if the student has
   more than 10 responses there);
4. each question gets a **guess threshold** `min(10th percentile of its times, 3 s)`; a faster MCQ answer is a
   probable guess.

Steps 3-4 are fitted on the calibration responses only and then applied to later (held-out or live) answers.

## Where each part of the paper is

| paper | code |
|---|---|
| Notation, qualified responses, stable questions (4.1) | `irt/equations.py` `is_qualified`, `irt/mstep.py` `run_mstep` |
| Algorithm 1, EM loop (4.2) | `irt/calibration.py` `calibrate` |
| Preprocessing (4.3) | `irt/loader.py`, `irt/preprocessing.py` |
| E-step, Eqs. 1-2 (4.4) | `irt/estep.py` `run_estep`, `irt/equations.py` `expected_outcome` |
| M-step, Eqs. 3-8 (4.5) | `irt/mstep.py` `correctness_difficulty`, `time_difficulty`, `composite_difficulty`, `difficulty_to_rating`; Eq. 4 in `irt/equations.py` `surprise_weight` |
| Bootstrap, Eqs. 9-10 (4.8) | `irt/estep.py` `run_bootstrap` |
| Question selection, Eq. 12 (4.9) | `irt/model.py` `Model.select_question` |
| Fatigue, Eq. 13 (5.1) | `irt/equations.py` `fatigue_adjusted` |
| Hints, Eqs. 14-18 (5.2-5.3) | `irt/equations.py` `observed_outcome`, `irt/mstep.py` `assign_hint_effectiveness` |
| Effective K, guess damping, repeats, discrimination, context, Eqs. 19-22 (5.4) | `irt/equations.py` `k_student`, `k_multipliers`, `repeat_multiplier`; `irt/mstep.py` `point_biserial`, `discrimination_multiplier` |
| Time winsorizing, Eq. 23 (5.5) | `irt/preprocessing.py` `fit_time_stats` |
| Live updates between calibrations (Algorithm 1, line 7) | `irt/model.py` `Model.update`, `Model.predict` |
| Split and metrics ECE / RMSE / AUC, Eqs. 24-26 (6) | `irt/evaluation.py` |

All hyperparameters are in `irt/config.py`. Defaults are the configuration behind the paper's results: a question is
stable after 10 qualified responses (first attempts that are not probable guesses); the M-step uses the qualified
responses of every student, weighted by each student's rating at that moment; student K is 80 until 10 qualified
attempts, then 40 (the paper's printed default is 40/20); question K 32/16 in the bootstrap; EM stops when no
question rating moves by more than 1 point (at most 50 iterations).

Each auxiliary component is a switch in `Components` (`irt/config.py`): `fatigue`, `guess_detection`,
`repeat_damping`, `discrimination`, `time_signal`, `hint_adjustment`, `context_weight`. All are on, as in the paper.

## Running it

Python 3.10+. The method uses the standard library only; stage 1 needs pandas and numpy.

```bash
pip install -r requirements.txt

python scripts/prepare_data.py --check     # stage 1: re-prepare data/raw/ and check it equals data/prepared/
python scripts/evaluate.py                 # the paper's Table 2, row "Ours" (about 2 minutes)
python scripts/evaluate.py --disable time_signal discrimination   # the same with components switched off
python -m pytest                           # unit tests
```

Using it from code:

```python
from irt import Config, calibrate, read_responses_csv

model = calibrate(read_responses_csv("data/prepared/english.csv"))
p = model.predict(student_id, question_id, "English")            # chance of a correct answer
q = model.select_question(student_id, "English", target=0.70)    # question nearest a 70% chance
p_before = model.update(response)                                 # after each new answer
model.save("model.json")
```