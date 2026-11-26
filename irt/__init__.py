"""irt: question difficulty calibration with dynamic ability snapshots.

Typical use::

    from irt import Config, calibrate, read_responses_csv

    model = calibrate(read_responses_csv("data/prepared/english.csv"))   # periodic calibration
    p = model.predict(student_id, question_id, topic_id)       # before serving a question
    model.update(response)                                     # after each answer
    next_q = model.select_question(student_id, topic_id, target=0.7)
    model.save("model.json")
"""

from .calibration import calibrate
from .config import Components, Config
from .evaluation import EvaluationReport, evaluate, time_split
from .loader import read_responses_csv
from .model import Model
from .records import QuestionState, Response

__all__ = ["Components", "Config", "EvaluationReport", "Model", "QuestionState", "Response", "calibrate",
           "evaluate", "read_responses_csv", "time_split"]
__version__ = "1.0.0"
