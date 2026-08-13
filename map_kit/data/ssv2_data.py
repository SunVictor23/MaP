"""SSv2 dataset adapter: the MaP 4-way multiple-choice formulation of
Something-Something-v2.

This module holds only the dataset-side helpers needed by the MaP SSv2 runners
(the task key, default paths, and the question/answer templating). The subset
is a class-balanced, hard-negative, fully deterministic 4-way MC set (see the
paper's Supplementary Material); a 10-item demo ships as ssv2_val_subset.json,
and --subset can point at the full file.

The templating mirrors the MVBench option-letter protocol so that the same
``check_ans`` scoring applies unchanged across CLEVRER and SSv2.
"""
import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

TASK_KEY = "action_recognition"
DEFAULT_SUBSET = os.path.join(_PKG_DIR, "ssv2_val_subset.json")
# Point this at your local Something-Something-v2 video directory, or override
# with --video_root on the command line.
DEFAULT_VIDEO_ROOT = os.environ.get(
    "SSV2_VIDEO_ROOT",
    "/path/to/something-something-v2/20bn-something-something-v2")


def ssv2_qa_template(data):
    """Build MC question text + gt "(X) text" from an ABCD-lettered item.

    Subset items carry `options` {A:.., B:.., ...} and `answer` = letter.
    Returns (question_text, "(X) answer_text") so check_ans matches by letter,
    identical to MVBench's option-letter scoring.
    """
    question = f"Question: {data['question']}\nOptions:\n"
    for letter in sorted(data["options"]):
        question += f"({letter}) {data['options'][letter]}\n"
    question = question.rstrip()
    ans_letter = data["answer"]
    answer = f"({ans_letter}) {data['options'][ans_letter]}"
    return question, answer
