import logging

import numpy as np
import sys


def get_test_logger() -> logging.Logger:
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    std_handler = logging.StreamHandler(sys.stdout)
    log.addHandler(std_handler)
    return log


def summarize_scores(eval_scores: np.array, scores: np.array, log: logging.Logger):
    latest_scores = scores[-30:].mean()
    polyfit = np.polyfit(np.arange(scores.shape[0]), scores, 1)
    slope = polyfit[0]
    log.info(f"Latest scores: {latest_scores}")
    log.info(
        f"Mean eval: {eval_scores.mean()}, eval std: {eval_scores.std()}, eval 95perc: {np.percentile(eval_scores, 95)}")
    log.info(f"Slope of scores over time: {slope}")
