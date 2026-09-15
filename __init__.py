"""
A/B Test Analysis Toolkit
=========================

Design the test before running it, validate the assignment, estimate the effect
two ways, and quantify what the result cannot support.
"""

from .design import minimum_detectable_effect, power_at, required_sample_size
from .inference import TestResult, regression_adjusted, two_proportion_test
from .peeking import PeekingResult, peeking_false_positive_rate
from .simulate import simulate_experiment
from .validate import SRMResult, sample_ratio_mismatch

__version__ = "1.0.0"
__all__ = [
    "minimum_detectable_effect",
    "required_sample_size",
    "power_at",
    "simulate_experiment",
    "sample_ratio_mismatch",
    "SRMResult",
    "two_proportion_test",
    "regression_adjusted",
    "TestResult",
    "peeking_false_positive_rate",
    "PeekingResult",
]
