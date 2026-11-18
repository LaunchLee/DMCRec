"""Compatibility import for the corrected dataset loaders.

Current experiments use experiments.dmcrec.exdata directly. The revised
loaders apply explicit protocol/split and feature-identity checks; consult
their signatures rather than assuming the historical loader API.
"""
from experiments.dmcrec.exdata import *
