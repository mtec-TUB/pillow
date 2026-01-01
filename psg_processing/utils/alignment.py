"""
Alignment options for signal and annotation synchronization.
"""

from enum import Enum


class Alignment(Enum):
    """Options for aligning signal and annotation lengths at front and/or end."""
    MATCH_SHORTER = "match_shorter"        # no padding, but cropping if necessary
    MATCH_LONGER = "match_longer"          # no cropping, but padding with custom values
    MATCH_SIGNAL = "match_signal"   # pad/crop to signal length
    MATCH_ANNOT = "match_annot"     # pad/crop to annotation length
