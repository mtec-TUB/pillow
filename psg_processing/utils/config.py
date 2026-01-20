"""
Configuration management for PSG processing.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path

from .alignment import Alignment


@dataclass
class ProcessorConfig:
    """Configuration dataclass for dataset processing."""

    dataset: str
    base_data_dir: str
    data_dir: Path
    ann_dir: Path
    output_dir: Path
    overwrite: bool
    logging_level: str
    action: str
    resample: int
    filter: bool
    filter_freq: dict
    channels: list[str]
    epoch_duration: int
    rm_move: bool
    rm_unk: bool
    n_wake_epochs: int
    alignment: Alignment
    pad_values: list
    min_sleep_epochs: int
    use_annot: bool
    output_format: str


def load_config_file(config_file_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_file_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        ValueError: If file doesn't exist or contains invalid YAML
    """
    file_path = Path(config_file_path)

    if not file_path.exists():
        raise ValueError(f"Configuration file not found: {config_file_path}")

    try:
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_file_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
