#!/usr/bin/env python3
"""
Unified dataset processing for sleep datasets.
This replaces all individual prepare_*.py scripts.

Usage:
  From command line:
    python process_dataset.py --dataset MESA --action prepare --resample 100

  From VS Code (interactive):
    Edit the VS_CODE_CONFIG dict below with your parameters, then run the script.
"""
import os
import argparse
import sys
from pathlib import Path

from dataclasses import dataclass
from typing import Optional

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from datasets.registry import get_dataset, DatasetRegistry
from psg_processing.core import Dataset_Explorer, DatasetProcessor
from psg_processing.file_handlers import get_handler

# ============================================================================
# VS CODE CONFIGURATION
# ============================================================================
# Edit these values to run the script interactively from VS Code
# When running from command line, these will be overridden by CLI arguments
VS_CODE_CONFIG = {
    "dataset": "EESM23",
    "base_data_dir": "/media/linda/Elements/sleep_data",
    "data_dir": None,  # Set to override automatic path
    "ann_dir": None,   # Set to override automatic path
    "output_dir": None,  # Set to override automatic path
    "action": "prepare",
    "resample": None,  # or None for original sampling rate
    "channels": [],  # Empty list means all channels
    "num_jobs": 1,
    "epoch_duration": 30,
    "rm_move": True,
    "rm_unk": True,
    "rm_wake": True,
    "min_sleep_time": 1,
    "overwrite": False,
    "allow_missing": False,
}
# ============================================================================

@dataclass
class ProcessorConfig:
    dataset: str
    base_data_dir: str
    data_dir: Path
    ann_dir: Path
    output_dir: Path
    action: str
    resample: int
    channels: list[str]
    epoch_duration: int
    overwrite: bool
    allow_missing: bool
    num_jobs: int
    rm_move: bool
    rm_unk: bool
    rm_wake: bool
    min_sleep_time: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
    description="Process sleep datasets for harmonization",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=f"""
                Available datasets:
                {', '.join(DatasetRegistry.list_datasets())}
                
                Examples:
                    python process_dataset.py --dataset ABC --data_dir /path/to/ABC/polysomnograpy --output_dir /path/to/output --action prepare --resample None
                    python process_dataset.py --dataset MESA --data_dir /path/to/MESA/ --action get_channel_names
                    python process_dataset.py --dataset SOF --data_dir /path/to/SOF/data --action get_channel_types
                        """,
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DatasetRegistry.list_datasets(),
        type=str.upper,
        help="Dataset to process",
    )

    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="/media/linda/Elements/sleep_data",  # Your current base path
        help="Base directory where all sleep datasets are stored",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Specific data directory (overrides automatic path generation)",
    )

    parser.add_argument(
        "--ann_dir",
        type=str,
        help="Specific annotation directory (overrides automatic path generation)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Specific output directory (overrides automatic path generation)",
    )

    parser.add_argument(
        "--action",
        type=str,
        default="prepare",
        choices=["prepare", "get_channel_names", "get_channel_types"],
        help="Action to perform",
    )
    
    parser.add_argument(
        "--resample",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=100,
        help="Integer resample frequency (Hz) or None"
    )

    parser.add_argument(
        "--channels", 
        nargs='+',
        default=[], # all
        help="List of desired channel names to process"
    )    

    parser.add_argument(
        "--num_jobs", 
        type=int,
        default=1,
        help="Number of parallel jobs to use for processing"
    )

    parser.add_argument(
        "--epoch_duration",
        type=int,
        default=30,
        choices=[1,2,3,5,10,15,30],
        help="Epoch duration in seconds"
    )

    parser.add_argument(
        "--rm_move",
        type=bool,
        default=True,
        help="Whether to remove Movement epochs during processing"
    )

    parser.add_argument(
        "--rm_unk",
        type=bool,
        default=True,
        help="Whether to remove Unknown epochs during processing"
    )

    parser.add_argument(
        "--rm_wake",
        type=bool,
        default=True,
        help="Whether to remove extensive Wake epochs during processing (more than 30min at front and end)"
    )

    parser.add_argument(
        "--min_sleep_time",
        type=int,
        default=1,
        help="Minimum required epochs after preprocessing (if less, the recording is discarded)",
    )
    
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    parser.add_argument("--allow_missing", action="store_true", help="Allow missing psg or annotation files during processing")

    return parser

def _resolve_paths(dataset, base_data_dir: str, data_dir: str | None, ann_dir: str | None, output_dir: str | None, resample_val):
    """Return resolved (data_dir, ann_dir, output_dir).

    - If data_dir or ann_dir not provided, use dataset.dataset_paths() and join with base_data_dir.
    - If output_dir not provided, construct under base_data_dir using dataset_name and resample info.
    """
    # Start from dataset-provided relative paths
    rel_data_dir, rel_ann_dir = dataset.dataset_paths()

    data_dir_resolved = data_dir if data_dir else os.path.join(base_data_dir, rel_data_dir)
    ann_dir_resolved = ann_dir if ann_dir else os.path.join(base_data_dir, rel_ann_dir)

    # If resample_val is None we want the 'orig' folder, else '<N>Hz_filt'
    res_label = "orig" if resample_val is None else f"{resample_val}Hz_filt"

    if output_dir:
        output_dir_resolved = os.path.join(
            output_dir,
            f"{dataset.dset_name}_harmonized",
            res_label,
        )
    else:
        output_dir_resolved = os.path.join(
            base_data_dir,
            dataset.dataset_name,
            f"{dataset.dset_name}_harmonized",
            res_label,
        )

    return data_dir_resolved, ann_dir_resolved, output_dir_resolved


def main(config):
    """
    Process a dataset.
    """
    config = ProcessorConfig(**vars(config))
    # Get the dataset
    dataset = get_dataset(config.dataset)()

    print(f"Processing dataset: {dataset.dset_name}")

    print(type(config.resample))

    config.data_dir, config.ann_dir, config.output_dir = _resolve_paths(
        dataset, config.base_data_dir, config.data_dir, config.ann_dir, config.output_dir, config.resample
    )

    print(f"Data directory: {config.data_dir}")
    print(f"Annotation directory: {config.ann_dir}")
    print(f"Output directory: {config.output_dir}")

    dataset.psg_file_handler = get_handler(dataset.dset_name, dataset.file_extensions['psg_ext'])()

    ret = dataset.preprocess(config.data_dir, config.ann_dir, config.output_dir)
    if ret is False:
        return

    if config.action == "prepare":
        # all channels will be processed if empty list
        if config.channels != []:
            dataset.channel_names = config.channels

        # Initialize a new DatasetProcessor
        processor = DatasetProcessor(dataset, config)
        processor.prepare_files()

    elif config.action == "get_channel_names":
        explorer = Dataset_Explorer(None, dataset.psg_file_handler, config.data_dir, config.ann_dir, **dataset.file_extensions)
        channels = list(explorer.get_all_channels())
        print(f"Available channels in {dataset.dset_name}: {(channels)}")

    elif config.action == "get_channel_types":
        explorer = Dataset_Explorer(None, dataset.psg_file_handler, config.data_dir, config.ann_dir, **dataset.file_extensions)
        explorer.get_all_channels()
        channel_types = explorer.get_channel_type()
        print(f"Channel types in {dataset.dset_name}: {channel_types}")

    else:
        raise ValueError(f"Unknown action: {config.action}")


if __name__ == "__main__":
    # Determine if we're running from CLI or VS Code interactive mode
    is_vscode_mode = len(sys.argv) == 1
    
    if is_vscode_mode:
        # Create a namespace from VS_CODE_CONFIG and pass to main
        args = argparse.Namespace(**VS_CODE_CONFIG)
        main(args)
    else:
        # Normal CLI execution
        parser = build_parser()
        args = parser.parse_args(sys.argv[1:])
        main(args)
