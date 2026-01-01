#!/usr/bin/env python3
"""
Main entry point for processing a sleep dataset.

Usage:
  From command line:
    python process_dataset.py --config custom_config.json
    python process_dataset.py --dataset ABC --data_dir /path/to/ABC/polysomnograpy --output_dir /path/to/output --action process --resample None

  From VS Code (interactive):
    Edit the config.json with your parameters, then run the script.
"""
import os
import argparse
import sys
from pathlib import Path
from enum import Enum

from typing import Optional

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from psg_processing.utils import Alignment, load_config_file, merge_configs, ProcessorConfig
from datasets.registry import get_dataset, DatasetRegistry
from psg_processing.core import Dataset_Explorer, DatasetProcessor
from psg_processing.file_handlers import get_handler

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
    description="Process sleep datasets for harmonization",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=f"""
                Available datasets:
                {', '.join(DatasetRegistry.list_datasets())}
                
                Modify config.json or provide a custom config file with --config to set parameters.
                Command line arguments override config file settings.""",
    )

    # Configuration file (optional)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file. CLI arguments override config file values.",
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        required=False,  # Changed to False since it can come from config file
        choices=DatasetRegistry.list_datasets(),
        type=str.upper,
        help="Dataset to process",
    )

    parser.add_argument(
        "--base_data_dir",
        type=str,
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
        choices=["prepare", "get_channel_names", "get_channel_types"],
        help="Action to perform",
    )
    
    parser.add_argument(
        "--resample",
        type=lambda x: None if x.lower() == "none" else int(x),
        help="Integer resample frequency (Hz) or None"
    )

    parser.add_argument(
        "--filter",
        type=bool,
        help="Whether to apply filtering according to AASM guidelines after resampling"
    )

    parser.add_argument(
        "--channels", 
        nargs='+',
        help="List of desired channel names to process"
    )

    parser.add_argument(
        "--epoch_duration",
        type=int,
        choices=[1,2,3,5,10,15,30],
        help="Epoch duration in seconds"
    )

    parser.add_argument(
        "--rm_move",
        type=bool,
        help="Whether to remove Movement epochs during processing"
    )

    parser.add_argument(
        "--rm_unk",
        type=bool,
        help="Whether to remove Unknown epochs during processing"
    )

    parser.add_argument(
        "--n_wake_epochs",
        type=str,
        help="Number of wake epochs at start and end to keep during processing (default 30min), or 'all' to keep all"
    )

    parser.add_argument(
        "--alignment",
        type=str,
        choices=[a.value for a in Alignment],
        help="How to align signal and annotation lengths, specify pad_value for padding options"
    )

    parser.add_argument(
        "--pad_values",
        nargs=2,
        help="Padding value for signal and labels when using alignment option that require padding (default: None for both)"
    )

    parser.add_argument(
        "--min_sleep_epochs",
        type=int,
        help="Minimum required sleep epochs (N1,N2,N3,REM) after preprocessing (if less, the recording is discarded)",
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

    if config.action == "process":
        # all channels will be processed if empty list
        if config.channels != []:
            dataset.channel_names = config.channels

        # Initialize a new DatasetProcessor
        processor = DatasetProcessor(dataset, config)
        processor.process_files()

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
    # Load defaults from config.yaml (located in same directory as this script)
    script_dir = Path(__file__).parent
    default_config_path = script_dir / "config.yaml"
    defaults = load_config_file(str(default_config_path))
    
    # Determine if we're running from CLI or VS Code interactive mode
    is_vscode_mode = len(sys.argv) == 1
    
    if is_vscode_mode:
        # VS Code mode: use defaults from config.yaml
        args = argparse.Namespace(**defaults)
        main(args)
    else:
        # Normal CLI execution
        parser = build_parser()
        cli_args = parser.parse_args(sys.argv[1:])
        
        # Load user's config file if specified
        file_config = None
        if cli_args.config:
            file_config = load_config_file(cli_args.config)
        
        # Extract only explicitly set CLI args (not None and not 'config' key)
        cli_args_dict = {k: v for k, v in vars(cli_args).items() 
                        if v is not None and k != 'config'}
        
        # Merge with precedence: CLI > user config file > defaults
        merged_config = merge_configs(defaults, file_config, cli_args_dict)
        
        # Validate that dataset is provided
        if not merged_config.get('dataset'):
            parser.error("--dataset is required (via CLI, config file, or config.yaml)")
        
        # Create namespace from merged config
        args = argparse.Namespace(**merged_config)
        main(args)
