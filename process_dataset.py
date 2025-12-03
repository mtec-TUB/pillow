#!/usr/bin/env python3
"""
Unified dataset processing for sleep datasets.
This replaces all individual prepare_*.py scripts.
"""
import os
from typing import List, Optional
import argparse
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from datasets import get_dataset, DatasetRegistry

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
    
    parser.add_argument("--resample", type=str, default="100", help="Resample frequency (Hz) or 'None'")

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
    
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    parser.add_argument("--allow_missing", action="store_true", help="Allow missing psg or annotation files during processing")

    return parser

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def _parse_resample(val: str):
    """Return None if the string is 'None' (case-insensitive), otherwise return the original string."""
    if val is None:
        return None
    if isinstance(val, str) and val.lower() == "none":
        return None
    return val

def _resolve_paths(dataset, base_data_dir: str, data_dir: str | None, ann_dir: str | None, output_dir: str | None, resample_val):
    """Return resolved (data_dir, ann_dir, output_dir).

    - If data_dir or ann_dir not provided, use proc.dataset_paths() and join with base_data_dir.
    - If output_dir not provided, construct under base_data_dir using dataset_name and resample info.
    """
    # Start from dataset-provided relative paths
    rel_data_dir, rel_ann_dir = dataset.dataset_paths()

    data_dir_resolved = data_dir if data_dir else os.path.join(base_data_dir, rel_data_dir)
    ann_dir_resolved = ann_dir if ann_dir else os.path.join(base_data_dir, rel_ann_dir)

    if output_dir:
        output_dir_resolved = os.path.join(
            output_dir,
            f"{dataset.dset_name}_harmonized",
            res_label,
        )
    else:
        # If resample_val is None we want the 'orig' folder, else '<N>Hz_filt'
        res_label = "orig" if resample_val is None else f"{resample_val}Hz_filt"
        output_dir_resolved = os.path.join(
            base_data_dir,
            dataset.dataset_name,
            f"{dataset.dset_name}_harmonized",
            res_label,
        )

    return data_dir_resolved, ann_dir_resolved, output_dir_resolved


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Get the dataset
    dataset = get_dataset(args.dataset)
    dataset = dataset()

    print(f"Processing dataset: {dataset.dset_name}")
    resample_val = _parse_resample(args.resample)
    data_dir, ann_dir, output_dir = _resolve_paths(
        dataset, args.base_data_dir, args.data_dir, args.ann_dir, args.output_dir, resample_val
    )

    print(f"Data directory: {data_dir}")
    print(f"Annotation directory: {ann_dir}")
    print(f"Output directory: {output_dir}")

    # Process the dataset
    dataset.process(args.action, data_dir, ann_dir, output_dir, resample_val, args.channels, args.num_jobs, args.overwrite, args.allow_missing)


if __name__ == "__main__":
    main()
