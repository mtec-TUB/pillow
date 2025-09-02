#!/usr/bin/env python3
"""
Unified dataset processor for sleep datasets.
This replaces all individual prepare_*.py scripts.
"""
import os
import argparse
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from dataset_processors import get_processor, DatasetRegistry


def main():
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
                            """
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset", 
        required=True,
        choices=DatasetRegistry.list_datasets(),
        help="Dataset to process"
    )

    parser.add_argument(
        "--base_data_dir", 
        type=str,
        default="/media/linda/Elements/sleep_data",  # Your current base path
        help="Base directory where all sleep datasets are stored"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str,
        help="Specific data directory (overrides automatic path generation)"
    )

    parser.add_argument(
        "--ann_dir", 
        type=str,
        help="Specific annotation directory (overrides automatic path generation)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Specific output directory (overrides automatic path generation)"
    )
    
    parser.add_argument(
        "--action", 
        type=str,
        default="prepare",
        choices=['prepare', 'get_channel_names', 'get_channel_types'],
        help="Action to perform"
    )
    
    parser.add_argument(
        "--resample", 
        type=str,
        default="100",
        help="Resample frequency (Hz) or 'None'"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()

    # Get the processor for this dataset
    processor_class = get_processor(args.dataset)
    processor = processor_class()
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Using processor: {processor_class.__name__}")
    
    # Set up paths if not manually specified
    if not args.data_dir or not args.ann_dir or not args.output_dir:
        data_dir, ann_dir = processor.dataset_paths()
        
        if not args.data_dir:
            data_dir = os.path.join(args.base_data_dir, data_dir)
        if not args.ann_dir:
            ann_dir = os.path.join(args.base_data_dir, ann_dir)
        if not args.output_dir:
            if args.resample == "None":
                output_dir = os.path.join(args.base_data_dir, processor.dataset_name, f"{args.dataset}_harmonized_test", "orig", "npz")
            else:
                output_dir = os.path.join(args.base_data_dir, processor.dataset_name, f"{args.dataset}_harmonized_test", f"{args.resample}Hz_filt", "npz")
    
    print(f"Data directory: {data_dir}")
    print(f"Annotation directory: {ann_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process the dataset
    processor.process(args.action, data_dir, ann_dir, output_dir, args.resample, args.overwrite)


if __name__ == "__main__":
    main()

