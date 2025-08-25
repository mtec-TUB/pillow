#!/usr/bin/env python3
"""
Unified dataset processor for sleep datasets.
This replaces all individual prepare_*.py scripts.
"""

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
  python process_dataset.py --dataset ABC --data_dir /path/to/ABC/data --output_dir /path/to/output --action prepare
  python process_dataset.py --dataset MESA --data_dir /path/to/MESA/data --action get_channel_names
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
        "--data_dir", 
        required=True,
        help="Directory containing the dataset files"
    )
    
    parser.add_argument(
        "--action", 
        required=True,
        choices=["prepare", "get_channel_names", "get_channel_types"],
        help="Action to perform"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir", 
        help="Output directory for processed files (required for 'prepare' action)"
    )
    
    parser.add_argument(
        "--ann_dir",
        help="Annotation directory (if different from data_dir)"
    )
    
    parser.add_argument(
        "--select_ch", 
        nargs='+',
        help="Specific channels to process (default: all available channels)"
    )
    
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite existing output files"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    
    # Validation
    if args.action == "prepare" and not args.output_dir:
        parser.error("--output_dir is required when action is 'prepare'")
    
    # Get the appropriate processor
    try:
        processor_class = get_processor(args.dataset)
        processor = processor_class()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Process the dataset
        processor.process(args)
        print(f"Successfully processed {args.dataset} dataset")
        return 0
        
    except Exception as e:
        print(f"Error processing {args.dataset} dataset: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
