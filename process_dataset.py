import logging
import os
import argparse
import sys
from pathlib import Path
import yaml

from datasets.registry import get_dataset, DatasetRegistry
from psg_processing.core import Dataset_Explorer, DatasetProcessor
from psg_processing.utils.config import ProcessorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process sleep datasets for harmonization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Load all registered datasets
        epilog=f"""
                Available datasets:
                {', '.join(DatasetRegistry.list_datasets())}
                """,
    )

    # Configuration file (optional)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file. Adapt config.yaml according to your needs.",
    )

    return parser


def resolve_paths(
    dataset,
    base_data_dir: str,
    data_dir: str | None,
    output_dir: str | None,
    output_format: str,
):
    """Return resolved (data_dir, ann_dir, output_dir).

    - If data_dir or ann_dir not provided, use dataset.dataset_paths() and join with base_data_dir.
    - If output_dir not provided, construct under base_data_dir using dataset_name and output format info.
    """
    if data_dir:
        dset_dir = data_dir
    elif base_data_dir:
        dset_dir = os.path.join(base_data_dir, dataset.dataset_name)
        
    rel_data_dir, rel_ann_dir = dataset.dataset_paths()

    if data_dir:
        psg_dir = os.path.join(data_dir, rel_data_dir)
        ann_dir = os.path.join(data_dir, rel_ann_dir)
    elif base_data_dir:
        psg_dir = os.path.join(base_data_dir, dataset.dataset_name, rel_data_dir)
        ann_dir = os.path.join(base_data_dir, dataset.dataset_name, rel_ann_dir)

    if output_dir:
        output_dir_resolved = os.path.join(
            output_dir, f"{dataset.dset_name}_harmonized", output_format
        )
    elif data_dir:
        output_dir_resolved = os.path.join(
            data_dir, f"{dataset.dset_name}_harmonized", output_format
        )
    elif base_data_dir:
        output_dir_resolved = os.path.join(
            base_data_dir, dataset.dataset_name, f"{dataset.dset_name}_harmonized", output_format
        )

    return dset_dir, psg_dir, ann_dir, output_dir_resolved

def load_config_file(config_file_path: str) -> dict:
    """Load configuration from a YAML file."""

    file_path = Path(config_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    try:
        with open(config_file_path, "r") as f:
            config = ProcessorConfig(**yaml.safe_load(f))
        print(f"Loaded configuration from: {config_file_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

def main(config):
    """
    Process a dataset.
    """
    # Get the dataset
    dataset = get_dataset(config.dataset)()

    print(f"Processing dataset: {dataset.dset_name}")

    dataset.dset_dir, config.psg_dir, config.ann_dir, config.output_dir = resolve_paths(
        dataset,
        config.base_data_dir,
        config.data_dir,
        config.output_dir,
        config.output_format
    )

    print(f"PSG directory: {config.psg_dir}")
    print(f"Annotation directory: {config.ann_dir}")
    print(f"Output directory: {config.output_dir}")

    # Preprocess file structure if needed (Reordering etc.)
    ret = dataset.preprocess(config.psg_dir, config.ann_dir, config.output_dir)
    if ret is False:
        return

    if config.action == "process":
        # If channels are specified in config, set them in dataset, else use all available
        if config.channels == []:
            config.channels = dataset.channel_names

        # Initialize DatasetProcessor
        processor = DatasetProcessor(dataset, config)
        processor.process_files()

    elif config.action == "get_channel_names":
        explorer = Dataset_Explorer(
            None, dataset, config.psg_dir, config.ann_dir, log_level=config.logging_level
        )
        channels = explorer.get_all_channels()
        print(f"Available channels in {dataset.dset_name}:")
        n_files = len(explorer.psg_fnames)
        for ch, count in channels.most_common():
            print(f"  {count:>4}/{n_files} ({100*count/n_files:>5.1f}%)  {ch}")

    elif config.action == "get_channel_types":
        explorer = Dataset_Explorer(
            None, dataset, config.psg_dir, config.ann_dir,log_level=logging.INFO
        )
        explorer.get_all_channels()
        channel_types = explorer.get_channel_type()
        print(f"Channel types in {dataset.dset_name}: {channel_types}")

    else:
        raise ValueError(f"Unknown action: {config.action}")


if __name__ == "__main__":

    parser = build_parser()
    cli_args = parser.parse_args(sys.argv[1:])

    # Load user's config file if specified
    if cli_args.config:
        config = load_config_file(cli_args.config)
    else:
        # Load defaults from config.yaml
        config = load_config_file("config.yaml")

    main(config)

