"""
Dataset exploration and analysis tools for PSG data.
"""

import os
import glob
import numpy as np
from numba import njit
from tqdm import tqdm
from natsort import natsorted
from collections import Counter
import logging

from ..utils import LoggingManager

# Amplitude resolution threshold (2^7)
DIGITAL_SIGNAL_MAX_UNIQUE_VALUES = 128


class Dataset_Explorer:
    """
    A class for exploring and analyzing polysomnography (PSG) datasets.

    This class helps discover available files, analyze channel information,
    and determine signal characteristics across different file formats.
    """

    def __init__(
        self,
        logger,
        dataset: object,
        data_dir: str,
        ann_dir: str,
        log_level=logging.INFO,
    ):
        """Initialize the Dataset_Explorer with empty containers and logger."""
        self.dataset = dataset
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.psg_fnames = []
        self.ann_fnames = []
        self.ch_names = Counter()
        self.get_channel_types = []

        # Setup logger with StreamHandler (console only)
        if not logger:
            logging_manager = LoggingManager(console_level=log_level)
            self.logger = logging_manager.create_pipeline_logger()
        else:
            self.logger = logger

    def get_files(self):
        """
        Discover and collect PSG signal files and annotation files.

        Returns:
            tuple: (psg_filenames, annotation_filenames) arrays
        """
        # Discover PSG signal files
        if not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory does not exist: {self.data_dir}")
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        
        psg_ext = self.dataset.file_extensions['psg_ext']

        self.logger.info(
            f"Searching for signal files: {os.path.join(self.data_dir, psg_ext)}"
        )

        self.psg_fnames = glob.glob(
            os.path.join(self.data_dir, psg_ext), recursive=True
        )
        # self.psg_fnames = natsorted(self.psg_fnames)
        self.logger.info(f"Found {len(self.psg_fnames)} signal files")

        # Discover annotation files
        if not os.path.exists(self.ann_dir):
            self.logger.error(f"Annotation directory does not exist: {self.ann_dir}")
            raise FileNotFoundError(
                f"Annotation directory does not exist {self.ann_dir}"
            )

        ann_ext = self.dataset.file_extensions['ann_ext']
        self.logger.info(
            f"Searching for annotation files: {os.path.join(self.ann_dir, ann_ext)}"
        )

        self.ann_fnames = glob.glob(
            os.path.join(self.ann_dir, ann_ext), recursive=True
        )

        # Add second annotation extension if provided
        if 'ann_ext2' in self.dataset.file_extensions:
            ann_ext2 = self.dataset.file_extensions['ann_ext2']
            self.logger.info(
                f"Searching for additional annotation files: {os.path.join(self.ann_dir, ann_ext2)}"
            )
            ann_fnames2 = glob.glob(
                os.path.join(self.ann_dir, ann_ext2), recursive=True
            )
            self.ann_fnames.extend(ann_fnames2)
            self.logger.info(f"Found {len(ann_fnames2)} additional annotation files")

        # self.ann_fnames = natsorted(self.ann_fnames)
        self.logger.info(f"Total annotation files: {len(self.ann_fnames)}")

        # Convert to numpy arrays for consistency
        self.psg_fnames = np.asarray(self.psg_fnames)
        self.ann_fnames = np.asarray(self.ann_fnames)

        # # Validate that we have matching numbers of files
        # if ann_ext != "" and len(self.ann_fnames) != len(self.psg_fnames):
        #     self.logger.warning(
        #         f"Number of PSG files and Annotation files do not match: ({len(self.psg_fnames)}/{len(self.ann_fnames)})"
        #     )

        return self.psg_fnames, self.ann_fnames

    def get_all_channels(self):
        """
        Discover all available channel names and frequencies across all PSG files.

        Returns:
            set: Set of tuples containing (channel_name, frequency) pairs for EDF files,
                 or just channel names for other formats.
        """
        self.logger.info("Getting all available channel names ...")

        self.get_files()

        self.ch_names = Counter()

        # Use tqdm for clean progress bar
        for psg_fname in tqdm(self.psg_fnames, desc="Processing files", unit="file"):
            channels = self.dataset.get_channels(self.logger, psg_fname)
            # for label, freq in zip(channels, freqs):
            #     self.ch_names.add((label, float(freq)))
            if not channels:
                self.logger.warning(f"Skipping file with no readable channels: {os.path.basename(psg_fname)}")
                continue
            self.ch_names.update(channels)

        self.logger.info(
            f"Discovery complete! Found {len(self.ch_names)} unique channels across all files."
        )

        return self.ch_names

    def get_channel_type(self):
        """
        Analyze channels to determine if they contain digital or analog signals.

        Returns:
            dict: Dictionary with 'analog' and 'digital' keys containing lists of channels
        """
        self.logger.info("Analyzing channel types (digital vs analog)...")
        self.logger.info(
            f"Found {len(self.ch_names)} channels to analyze across {len(self.psg_fnames)} files"
        )

        channel_types = {"analog": [], "digital": []}

        # Set default to digital and only change to analog if we find evidence of it being analog in any file
        channel_names = self.ch_names.keys()  # Get just the channel names without frequencies
        channel_dict = {channel: 'digital' for channel in channel_names}

        channel_dict = self.analyze_channel_types(0, channel_dict)

        # Convert to final lists
        for channel, ch_type in channel_dict.items():
            channel_types[ch_type].append(channel)

        # Print final summary
        self.logger.info(f"\nAnalysis complete!")
        self.logger.info(f"Analog channels: {len(channel_types['analog'])}")
        self.logger.info(f"Digital channels: {len(channel_types['digital'])}")

        return channel_types
    
    def analyze_channel_types(self, psg_idx, channel_dict):
        try:
            outer_bar = tqdm(self.psg_fnames[psg_idx:], desc="Analyzing files", unit="file")
            for psg_idx, psg_fname in enumerate(outer_bar):
                channels = self.dataset.get_channels(self.logger, psg_fname)

                for channel in tqdm(channels, desc=f"Checking channels in {os.path.basename(psg_fname)}", unit="channel", leave=False):
                    if channel in channel_dict and channel_dict[channel] == 'digital':
                        signal = self.dataset.read_signal(self.logger, psg_fname, channel)
                        if signal is not None and not self._is_digital(signal):
                            channel_dict[channel] = 'analog'

                still_digital = [ch for ch, t in channel_dict.items() if t == "digital"]
                outer_bar.set_postfix({"n_digital": len(still_digital)})

                if all(ch_type == 'analog' for ch_type in channel_dict.values()):
                    break  # All channels classified as analog, no need to continue checking

        except KeyboardInterrupt:
            self.logger.warning(f"\n\nKeyboard interrupt detected during channel analysis!")
            # Convert to final lists
            channel_types = {"analog": [], "digital": []}
            for channel, ch_type in channel_dict.items():
                channel_types[ch_type].append(channel)
            print(f"Current status of analysis: {channel_types}")
            exec = input("Do you want to continue analysing (y) or finalizing the results now (n):")  # Wait for user input before finalizing
            if str(exec).lower() == "y":
                self.analyze_channel_types(psg_idx, channel_dict)  # Continue analysis recursively
            else:
                self.logger.info(f"Finalizing results with current analysis status...")
        except Exception as e:
            self.logger.error(f"Error during channel type analysis: {e}")   
        finally:  
            return channel_dict

    def _is_digital(self,signal):
        """
        Determine if a signal is digital or analog based on the number of unique values.

        Digital signals typically have a limited number of discrete values,
        while analog signals have continuous values.

        Args:
            signal (numpy.ndarray): Input signal to analyze

        Returns:
            bool: True if signal appears to be digital, False if analog
        """
        if signal.size == 0:
            return False

        unique = set()

        for v in signal.flat:
            unique.add(v)
            if len(unique) > DIGITAL_SIGNAL_MAX_UNIQUE_VALUES:
                return False

        return True
