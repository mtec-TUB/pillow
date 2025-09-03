"""
Dataset exploration and analysis tools for PSG data.
"""

import os
import glob
import numpy as np
from numba import njit
from tqdm import tqdm
import logging

from ..file_handlers import FileHandlerFactory
from ..utils.logging_manager import LoggingManager

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
        data_dir: str,
        ann_dir: str,
        psg_ext: str,
        ann_ext: str,
        ann_ext2=None,
        log_level=logging.INFO,
    ):
        """Initialize the Dataset_Explorer with empty containers and logger."""
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.psg_ext = psg_ext
        self.ann_ext = ann_ext
        self.ann_ext2 = ann_ext2
        self.psg_fnames = []
        self.ann_fnames = []
        self.ch_names = []
        self.get_channel_types = []
        self.file_factory = FileHandlerFactory()

        # Setup logger with StreamHandler (console only)
        self.logger = LoggingManager.setup_logger(level=log_level)

    def get_files(self):
        """
        Discover and collect PSG signal files and annotation files.

        Args:
            psg_ext (str): File extension pattern for PSG files (default: "*.edf")
            ann_ext (str): File extension pattern for annotation files (default: "*.xml")
            ann_ext2 (str, optional): Second annotation file extension pattern

        Returns:
            tuple: (psg_filenames, annotation_filenames) arrays
        """
        # Discover PSG signal files
        self.logger.info(
            f"Searching for signal files: {os.path.join(self.data_dir, self.psg_ext)}"
        )
        self.psg_fnames = glob.glob(
            os.path.join(self.data_dir, self.psg_ext), recursive=True
        )
        self.psg_fnames.sort()
        self.logger.info(f"Found {len(self.psg_fnames)} signal files")

        # Return early if no annotation files needed
        if self.ann_ext is None:
            self.logger.info("No annotation files requested")
            return self.psg_fnames, None

        # Discover annotation files
        self.logger.info(
            f"Searching for annotation files: {os.path.join(self.ann_dir, self.ann_ext)}"
        )
        self.ann_fnames = glob.glob(
            os.path.join(self.ann_dir, self.ann_ext), recursive=True
        )

        # Add second annotation extension if provided
        if self.ann_ext2:
            self.logger.info(
                f"Searching for additional annotation files: {os.path.join(self.ann_dir, self.ann_ext2)}"
            )
            ann_fnames2 = glob.glob(
                os.path.join(self.ann_dir, self.ann_ext2), recursive=True
            )
            self.ann_fnames.extend(ann_fnames2)
            self.logger.info(f"Found {len(ann_fnames2)} additional annotation files")

        self.ann_fnames.sort()
        self.logger.info(f"Total annotation files: {len(self.ann_fnames)}")

        # Convert to numpy arrays for consistency
        self.psg_fnames = np.asarray(self.psg_fnames)
        self.ann_fnames = np.asarray(self.ann_fnames)

        # Validate that we have matching numbers of files
        if self.ann_ext != "":
            assert len(self.ann_fnames) == len(self.psg_fnames), (
                f"\nAnnotation files: {len(self.ann_fnames)} "
                f"\nPSG files: {len(self.psg_fnames)} "
                f"\n-> Counts don't match"
            )

        return self.psg_fnames, self.ann_fnames

    def get_all_channels(self):
        """
        Discover all available channel names and frequencies across all PSG files.

        Supports multiple file formats: EDF, H5, CSV, and WFDB.

        Returns:
            set: Set of tuples containing (channel_name, frequency) pairs for EDF files,
                 or just channel names for other formats.
        """
        self.logger.info("Getting all available channel names ...")

        self.get_files()
        self.logger.info(f"Found {len(self.psg_fnames)} files to process")

        self.ch_names = set()

        # Use tqdm for clean progress bar
        for psg_fname in tqdm(self.psg_fnames, desc="Processing files", unit="file"):
            handler = self.file_factory.get_handler(psg_fname)

            if handler:
                channels = handler.get_channels(psg_fname)
                self.ch_names.update(channels)
            else:
                self.logger.warning(f"Unsupported file format for {psg_fname}")

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
        self.logger.info("\nAnalyzing channel types (digital vs analog)...")
        self.logger.info(
            f"Found {len(self.ch_names)} channels to analyze across {len(self.psg_fnames)} files"
        )
        self.logger.info(
            "\nTIP: Press Ctrl+C during any channel analysis to skip remaining files"
        )
        self.logger.info("    and classify that channel as DIGITAL immediately.\n")

        channel_types = {"analog": [], "digital": []}
        total_channels = len(self.ch_names)

        # Main progress bar for channels
        channel_progress = tqdm(
            self.ch_names, 
            desc="Analyzing channels", 
            unit="channel",
            leave=True,
            ncols=100
        )

        for channel_idx, channel in enumerate(channel_progress):
            try:
                # Update main progress bar description to show current channel
                channel_progress.set_description(f"Analyzing: {channel[:20]}")

                # Check multiple files to determine channel type
                is_analog_found = False
                files_checked = 0

                # Use nested tqdm for file checking within each channel
                file_progress = tqdm(
                    self.psg_fnames,
                    desc="  Checking files",
                    unit="file",
                    leave=False,
                    ncols=80,
                )

                for psg_fname in file_progress:
                    handler = self.file_factory.get_handler(psg_fname)

                    if not handler:
                        continue

                    # Update file progress description with current file
                    file_progress.set_postfix_str(
                        f"{os.path.basename(psg_fname)[:25]}..."
                    )

                    signal = handler.read_signal(psg_fname, channel)

                    if signal is None:
                        continue  # Channel not found in this file

                    files_checked += 1

                    if not self._is_digital(signal):
                        # If any file shows analog signal, classify as analog
                        file_progress.close()
                        channel_types["analog"].append(channel)
                        channel_progress.set_postfix_str(f"ANALOG ({files_checked} files)")
                        is_analog_found = True
                        break

                # Close file progress bar if not already closed
                if not file_progress.disable:
                    file_progress.close()

                # If no analog signals found, classify as digital
                if not is_analog_found:
                    channel_types["digital"].append(channel)
                    channel_progress.set_postfix_str(f"DIGITAL ({files_checked} files)")

            except KeyboardInterrupt:
                # Handle user interruption gracefully
                channel_progress.close()
                self.logger.warning(f"\n\nKeyboard interrupt detected!")
                self.logger.warning(
                    f"   Classifying '{channel}' as DIGITAL and continuing..."
                )

                if channel not in channel_types["analog"]:
                    channel_types["digital"].append(channel)

                # Ask if user wants to continue or stop completely
                try:
                    user_choice = (
                        input("\nContinue with next channel? (y/N): ").strip().lower()
                    )
                    if user_choice not in ["y", "yes"]:
                        self.logger.info(
                            "Analysis stopped by user. Returning partial results..."
                        )
                        break
                    
                    # Restart the progress bar
                    remaining_channels = list(self.ch_names)[channel_idx+1:]
                    channel_progress = tqdm(
                        remaining_channels,
                        desc="Analyzing channels",
                        unit="channel", 
                        leave=True,
                        ncols=100
                    )
                    self.logger.info("Continuing with analysis...\n")
                except KeyboardInterrupt:
                    self.logger.info("\nAnalysis stopped completely by user.")
                    break

            except Exception as e:
                channel_progress.set_postfix_str(f"ERROR - defaulting to DIGITAL")
                channel_types["digital"].append(channel)  # Default to digital on error

        # Close the main progress bar
        if 'channel_progress' in locals():
            channel_progress.close()

        # Print final summary
        self.logger.info(f"\nAnalysis complete!")
        self.logger.info(f"Analog channels: {len(channel_types['analog'])}")
        self.logger.info(f"Digital channels: {len(channel_types['digital'])}")

        return channel_types

    def _is_digital(self, signal):
        """
        Determine if a signal is digital or analog based on the number of unique values.

        Digital signals typically have a limited number of discrete values,
        while analog signals have continuous values.

        Args:
            signal (numpy.ndarray): Input signal to analyze

        Returns:
            bool: True if signal appears to be digital, False if analog
        """
        # Use the compiled version for performance
        unique_count = Dataset_Explorer._count_unique_values(signal)

        # Debug output using logger
        if unique_count <= DIGITAL_SIGNAL_MAX_UNIQUE_VALUES:
            self.logger.debug(
                f"Number of unique values in signal: {unique_count} - DIGITAL"
            )
        else:
            self.logger.debug(
                f"Number of unique values in signal: {unique_count} - ANALOG"
            )

        return unique_count <= DIGITAL_SIGNAL_MAX_UNIQUE_VALUES

    @staticmethod
    @njit
    def _count_unique_values(signal):
        """
        Count unique values in a signal efficiently with numba compilation.

        This uses a simple algorithm that works with numba:
        - Sort the signal
        - Count consecutive different values
        """
        if signal.size == 0:
            return 0

        # Quick check: if signal is constant, only 1 unique value
        if signal.max() == signal.min():
            return 1

        # Create a sorted copy for efficient unique counting
        sorted_signal = np.sort(signal.flatten())

        unique_count = 1  # At least one unique value
        for i in range(1, sorted_signal.size):
            if sorted_signal[i] != sorted_signal[i - 1]:
                unique_count += 1
                # Early exit if we exceed the threshold
                if unique_count > DIGITAL_SIGNAL_MAX_UNIQUE_VALUES:
                    return unique_count

        return unique_count
