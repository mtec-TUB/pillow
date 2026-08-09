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
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import logging

from ..utils import LoggingManager

# Amplitude resolution threshold (2^7)
DIGITAL_SIGNAL_MAX_UNIQUE_VALUES = 128


@njit(cache=True)
def _is_digital_numba(flat_signal, max_unique):
    """Numba-compiled early-exit unique-value count, called per channel/file."""
    unique_vals = np.empty(max_unique + 1, dtype=flat_signal.dtype)
    n_unique = 0
    for v in flat_signal:
        found = False
        for i in range(n_unique):
            if unique_vals[i] == v:
                found = True
                break
        if not found:
            unique_vals[n_unique] = v
            n_unique += 1
            if n_unique > max_unique:
                return False
    return True


def _is_digital(signal):
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

    return _is_digital_numba(signal.ravel(), DIGITAL_SIGNAL_MAX_UNIQUE_VALUES)


class _ChannelDiscoveryTask:
    """Picklable unit of work run in a worker process: discover channels of one file."""

    def __init__(self, dataset, psg_fname, console_level):
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.console_level = console_level

    def run(self):
        logging_manager = LoggingManager(console_level=self.console_level)
        logger, _ = logging_manager.create_file_logger(os.path.basename(self.psg_fname))
        channels = self.dataset.get_channels(logger, self.psg_fname)
        return self.psg_fname, channels


class _ChannelTypeTask:
    """Picklable unit of work run in a worker process: classify one file's still-digital channels."""

    def __init__(self, dataset, psg_fname, digital_channels, console_level):
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.digital_channels = digital_channels
        self.console_level = console_level

    def run(self):
        logging_manager = LoggingManager(console_level=self.console_level)
        logger, _ = logging_manager.create_file_logger(os.path.basename(self.psg_fname))

        newly_analog = []
        channels = self.dataset.get_channels(logger, self.psg_fname)
        for channel in channels:
            if channel in self.digital_channels:
                signal = self.dataset.read_signal(logger, self.psg_fname, channel)
                if signal is not None and not _is_digital(signal):
                    newly_analog.append(channel)

        return self.psg_fname, newly_analog


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
        num_workers=None,
    ):
        """Initialize the Dataset_Explorer with empty containers and logger."""
        self.dataset = dataset
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.psg_fnames = []
        self.ann_fnames = []
        self.ch_names = Counter()
        self.get_channel_types = []
        self.num_workers = num_workers or os.cpu_count() or 1

        # Setup logger with StreamHandler (console only)
        self.logging_manager = LoggingManager(console_level=log_level)

        self.logger = logger

    def get_files(self, which="both"):
        """
        Discover and collect PSG signal files and annotation files.

        Returns:
            tuple: (psg_filenames, annotation_filenames) arrays
        """
        if not self.logger:
            self.logger = self.logging_manager.create_pipeline_logger()

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
        self.psg_fnames = natsorted(self.psg_fnames)
        self.logger.info(f"Found {len(self.psg_fnames)} signal files")

        if which == "both":
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

            self.logger.info(f"Total annotation files: {len(self.ann_fnames)}")

        # # Validate that we have matching numbers of files
        # if ann_ext != "" and len(self.ann_fnames) != len(self.psg_fnames):
        #     self.logger.warning(
        #         f"Number of PSG files and Annotation files do not match: ({len(self.psg_fnames)}/{len(self.ann_fnames)})"
        #     )

        return self.psg_fnames, self.ann_fnames

    def get_all_channels(self):
        """
        Discover all available channel names and frequencies across all PSG files.

        Files are independent of one another, so they are discovered in a pool of
        worker processes (self.num_workers) instead of sequentially.

        Returns:
            set: Set of tuples containing (channel_name, frequency) pairs for EDF files,
                 or just channel names for other formats.
        """

        self.get_files(which="psg")
        self.logger.info("Getting all available channel names ...")

        self.ch_names = Counter()
        console_level = self.logging_manager.console_level

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            fname_iter = iter(self.psg_fnames)
            tasks = []

            with tqdm(total=len(self.psg_fnames), desc="Processing files", unit="file") as pbar:
                more_files = True
                while True:
                    if not more_files and not tasks:
                        break

                    # Keep the worker pool saturated without queueing every file up-front
                    while len(tasks) < self.num_workers:
                        try:
                            psg_fname = next(fname_iter)
                        except StopIteration:
                            more_files = False
                            break
                        task = _ChannelDiscoveryTask(self.dataset, psg_fname, console_level)
                        tasks.append(executor.submit(task.run))

                    if not tasks:
                        break

                    done, _ = wait(tasks, return_when=FIRST_COMPLETED)
                    for future in done:
                        tasks.remove(future)
                        psg_fname, channels = future.result()
                        if not channels:
                            self.logger.warning(
                                f"Skipping file with no readable channels: {os.path.basename(psg_fname)}"
                            )
                        else:
                            self.ch_names.update(channels)
                        pbar.update(1)

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
        if not self.psg_fnames:
            self.get_files(which="psg")

        self.logger.info("Analyzing channel types (digital vs analog)...")
        if not self.ch_names:
            channel_names = self.dataset.channel_names
        else:
            channel_names = self.ch_names.keys() # Get just the channel names without frequencies

        self.logger.info(
            f"Found {len(channel_names)} channels to analyze across {len(self.psg_fnames)} files"
        )

        channel_types = {"analog": [], "digital": []}

        # Set default to digital and only change to analog if we find evidence of it being analog in any file
        channel_dict = {channel: 'digital' for channel in channel_names}

        channel_dict = self.analyze_channel_types(self.psg_fnames, channel_dict)

        # Convert to final lists
        for channel, ch_type in channel_dict.items():
            channel_types[ch_type].append(channel)

        # Print final summary
        self.logger.info(f"\nAnalysis complete!")
        self.logger.info(f"Analog channels: {len(channel_types['analog'])}")
        self.logger.info(f"Digital channels: {len(channel_types['digital'])}")

        return channel_types

    def analyze_channel_types(self, remaining_fnames, channel_dict):
        """
        Classify channels as digital/analog by scanning files in a pool of worker processes.

        A channel only ever needs to be found analog ONCE, so submission of new files
        stops (and any in-flight ones are cancelled) as soon as every channel has been
        reclassified as analog. Each worker checks a file against the still-digital channel
        set as of submission time.
        """
        console_level = self.logging_manager.console_level
        completed_fnames = set()

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                fname_iter = iter(remaining_fnames)
                tasks = {}

                with tqdm(total=len(remaining_fnames), desc="Analyzing files", unit="file") as outer_bar:
                    more_files = True
                    while True:
                        still_digital = {ch for ch, t in channel_dict.items() if t == 'digital'}
                        if not still_digital:
                            executor.shutdown(cancel_futures=True)
                            break  # All channels classified as analog, no need to continue checking

                        if not more_files and not tasks:
                            break

                        while len(tasks) < self.num_workers and more_files:
                            try:
                                psg_fname = next(fname_iter)
                            except StopIteration:
                                more_files = False
                                break
                            task = _ChannelTypeTask(self.dataset, psg_fname, still_digital, console_level)
                            tasks[executor.submit(task.run)] = psg_fname

                        if not tasks:
                            break

                        done, _ = wait(tasks.keys(), return_when=FIRST_COMPLETED)
                        for future in done:
                            psg_fname = tasks.pop(future)
                            _, newly_analog = future.result()
                            for channel in newly_analog:
                                channel_dict[channel] = 'analog'
                            completed_fnames.add(psg_fname)
                            outer_bar.update(1)

                        n_digital = sum(1 for t in channel_dict.values() if t == 'digital')
                        outer_bar.set_postfix({"n_digital": n_digital})

        except KeyboardInterrupt:
            self.logger.warning(f"\n\nKeyboard interrupt detected during channel analysis!")
            # Convert to final lists
            channel_types = {"analog": [], "digital": []}
            for channel, ch_type in channel_dict.items():
                channel_types[ch_type].append(channel)
            print(f"Current status of analysis: {channel_types}")
            exec = input("Do you want to continue analysing (y) or finalizing the results now (n):")  # Wait for user input before finalizing
            if str(exec).lower() == "y":
                still_pending = [f for f in remaining_fnames if f not in completed_fnames]
                channel_dict = self.analyze_channel_types(still_pending, channel_dict)  # Continue analysis with the remaining files
            else:
                self.logger.info(f"Finalizing results with current analysis status...")
        except Exception as e:
            self.logger.error(f"Error during channel type analysis: {e}")
        finally:
            return channel_dict
