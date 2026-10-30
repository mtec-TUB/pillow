"""
Dataset exploration and analysis tools for PSG data.
"""

import os
import glob
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import List

import numpy as np
from numba import njit
from tqdm import tqdm
from natsort import natsorted

from psg_processing.utils import LoggingManager

# Amplitude resolution threshold (2^7) to decide about channel type (digital vs analog)
DIGITAL_SIGNAL_MAX_UNIQUE_VALUES = 128


@njit(cache=True)
def _is_digital_numba(flat_signal, max_unique):
    """Check whether a flattened signal has at most the allowed unique values.

    Parameters:
        flat_signal: One-dimensional signal values to inspect.
        max_unique: Maximum number of unique values considered digital.

    Returns:
        ``True`` if the signal has no more than ``max_unique`` values;
        otherwise ``False``.
    """
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
    """Determine whether a signal is digital using its unique-value count.

    Parameters:
        signal: Signal array to classify.

    Returns:
        ``True`` for a digital signal and ``False`` for an analog or empty
        signal.
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
        """Discover the readable channels in the assigned PSG file.

        Returns:
            A tuple containing the PSG path and its discovered channel names.
        """
        logging_manager = LoggingManager(console_level=self.console_level)
        logger, _ = logging_manager.create_file_logger(os.path.basename(self.psg_fname))
        channels = self.dataset.get_channels(logger, self.psg_fname)
        return self.psg_fname, channels


class _ChannelTypeTask:
    """Picklable unit of work run in a worker process:
    classify one file's still-digital channels."""

    def __init__(self, dataset, psg_fname, digital_channels, console_level):
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.digital_channels = digital_channels
        self.console_level = console_level

    def run(self):
        """Classify still-digital channels in the assigned PSG file.

        Returns:
            A tuple containing the PSG path and channels found to be analog.
        """
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


class DatasetExplorer:
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
        """Initialize the DatasetExplorer with empty containers and logger."""
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

    def get_files(self, which="both") -> tuple[List[str], list[str]]:
        """Discover PSG signal files and, optionally, annotation files.

        Parameters:
            which: ``"psg"`` to discover only signal files, or ``"both"`` to
                also discover annotation files.

        Returns:
            A tuple ``(psg_filenames, annotation_filenames)``.

        Raises:
            FileNotFoundError: If a required data directory does not exist.
        """
        if not self.logger:
            self.logger = self.logging_manager.create_pipeline_logger()

        # Discover PSG signal files
        if not os.path.exists(self.data_dir):
            self.logger.error("Data directory does not exist: %s", self.data_dir)
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        psg_ext = self.dataset.file_extensions['psg_ext']

        self.logger.info(
            "Searching for signal files: %s", os.path.join(self.data_dir, psg_ext)
        )

        self.psg_fnames = glob.glob(
            os.path.join(self.data_dir, psg_ext), recursive=True
        )
        self.psg_fnames = natsorted(self.psg_fnames)
        self.logger.info("Found %d signal files", len(self.psg_fnames))

        if which == "both":
            # Discover annotation files
            if not os.path.exists(self.ann_dir):
                self.logger.error("Annotation directory does not exist: %s", self.ann_dir)
                raise FileNotFoundError(
                    f"Annotation directory does not exist {self.ann_dir}"
                )

            ann_ext = self.dataset.file_extensions['ann_ext']
            self.logger.info(
                "Searching for annotation files: %s", os.path.join(self.ann_dir, ann_ext)
            )

            self.ann_fnames = glob.glob(
                os.path.join(self.ann_dir, ann_ext), recursive=True
            )

            # Add second annotation extension if provided
            if 'ann_ext2' in self.dataset.file_extensions:
                ann_ext2 = self.dataset.file_extensions['ann_ext2']
                self.logger.info(
                    "Searching for additional annotation files: %s",
                    os.path.join(self.ann_dir, ann_ext2)
                )
                ann_fnames2 = glob.glob(
                    os.path.join(self.ann_dir, ann_ext2), recursive=True
                )
                self.ann_fnames.extend(ann_fnames2)
                self.logger.info("Found %d additional annotation files", len(ann_fnames2))

            self.logger.info("Total annotation files: %d", len(self.ann_fnames))

        # # Validate that we have matching numbers of files
        # if ann_ext != "" and len(self.ann_fnames) != len(self.psg_fnames):
        #     self.logger.warning(
        #         f"Number of PSG files and Annotation files do not match: "
        #         f"({len(self.psg_fnames)}/{len(self.ann_fnames)})"
        #     )

        return self.psg_fnames, self.ann_fnames

    def get_all_channels(self):
        """Discover all available channel names across the PSG files.

        Returns:
            A counter containing each discovered channel name and its frequency
            of occurrence.
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
                                "Skipping file with no readable channels: %s",
                                os.path.basename(psg_fname)
                            )
                        else:
                            self.ch_names.update(channels)
                        pbar.update(1)

        self.logger.info(
            "Discovery complete! Found %d unique channels across all files.", len(self.ch_names)
        )

        return self.ch_names

    def get_channel_type(self):
        """Classify channels as analog or digital.

        Returns:
            A dictionary with ``"analog"`` and ``"digital"`` keys containing
            channel-name lists.
        """
        if not self.psg_fnames:
            self.get_files(which="psg")

        self.logger.info("Analyzing channel types (digital vs analog)...")
        if not self.ch_names:
            channel_names = self.dataset.channel_names
        else:
            channel_names = self.ch_names.keys() # Get just the channel names without frequencies

        self.logger.info("Found %d channels to analyze across %d files",
                         len(channel_names), len(self.psg_fnames))

        channel_types = {"analog": [], "digital": []}

        # Set default to digital and only change to analog
        # if we find evidence of it being analog in any file
        channel_dict = {channel: 'digital' for channel in channel_names}

        channel_dict = self.analyze_channel_types(self.psg_fnames, channel_dict)

        # Convert to final lists
        for channel, ch_type in channel_dict.items():
            channel_types[ch_type].append(channel)

        # Print final summary
        self.logger.info("\nAnalysis complete!")
        self.logger.info("Analog channels: %d", len(channel_types['analog']))
        self.logger.info("Digital channels: %d", len(channel_types['digital']))

        return channel_types

    def analyze_channel_types(self, remaining_fnames, channel_dict):
        """Classify channels by scanning PSG files in worker processes.

        Parameters:
            remaining_fnames: PSG file paths that still need to be scanned.
            channel_dict: Mapping from channel names to their current
                ``"digital"`` or ``"analog"`` classification.

        Returns:
            The updated channel classification mapping.
        """
        console_level = self.logging_manager.console_level
        completed_fnames = set()

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                fname_iter = iter(remaining_fnames)
                tasks = {}

                with tqdm(total=len(remaining_fnames), desc="Analyzing files",
                          unit="file") as outer_bar:
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
                            task = _ChannelTypeTask(self.dataset, psg_fname,
                                                    still_digital, console_level)
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
            self.logger.warning("\n\nKeyboard interrupt detected during channel analysis!")
            # Convert to final lists
            channel_types = {"analog": [], "digital": []}
            for channel, ch_type in channel_dict.items():
                channel_types[ch_type].append(channel)
            print(f"Current status of analysis: {channel_types}")
            # Wait for user input before finalizing
            answer = input("Do you want to continue analysing (y) "
                           "or finalizing the results now (n):")
            if str(answer).lower() == "y":
                still_pending = [f for f in remaining_fnames if f not in completed_fnames]
                # Continue analysis with the remaining files
                channel_dict = self.analyze_channel_types(still_pending, channel_dict)
            else:
                self.logger.info("Finalizing results with current analysis status...")
        except Exception as e:
            self.logger.error("Error during channel type analysis: %s", e)
        return channel_dict
