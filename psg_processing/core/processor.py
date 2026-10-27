"""
Main dataset processor for PSG data preparation.
"""

import logging
import os
import re
import glob
from dataclasses import dataclass
from math import ceil, floor
from datetime import datetime, date, time, timedelta
from pathlib import Path
from decimal import Decimal
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from pyedflib import EdfWriter
import h5py
import numpy as np
from tqdm import tqdm

from psg_processing.utils import LoggingManager, Alignment, SleepStage, SLEEP_STAGES
from psg_processing.core.dataset_explorer import DatasetExplorer
from psg_processing.core.signal_processor import SignalProcessor

logging.captureWarnings(True)

SignalStart = datetime | None
AnnotationStart = datetime | int | float | Decimal | None
LightMarker = datetime | time | int | float | None


@dataclass
class FileData:
    """Metadata shared while processing the channels of one PSG file."""

    psg_fname: str
    ann_fname: str | None
    start_datetime: SignalStart = None
    file_duration: float = 0.0
    ann_start_datetime: AnnotationStart = None
    lights_off: LightMarker = None
    lights_on: LightMarker = None
    front_adjustment_sec: float = 0.0
    labels: np.ndarray | None = None

@dataclass
class ChannelData:
    """Metadata and processed data for a single channel of a PSG file."""

    ch_name_orig: str
    signal: np.ndarray
    sampling_rate: int | float
    unit: str
    file_output_path: str


class DatasetProcessor:
    """
    Main processor for PSG dataset preparation and signal processing.

    This class orchestrates the entire dataset processing pipeline,
    handling file processing, signal cleaning, and output generation.
    """

    def __init__(self, dataset, config):

        self.logging_manager = LoggingManager(console_level=config.logging_level)
        self.dataset = dataset
        self.config = config
        self.pipeline_logger = self.logging_manager.create_pipeline_logger()

    def process_files(self):
        """
        Process all PSG files in the dataset according to the provided configuration.
        """

        try:
            tasks = []

            if self.config.overwrite:
                log_files = glob.glob(os.path.join(self.config.output_dir, "**", "*.log"),
                                      recursive=True)
                for f in log_files:
                    os.remove(f)

            self.pipeline_logger.info("Starting dataset processing")

            # Get all PSG and Annot files names
            psg_fnames, ann_fnames  = DatasetExplorer(
                self.pipeline_logger,
                self.dataset,
                self.config.psg_dir,
                self.config.ann_dir,
                log_level=self.config.logging_level,
                num_workers=self.config.num_workers,
            ).get_files()

            if self.config.use_annot:
                annotation_map = self._build_annot_lookup(psg_fnames, ann_fnames)
                if len(annotation_map) < len(psg_fnames):
                    self.pipeline_logger.info(
                        "%d/%d PSG files have no matching annotation file and will be skipped.",
                        len(psg_fnames) - len(annotation_map),
                        len(psg_fnames),
                    )
                n_files_to_process = len(annotation_map)
            else:
                n_files_to_process = len(psg_fnames)

            # Process psg files in parallel
            max_workers = self.config.num_workers or os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                psg_iter = iter(psg_fnames)

                # Progress Bar
                with tqdm(total=n_files_to_process, desc="Processing files") as pbar:
                    more_files_to_process = True
                    while True:
                        if not more_files_to_process and not tasks:
                            break

                        # Fill up worker tasks until max_workers is reached
                        # or no more files to process.
                        # Do not enqueue all files at once to handle Errors
                        # and stop processing immediatly
                        while len(tasks) < max_workers:
                            try:
                                psg_fname = next(psg_iter)
                            except StopIteration:
                                more_files_to_process = False
                                break

                            # Find matching annotation file
                            if self.config.use_annot:
                                ann_fname = annotation_map.get(psg_fname, None)
                                if ann_fname is None:
                                    # No matching annotation file found, skip this PSG file
                                    continue
                            else:
                                ann_fname = None

                            file_processor = FileProcessor(self.config,
                                                           self.dataset,
                                                           psg_fname, ann_fname)

                            future = executor.submit(file_processor.process_file)
                            tasks.append(future)

                        # Get notified when at least one file is finished and resume this one
                        # Filling with new file will happen in the next loop iteration
                        # -> handle errors before enqueuing new files
                        # Maximal "n_worker" files will still get finished
                        # before stopping pipeline if an error occurs
                        done,_ = wait(tasks, return_when=FIRST_COMPLETED)

                        for future in done:
                            try:
                                future.result()  # Check for exceptions
                                pbar.update(1)
                            except Exception as e:
                                self.pipeline_logger.error("Processing failed: %s", e)
                                executor.shutdown(cancel_futures=True)
                                break
                            finally:
                                tasks.remove(future)
                        else:
                            continue
                        break  # Break outer loop if an exception occurred in the inner loop

            # Finalize processing
            self.pipeline_logger.info("=" * 60)
            self.pipeline_logger.info("DATASET PROCESSING COMPLETED")
        except KeyboardInterrupt:
            if tasks:
                for task in tasks:
                    task.cancel()
                executor.shutdown(cancel_futures=True)
            self.pipeline_logger.info("=" * 60)
            self.pipeline_logger.info("Stopped processing")

    def _build_annot_lookup(self, psg_fnames, ann_fnames):
        annotation_map = {}

        # Build lookup dict by identifier
        ann_lookup = {}

        for ann_fname in ann_fnames:
            ann_base = str(Path(ann_fname).relative_to(self.config.ann_dir))
            ann_id = self.dataset.get_file_identifier(None, ann_base)[1]
            ann_lookup[ann_id] = ann_fname

        for psg_fname in psg_fnames:
            psg_base = str(Path(psg_fname).relative_to(self.config.psg_dir))
            psg_id = self.dataset.get_file_identifier(psg_base, None)[0]
            match_ann_id = ann_lookup.get(psg_id, None)
            if match_ann_id:
                annotation_map[psg_fname] = match_ann_id
            # else:
            #     self.pipeline_logger.warning(
            #         f"No matching annotation file found for PSG: "
            #         f"{Path(psg_fname).relative_to(self.config.psg_dir)}. Skipping file."
            #     )
        return annotation_map


class FileProcessor:
    """Class to process a single PSG file with one or multiple channels
    and save the processed data to file."""

    def __init__(self, config, dataset, psg_fname, ann_fname):
        self.logging_manager = LoggingManager(console_level=config.logging_level)
        self.logger = None
        self.config = config
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.ann_fname = ann_fname

    def process_file(self):
        """Process a single PSG file for all specified channels."""

        try:
            # Start buffering logs for this file
            file_id = Path(self.psg_fname).stem
            self.logger, buffer_handler = self.logging_manager.create_file_logger(
                file_identifier=file_id
            )

            self.logger.info("Signal file: %s",
                             Path(self.psg_fname).relative_to(self.config.psg_dir))
            if self.config.use_annot:
                self.logger.info("Annotation file: %s",
                                 Path(self.ann_fname).relative_to(self.config.ann_dir))

            file_data = FileData(self.psg_fname, self.ann_fname)

            if self.config.output_format == "npz":
                log_paths = {}  # Store log paths for each channel separately
            else:   # edf, hdf5
                file_output_path, log_path = self._setup_file_output()
                # Skip if file already exists and overwrite is False
                if os.path.exists(file_output_path) and not self.config.overwrite:
                    self.logger.info("File already exists: %s, skipping file.", file_output_path)
                    return

            # Get Start datetime and duration of polysomnography data
            file_info = self.dataset.get_file_info(self.logger, self.psg_fname)
            if file_info == {}:
                return

            start_datetime = file_info["start_datetime"]
            if isinstance(start_datetime, datetime):
                start_datetime = start_datetime.replace(tzinfo=None)
            file_data.start_datetime = start_datetime
            self.logger.info("Start datetime: %s", start_datetime)

            if file_info["file_duration"] // self.config.epoch_duration < 1:
                self.logger.warning(
                    "File does not hold at least one epoch, only %.2f seconds, skipping file.",
                    file_info["file_duration"],
                )
                return

            file_data.file_duration = file_info["file_duration"]
            self.logger.info("File duration: %s sec, %.2f h",
                             file_data.file_duration, file_data.file_duration / 3600)

            # Annotations and file-level label preparation
            if self.config.use_annot:
                ann_stage_events, ann_startdatetime, lights_off, lights_on = \
                    self.dataset.ann_parse(self.ann_fname)
                if isinstance(ann_startdatetime, datetime):
                    ann_startdatetime = ann_startdatetime.replace(tzinfo=None)
                file_data.ann_start_datetime = ann_startdatetime
                file_data.lights_off = lights_off
                file_data.lights_on = lights_on

                if ann_stage_events == []:
                    self.logger.warning("No sleep stage annotations found in %s, skipping file.",
                                        os.path.basename(self.ann_fname))
                    return

                # Map dataset-labels to standardized labels and check consistency
                labels = self.dataset.ann_label(self.logger, ann_stage_events,
                                                self.config.epoch_duration)
                # Check how many sleep epochs are in the file
                sleep_mask = np.isin(labels if labels.ndim == 1 else labels[:, 0], SLEEP_STAGES)
                if np.sum(sleep_mask) < self.config.min_sleep_epochs:
                    self.logger.warning(
                        "File contains less sleep epochs than required, skipping file."
                    )
                    return

                # Front alignment (file level)
                start_delay = self._get_start_delay(ann_startdatetime, start_datetime)

                if start_delay != 0:
                    if not self.dataset.has_front_alignment:
                        raise ValueError(
                            f"Signal/annotation start-time mismatch ({start_delay:.1f}s) but "
                            f"dataset {self.dataset.dset_name} has has_front_alignment=False. "
                            f"Please verify."
                        )
                    front_params = self.dataset.compute_front_alignment(
                        self.logger, self.config.alignment, self.config.pad_values,
                        self.config.epoch_duration, start_delay,
                    )
                    labels = self._apply_front_label_adjustment(
                        labels, front_params["label_adjust_front"]
                    )
                    front_adjustment_sec = front_params["signal_adjust_front_sec"]
                    if front_adjustment_sec and isinstance(start_datetime, datetime):
                        start_datetime = start_datetime - timedelta(seconds=front_adjustment_sec)
                        file_data.start_datetime = start_datetime
                        self.logger.info("Adjusted start datetime after front alignment: %s",
                                         start_datetime)
                    file_data.front_adjustment_sec = front_adjustment_sec

                file_data.labels = labels  # adjusted labels, not yet epoch-selected

            # Intersection of available channels in psg file and configured channels to process,
            # plus all channels whose harmonized name is listed in mapped_channels
            file_channels = self.dataset.get_channels(self.logger, self.psg_fname)
            channels = list(
                (set(self.config.channels) & set(file_channels))
                | set(self.dataset.channels_mapped_to(file_channels, self.config.mapped_channels))
            )
            if len(channels) == 0:
                self.logger.warning("No selected channels found in this file. Skipping.")
                return

            # Per-channel processing: raw signal -> filtered/resampled/epoched signal
            all_channel_data: dict[str, ChannelData] = {}
            for channel in sorted(channels):
                # Set current channel for logging
                buffer_handler.set_channel(channel)
                channel_harm = self._harmonize_channel_name(channel)
                if self.config.output_format == "npz":
                    file_output_path, log_path = self._setup_channel_output(channel_harm)
                    log_paths[channel] = log_path

                # Skip if file already exists and overwrite is False
                # (only for output format npz on this level)
                if os.path.exists(file_output_path) and not self.config.overwrite:
                    self.logger.info(" File already exists: %s, skipping channel %s.",
                                     file_output_path, channel)
                    continue

                channel_processor = ChannelProcessor(
                    self.logger, self.config, self.dataset, file_data, channel, file_output_path
                )
                ch_result = channel_processor.process_channel()

                if ch_result is not None:
                    all_channel_data[channel_harm] = ch_result
                self.logger.info("=" * 40)

            if not all_channel_data:
                return

            # File-level post-processing

            # If configured, use lights markers to get selection range
            if self.config.select_epochs == "lights":
                # In some datasets, lights markers are derived on dataset level,
                # in others they were already extracted from the annotation file
                lights_off_ds, lights_on_ds = self.dataset.get_light_times(self.logger,
                                                                           self.psg_fname)
                if lights_off_ds is not None:
                    file_data.lights_off = lights_off_ds
                if lights_on_ds is not None:
                    file_data.lights_on = lights_on_ds
                # Compute epoch indices for lights markers
                loff_epoch, lon_epoch = self._get_lights_epochs(file_data)
            else:
                loff_epoch, lon_epoch = None, None

            # get the first channel's epoch count
            n_signal_epochs = len(next(iter(all_channel_data.values())).signal)
            # epoch count before any end-alignment padding
            n_real_signal_epochs = n_signal_epochs

            # End alignment: reconcile signal epoch count with label count.
            if self.config.use_annot:
                n_label_epochs = len(file_data.labels)
                if n_signal_epochs != n_label_epochs:
                    if not self.dataset.has_end_alignment:
                        raise ValueError(
                            f"Signal ({n_signal_epochs} epochs) and labels "
                            f"({n_label_epochs} epochs) do not match. "
                            f"Dataset {self.dataset.dset_name} has no end alignment "
                            f"defined (has_end_alignment=False). Please verify."
                        )
                    file_data.labels = self._end_align(
                        file_data.labels,
                        all_channel_data,
                        n_signal_epochs,
                        n_label_epochs,
                    )

            # Compute epoch selection based on lights marker
            # and/or MOVE/UNK removal and/or truncate_non_sleep_end
            n_epochs = len(next(iter(all_channel_data.values())).signal)
            labels_sel = file_data.labels if self.config.use_annot else None
            select_idx = self._compute_select_idx(labels_sel, loff_epoch, lon_epoch, n_epochs)
            if select_idx is None:
                return

            # Skip file if selection falls entirely in a padded region with no real signal data.
            if not self._valid_selection_range(file_data, select_idx, n_real_signal_epochs):
                return

            # Apply selection to labels and signals
            if self.config.use_annot:
                file_data.labels = file_data.labels[select_idx]
            for ch_data in all_channel_data.values():
                ch_data.signal = ch_data.signal[select_idx]

            labels_shape = file_data.labels.shape if self.config.use_annot else None
            self.logger.info(
                "Data after selection: signal %s, labels %s",
                next(iter(all_channel_data.values())).signal.shape,
                labels_shape
            )

            # Mirror the selection also in the start datetime that is store in the output file metadata
            if isinstance(start_datetime, datetime):
                file_data.start_datetime = start_datetime + timedelta(
                    seconds=float(select_idx[0] * self.config.epoch_duration)
                )

            self._save_processed_data(file_data, all_channel_data)
            self.logger.info("Successfully processed")

        except Exception as e:
            # Log the exception
            self.logger.error("Error processing file: %s", e, exc_info=True)
            raise  # Re-raise after logging
        finally:

            # Flush buffered logs to console and file(s), even if an exception occurred
            if self.config.output_format == "npz":
                if not log_paths:
                    # Error happened before log paths could be set up,
                    # flush to console for this case
                    buffer_handler.flush_to_console()
                # For npz, flush to each channel's log file with channel filtering
                for channel, log_path in log_paths.items():
                    buffer_handler.flush_to_console_and_file(log_path, channel=channel)
            elif self.config.output_format in ["edf", "hdf5"]:
                # For edf/hdf5, flush all channels' logs to single file
                buffer_handler.flush_to_console_and_file(log_path)

    def _apply_front_label_adjustment(self, labels, label_adjust_front):
        """Crop (negative) or pad (positive) the front of the label array.

        Works for both 1-D (n_epochs,) and 2-D (n_epochs, n_scorers) arrays.
        """
        if label_adjust_front == 0:
            return labels

        pad_val = self.config.pad_values["label"]
        if label_adjust_front < 0:
            return labels[-label_adjust_front:]

        if labels.ndim == 1:
            pad = np.full(label_adjust_front, pad_val, dtype=labels.dtype)
            return np.concatenate([pad, labels])

        pad = np.full((label_adjust_front, labels.shape[1]), pad_val, dtype=labels.dtype)
        return np.vstack([pad, labels])

    def _compute_select_idx(self, labels, lights_off_epoch, lights_on_epoch, n_epochs):
        """Compute the epoch indices to keep based on lights markers and MOVE/UNK removal.

        labels    : 1-D or 2-D label array, or None when use_annot=False.
                    When None, label-dependent modes (integer wake buffer, rm_move/rm_unk,
                    truncate_non_sleep_end) are skipped.
        n_epochs  : total epoch count; derived from labels when labels is not None.
        Uses the first scorer column when labels are 2-D.
        Returns None if no epochs survive.
        """
        if labels is not None:
            # only use first scorer column for seleciton
            labels  = labels if labels.ndim == 1 else labels[:, 0]
            sleep_mask = np.isin(labels, SLEEP_STAGES)
            sleep_idx  = np.where(sleep_mask)[0]
            if self.config.select_edges == "non-wake":
                # Anchor on any non-Wake epoch (incl. Movement/Unknown)
                # matching SleePyCo's reference behavior
                edge_idx = np.where(labels != SleepStage.W)[0]
            else:
                edge_idx = sleep_idx
        else:
            labels  = None
            sleep_idx = np.array([], dtype=int)
            edge_idx  = np.array([], dtype=int)

        start_idx = 0
        end_idx   = n_epochs

        if self.config.select_epochs == "lights":
            if lights_off_epoch is not None:
                start_idx = lights_off_epoch

            if lights_on_epoch is not None:
                if lights_on_epoch <= n_epochs:
                    end_idx = lights_on_epoch
                elif lights_on_epoch == n_epochs + 1:
                    pass  # Lights On fell in the last cropped partial epoch -> keep all
                else:
                    # Maybe padding both signal and annotations until Lights On ??
                    self.logger.warning("Lights On is %d epochs after recording ends "
                                        "(marker is ignored for epoch selection).",
                                        lights_on_epoch - n_epochs)
            else:
                if self.config.truncate_non_sleep_end and len(sleep_idx) > 0:
                    end_idx = sleep_idx[-1] + 1
                    self.logger.info("Removed %d Non-Sleep epochs at end of night.",
                                     n_epochs - end_idx)

        elif isinstance(self.config.select_epochs, int):
            if labels is None:
                self.logger.warning("select_epochs config is an integer but use_annot=False, "
                                    "no labels available to identify sleep stages, "
                                    "keeping all epochs.")
            elif len(edge_idx) > 0:
                n_select  = self.config.select_epochs
                start_idx = max(0, edge_idx[0] - n_select)
                end_idx   = min(n_epochs, edge_idx[-1] + n_select + 1)
                if start_idx > 0 or end_idx != n_epochs:
                    n_crop = n_epochs - (end_idx - start_idx)
                    self.logger.info(
                        "Cropped %d epochs (%.1f min) of extensive wake",
                        n_crop,
                        n_crop * self.config.epoch_duration / 60
                    )

        if labels is not None:
            move_idx = np.where(labels == SleepStage.MOVE)[0] if self.config.rm_move else []
            unk_idx  = np.where(labels == SleepStage.UNK)[0]  if self.config.rm_unk  else []
            if len(move_idx) > 0:
                self.logger.info("Removing %d Movement epochs", len(move_idx))
            if len(unk_idx) > 0:
                self.logger.info("Removing %d Unknown epochs", len(unk_idx))
            remove_idx = np.union1d(move_idx, unk_idx)
        else:
            remove_idx = np.array([], dtype=int)

        # choose all epochs in the range [start_idx, end_idx) except those in remove_idx
        select_idx = np.setdiff1d(np.arange(start_idx, end_idx), remove_idx)

        if len(select_idx) == 0:
            self.logger.warning(
                "No data left after epoch selection. \n start_idx=%d, end_idx=%d, removed=%d",
                start_idx,
                end_idx,
                len(remove_idx)
            )
            return None
        return select_idx

    def _valid_selection_range(self, file_data: FileData, select_idx, n_real_signal_epochs):
        """Check if the selected epochs are within the valid real signal range
        and not entirely in the padded region."""

        # Front padding: front_adjustment_sec > 0 means N epochs of padding were prepended.
        # Only full epochs that are entirely padding are counted (floor division).
        front_adjustment_sec = file_data.front_adjustment_sec
        n_front_pad_epochs = int(front_adjustment_sec // self.config.epoch_duration)
        if (n_front_pad_epochs > 0
                and select_idx[-1] < n_front_pad_epochs):
            self.logger.warning(
                "All selected epochs (%d-%d) fall entirely outside the signal region. "
                "Skipping file.",
                select_idx[0], select_idx[-1]
            )
            return False

        # End padding: signal was extended by end alignment and selection starts beyond real data.
        if (self.config.use_annot
                and select_idx[0] >= n_real_signal_epochs):
            self.logger.warning(
                "All selected epochs (%d-%d) fall entirely outside the signal region. "
                "Skipping file.",
                select_idx[0], select_idx[-1]
            )
            return False
        return True

    def _end_align(self, labels, all_channel_data, n_signal_epochs, n_labels_epochs):
        """Reconcile signal epoch count vs label epoch count using the alignment config.

        All channels share the same n_signal_epochs (same file, same duration).
        Mutates channel signals in-place inside all_channel_data; may also mutate labels.
        Returns the (possibly updated) labels array.
        """
        self.logger.info(
            "End alignment needed: signal has %d epochs, labels have %d epochs.",
            n_signal_epochs, n_labels_epochs
        )
        if n_signal_epochs > n_labels_epochs:
            if self.config.alignment in (Alignment.MATCH_SHORTER.value,
                                         Alignment.MATCH_ANNOT.value):
                self.logger.info("End alignment: cropping signal to %d epochs.", n_labels_epochs)
                # in-place modification
                for ch in all_channel_data.values():
                    ch.signal = ch.signal[:n_labels_epochs]
            elif self.config.alignment in (Alignment.MATCH_LONGER.value,
                                           Alignment.MATCH_SIGNAL.value): #pad labels
                n_pad   = n_signal_epochs - n_labels_epochs
                pad_val = self.config.pad_values["label"]
                self.logger.info("End alignment: padding labels by %d epochs.", n_pad)
                if labels.ndim == 1:
                    labels = np.concatenate([labels, np.full(n_pad, pad_val, dtype=labels.dtype)])
                else:
                    labels = np.vstack([labels, np.full((n_pad, labels.shape[1]), pad_val,
                                                        dtype=labels.dtype)])
        else:
            if self.config.alignment in (Alignment.MATCH_SHORTER.value,
                                         Alignment.MATCH_SIGNAL.value):
                self.logger.info("End alignment: cropping labels to %d epochs.", n_signal_epochs)
                labels = labels[:n_signal_epochs]
            elif self.config.alignment in (Alignment.MATCH_LONGER.value,
                                           Alignment.MATCH_ANNOT.value):  # pad signal
                n_pad   = n_labels_epochs - n_signal_epochs
                pad_val = np.float64(self.config.pad_values["signal"])
                self.logger.info("End alignment: padding signal by %d epochs.", n_pad)
                # in-place modification
                for ch in all_channel_data.values():
                    sig = ch.signal
                    ch.signal = np.vstack(
                        [sig, np.full((n_pad, sig.shape[1]), pad_val)]
                    )
        return labels

    def _harmonize_channel_name(self, channel):
        """Harmonize channel name based on dataset-specific mapping."""
        if self.config.map_channel_names:
            channel_harm = self.dataset.map_channel(channel)
            self.logger.info(" Mapped channel name %s to %s", channel, channel_harm)
            return channel_harm

        return channel

    def _setup_file_output(self):
        """ Setup output for edf and hdf5 output format where all channels of one file
        are stored together and share the same log file. """
        # Create output directory
        # (including subfolders of original data structure if keep_folder_structure is True)
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(Path(self.psg_fname).relative_to(self.config.psg_dir))[0]
        else:
            relative_path = ""

        # Output is generated per PSG file containing all channels,
        # log files are saved per PSG file separately
        output_dir = os.path.join(self.config.output_dir, relative_path)
        log_dir = os.path.join(output_dir, "log_files")
        log_filename = f"{Path(self.psg_fname).stem}.log"

        # Generate output file name
        filename = f"{Path(self.psg_fname).stem}.{self.config.output_format}"
        file_output_path = os.path.join(output_dir, filename)

        # Prepare log file path (will be written after each file is processed)
        log_file_path = os.path.join(log_dir, log_filename)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        return file_output_path, log_file_path

    def _setup_channel_output(self, channel):
        """Setup output for npz output format where each channel is stored separately
        and has its own log file."""

        # Create output directory
        # (including subfolders of original data structure if keep_folder_structure is True)
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(Path(self.psg_fname).relative_to(self.config.psg_dir))[0]
        else:
            relative_path = ""

        # Output is generated per channel and sorted into channel folders
        # with corresponding log file
        # replace slash in folder names to avoid nested output structure
        # replace colon because it is often not accepted in folder names
        channel_clean = re.sub(r"[:/]", "_", channel)
        output_dir = os.path.join(self.config.output_dir, relative_path, channel_clean)
        log_dir = output_dir
        log_filename = channel_clean + ".log"

        # Generate output file name
        filename = f"{Path(self.psg_fname).stem}.{self.config.output_format}"
        file_output_path = os.path.join(output_dir, filename)

        # Prepare log file path (will be written after each file is processed)
        log_file_path = os.path.join(log_dir, log_filename)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        return file_output_path, log_file_path

    def _get_start_delay(self, ann_start_datetime, signal_start_datetime):
        """ Check if annotation and signal start datetime are aligned and calculate the delay
        between annotation and signal start if necessary.
        """
        delay = 0
        if ann_start_datetime is not None:
            # If annotation start datetime is given as a datetime object,
            # compare with signal start datetime and calculate start delay
            if isinstance(ann_start_datetime, datetime) and signal_start_datetime is not None:
                # Strip timezone info before comparison/subtraction
                ann_dt = ann_start_datetime
                sig_dt = signal_start_datetime
                if ann_dt.time() != sig_dt.time():
                    delay = (ann_dt - sig_dt).total_seconds()
                    if abs(delay) >= 24*3600:
                        direction = "after" if delay > 0 else "before"
                        raise ValueError(f"Annotation start datetime {ann_dt} is more than 24h "
                                         f"{direction} signal start datetime {sig_dt}. "
                                         f"Check implementation!")
                    if abs(delay) > 20*3600:
                        delay = delay % (-np.sign(delay)*24*3600)  # Wrap around to next day

            # If annotation start datetime is given as a numeric value,
            # it indicates a delay in seconds or samples (depending on dataset)
            elif isinstance(ann_start_datetime, (int, float, Decimal)):
                delay = ann_start_datetime
            else:
                raise ValueError(f"Unsupported format of annotation start datetime: "
                                 f"{ann_start_datetime}")
            if delay != 0:
                self.logger.info("Start of signal: %s, Start of labels: %s",
                                 signal_start_datetime, ann_start_datetime)

        return delay

    def _get_lights_epochs(self, channel_data: FileData):
        """Get the epochs where Lights Off and Lights On happen based on the configured
        lights marker in annotation or PSG data.
        - The epoch in which Lights Off event happens is return as lights_off_epoch
        - The epoch AFTER which Lights On event happens is returned as lights_on_epoch,
            meaning that the epoch in which Lights On event happens is still included
            in the selected data.

        Args:
            channel_data: File metadata containing the recording start and light markers.

        Returns:
            _type_: epochs of Lights Off and Lights On
        """

        startdatetime = channel_data.start_datetime
        front_adjustment_sec = channel_data.front_adjustment_sec
        lights_off_epoch = 0
        lights_on_epoch  = None

        if channel_data.lights_off is not None:
            lights_off = channel_data.lights_off
            if isinstance(lights_off, (datetime,time)):
                if isinstance(lights_off, time):
                    # if date of marker timestamp is not available,
                    # take date of signal start for easier comparison
                    lights_off = datetime.combine(startdatetime.date(),lights_off)

                lights_off_sec = (lights_off - startdatetime).total_seconds()

                if lights_off_sec != 0:
                    if -3600 < lights_off_sec < 0:
                        # Between -3600 and 0, before signal start but in 1 hour range
                        self.logger.info("Lights Off time %s is before signal start time %s "
                                         "(marker is ignored for epoch selection).",
                                          lights_off, startdatetime.time())
                        # maybe padding with wake epochs until Lights Off time is reached?
                        # For now, just keep all epochs and do not select
                        # (keep lights_off_epoch at 0)
                    else:
                        if lights_off_sec < 0:
                            # Assume Lights Off is after and signal start before midnight
                            lights_off_sec += 24*3600 # add 24h

                        # Round to full epoch
                        lights_off_epoch = self._round_marker_time("lights_off", lights_off_sec,
                                                                    self.config.epoch_duration,
                                                                    lights_off.time())

                        lights_off_delta = timedelta(
                            seconds=lights_off_epoch * self.config.epoch_duration
                        )
                        self.logger.info("Select only epochs after %s "
                                         "(based on Lights Off marker in epoch %d)",
                                         (startdatetime + lights_off_delta).time(),
                                          lights_off_epoch)

                else:
                    self.logger.info("Lights Off time is at the start of the signal, "
                                     "no need of epoch selection based on Lights Off time.")

            elif isinstance(lights_off, (int,float)):
                # Numeric markers are relative to the original signal start.
                lights_off_sec = lights_off + front_adjustment_sec
                if lights_off_sec != 0:
                    # Lights Off probably starts before PSG Data
                    # (1 hour range before signal start)
                    if -3600 < lights_off_sec < 0:
                        self.logger.warning("Lights Off time %s is before signal start time %s "
                                            "(marker is ignored for epoch selection).",
                                             lights_off_sec, startdatetime.time())
                        # front_padding =
                        # maybe padding with wake epochs until Lights Off time is reached?
                        # For now, just keep all epochs and do not select
                    elif lights_off_sec < 0:
                        raise ValueError(f"Lights Off time ({lights_off_sec}) is more than 1 hour "
                                         f"before signal start ({startdatetime.time()})")
                    else:
                        # Round to full epoch
                        lights_off_epoch = self._round_marker_time("lights_off", lights_off_sec,
                                                                    self.config.epoch_duration)
                        self.logger.info("Select only epochs after Lights Off at second %s "
                                         "(epoch %d)",
                                         lights_off_sec, lights_off_epoch)
                else:
                    self.logger.info("Lights Off time is at the start of the signal, "
                                     "no need of epoch selection based on Lights Off time.")
            else:
                raise ValueError(f"Lights Off time has unsupported format: {lights_off}")

        else:
            self.logger.warning("Lights Off time not available, keeping all wake epochs at start.")

        if channel_data.lights_on is not None:
            lights_on = channel_data.lights_on
            # Info: Lights On marker can happen to be after signal end,
            # cannot check this here because signal length is not determined yet
            if isinstance(lights_on, (datetime,time)):
                if isinstance(lights_on, time):
                    # if date of marker timestamp is not available,
                    # take date of signal start for easier comparison
                    lights_on = datetime.combine(startdatetime.date(),lights_on)

                # seconds from recording start until Lights On
                lights_on_sec = (lights_on - startdatetime).total_seconds()

                if lights_on_sec < 0:
                    # Assume Lights On is after and signal start before midnight
                    lights_on_sec += 24 * 3600  # add 24h

                lights_on_epoch = self._round_marker_time("lights_on", lights_on_sec,
                                                          self.config.epoch_duration,
                                                          lights_on.time())
                lights_on_delta = timedelta(seconds=lights_on_epoch*self.config.epoch_duration)
                self.logger.info("Select only epochs before %s "
                                 "(based on Lights On marker in epoch %d)",
                                 (startdatetime + lights_on_delta).time(),
                                   lights_on_epoch)

            elif isinstance(lights_on, (int,float)):
                # Numeric markers are relative to the original signal start.
                lights_on_sec = lights_on + front_adjustment_sec

                if lights_on_sec < 0:
                    raise ValueError(f"Lights On time ({lights_on_sec}) is before signal start "
                                     f"({startdatetime.time()})")

                lights_on_epoch = self._round_marker_time("lights_on", lights_on_sec,
                                                          self.config.epoch_duration)
                self.logger.info("Select only epochs before Lights On at second %s (epoch %d)",
                                 lights_on_sec, lights_on_epoch)
            else:
                raise ValueError(f"Lights On time has unsupported format: {lights_on}")
        else:
            self.logger.warning("Lights On time not available.")

        return lights_off_epoch, lights_on_epoch

    def _round_marker_time(self, marker, marker_sec, epoch_duration, marker_time=None):
        """Helper function to round Lights Off/On time to the next (or previous) epoch
        if it is not exactly at the start/end of an epoch and log this behavior.

        Args:
            marker (_type_): Type of marker which is processed ("lights_off" or "lights_on")
            marker_sec (_type_): timestamp of marker in seconds from signal start
            marker_time (_type_): timestamp of marker as datetime.time object
                for logging purposes (if not available, )
            epoch_duration (_type_): epoch duration in seconds to round the marker time to

        Returns:
            _type_: epoch number corresponding to the rounded marker time
        """
        if marker_time is None:
            marker_time = marker_sec
        if marker_sec % epoch_duration != 0:
            if marker == "lights_off":
                round_marker_epoch = floor(marker_sec / epoch_duration)
                self.logger.info("Lights Off time %s is floored to %dsec from signal start.",
                                 marker_time, round_marker_epoch * epoch_duration)
            elif marker == "lights_on":
                round_marker_epoch = ceil(marker_sec / epoch_duration)
                self.logger.info("Lights On time %s is ceiled to  %dsec from signal start.",
                                 marker_time, round_marker_epoch * epoch_duration)
        else:
            round_marker_epoch = marker_sec / epoch_duration
        return int(round_marker_epoch)

    def _save_processed_data(
        self,
        file_data: FileData,
        all_channel_data: dict[str, ChannelData],
    ):
        """Save processed data to file for npz, edf, hdf5 formats."""
        output_format = self.config.output_format
        labels = file_data.labels
        channels_sorted = sorted(all_channel_data.keys())
        channel_data = [all_channel_data[ch] for ch in channels_sorted]
        n_epochs = [len(channel.signal) for channel in channel_data]
        if len(np.unique(n_epochs)) != 1:
            self.logger.error("All channels must have the same number of epochs after processing "
                              "to be saved together, but got different number of epochs: %s",
                              n_epochs)
            raise ValueError("All channels must have the same number of epochs after processing "
                             "to be saved together, but got different number of epochs: "
                             f"{n_epochs}")

        # check if there is a signal with only NaN values
        if any(np.isnan(channel.signal).all() for channel in channel_data):
            self.logger.warning("One or more channels contain only NaN values after processing. "
                                "Skipping this file.")
            return

        if output_format == "npz":
            for channel_name, channel in zip(channels_sorted, channel_data):
                save_dict = {
                    "x": channel.signal,
                    "fs": channel.sampling_rate,
                    "ch_label": channel_name,
                    "ch_label_orig": channel.ch_name_orig,
                    "file_duration": len(channel.signal) * self.config.epoch_duration,
                    "epoch_duration": self.config.epoch_duration,
                    "n_epochs": len(channel.signal),  # after cleaning
                }

                # Write Annotations
                if self.config.use_annot:
                    # Handle multiple scorers
                    if labels.ndim == 1:
                        save_dict["y"] = labels
                    elif labels.ndim == 2:
                        save_dict["y"] = labels[:, 0]
                        save_dict["y2"] = labels[:, 1]
                save_dict["unit"] = channel.unit or "n/a"
                np.savez(channel.file_output_path, **save_dict)

        elif output_format == "edf":
            file_output_path = channel_data[0].file_output_path
            with EdfWriter(file_output_path, n_channels=len(channel_data)) as edf_writer:
                for i, (channel_name, channel) in enumerate(zip(channels_sorted, channel_data)):
                    signal = channel.signal.flatten()
                    scale = 10**3  # to get 3 decimals for physical min and max
                    phys_min = floor(np.nanmin(signal) * scale) / scale
                    phys_max = ceil(np.nanmax(signal) * scale) / scale
                    if phys_min == phys_max:
                        phys_min -= 1.0
                        phys_max += 1.0
                    channel_info = {
                        "label": channel_name,
                        "dimension": channel.unit or "n/a",
                        "sample_frequency": float(channel.sampling_rate),
                        "physical_min": phys_min,
                        "physical_max": phys_max,
                        "digital_min": -32768,
                        "digital_max": 32767,
                        "transducer": "",
                        "prefilter": "",
                    }
                    edf_writer.setSignalHeader(i, channel_info)
                if isinstance(file_data.start_datetime, datetime):
                    if file_data.start_datetime.date() > date(1985, 1, 1):
                        edf_start_datetime = file_data.start_datetime
                    else:
                        edf_start_datetime = datetime.combine(
                            date(1985, 1, 1), file_data.start_datetime.time()
                        )
                else:
                    edf_start_datetime = datetime(1985, 1, 1, 0, 0, 0)

                # Write samples
                edf_writer.setStartdatetime(edf_start_datetime)
                edf_writer.writeSamples([channel.signal.flatten() for channel in channel_data])

                # Write annotations
                if self.config.use_annot:
                    duration = self.config.epoch_duration
                    for epoch_idx, label in enumerate(labels):
                        # one value per scorer, comma-separated (e.g. "0" or "0,6")
                        edf_writer.writeAnnotation(
                            onset_in_seconds=epoch_idx * duration,
                            duration_in_seconds=duration,
                            description=",".join(str(int(l)) for l in np.atleast_1d(label)),
                        )

        elif output_format == "hdf5":
            file_output_path = channel_data[0].file_output_path
            with h5py.File(file_output_path, "w") as h5f:
                # Metadata
                h5f.attrs["epoch_duration"] = self.config.epoch_duration
                h5f.attrs["file_duration"] = (len(channel_data[0].signal)
                                              * self.config.epoch_duration)
                h5f.attrs["n_epochs"] = len(channel_data[0].signal)
                grp_signals = h5f.create_group("signals")
                for channel_name, channel in zip(channels_sorted, channel_data):
                    signal = channel.signal.flatten()

                    # if channel_name came from h5 originally, keep only the last part after slash
                    if ("h5" in self.dataset.file_extensions["psg_ext"]
                            and "/" in channel_name):
                        group_name = channel_name.split("/")[-1]
                    else:
                        group_name = channel_name
                    ch_grp = grp_signals.create_group(group_name)
                    # Signal
                    ch_grp.create_dataset("data", data=signal, compression="gzip", shuffle=True)
                    ch_grp.attrs["ch_label"] = channel_name
                    ch_grp.attrs["ch_label_orig"] = channel.ch_name_orig
                    ch_grp.attrs["unit"] = channel.unit or "n/a"
                    ch_grp.attrs["sampling_rate"] = channel.sampling_rate

                # Annotations
                if self.config.use_annot:
                    labels = np.asarray(labels)
                    if labels.ndim == 1:
                        h5f.create_dataset("y", data=labels, compression="gzip")
                    elif labels.ndim == 2:
                        h5f.create_dataset("y", data=labels[:, 0], compression="gzip")
                        h5f.create_dataset("y2", data=labels[:, 1], compression="gzip")


class ChannelProcessor:
    """Process a single channel: raw signal-> epoched signal array.

    - Read raw signal and sampling rate from file.
    - Resample / filter / clip (all fs-dependent).
    - Apply the per-channel raw-sample front offset (front_adjustment_sec × fs)
      AFTER resampling and filtering to avoid boundary artefacts.
    - Reshape into epochs.
    """

    def __init__(
        self,
        logger: logging.Logger,
        config: ProcessorConfig,
        dataset,
        file_data: FileData,
        channel: str,
        file_output_path: str,
    ):
        self.logger  = logger
        self.config  = config
        self.dataset = dataset
        self.file_data = file_data
        self.channel = channel
        self.file_output_path = file_output_path

    def process_channel(self):
        """Read and process one channel and return its processed data.

        Returns None if the channel cannot be processed (missing signal, etc.).
        """

        # Read raw signal
        psg_data = self.dataset.get_signal_data(self.logger,
                                                self.file_data.psg_fname,
                                                self.channel)
        if psg_data == {}:
            return None
        signal = psg_data["signal"].astype(np.float64)
        sampling_rate = psg_data["sampling_rate"]
        unit = psg_data.get("unit", "n/a")
        del psg_data

        self.logger.info("Channel %s has %d samples (%.2f Hz).",
                         self.channel, len(signal), sampling_rate)

        # Rescale amplitude
        target_unit = self._get_target_unit(self.channel, self.dataset.channel_groups)
        if target_unit is not None:
            signal, unit = self._rescale_signal(signal, unit, target_unit, self.channel)

        # Resample / filter / clip (fs-dependent, must come before raw-sample crop)
        if self.config.resample is not None or self.config.filter:
            signal_processor = SignalProcessor(
                self.logger, signal, self.channel, self.config, self.dataset.channel_types
            )
            if self.config.resample is not None:
                signal_processor.resample_signal(sampling_rate, self.config.resample)
                sampling_rate = self.config.resample
            if self.config.filter:
                signal_processor.filter_signal(sampling_rate, self.dataset.channel_groups)
            signal_processor.clip_signal()
            signal = signal_processor.signal

        # Apply raw-sample front offset
        if self.file_data.front_adjustment_sec != 0.0:
            signal = self._apply_partial_epoch_offset(
                signal, sampling_rate, self.file_data.front_adjustment_sec
            )

        # Reshape into epochs
        n_epoch_samples = self.config.epoch_duration * sampling_rate
        if not n_epoch_samples.is_integer():
            raise ValueError(
                f"Epoch duration {self.config.epoch_duration} sec with sampling rate {sampling_rate} Hz "
                "does not yield an integer number of samples per epoch."
            )
        n_epoch_samples = int(n_epoch_samples)

        n_epochs, remainder = divmod(len(signal), n_epoch_samples)
        if n_epochs < 1:
            self.logger.warning(
                "Channel %s: fewer than one full epoch of signal available, skipping.",
                self.channel
            )
            return None
        if remainder > 0:
            self.logger.info("Signal cropped to full epochs (%.2f sec removed).",
                             remainder / sampling_rate)

        signal = signal[:n_epochs * n_epoch_samples].reshape(n_epochs, -1)
        return ChannelData(
            ch_name_orig=self.channel,
            signal=signal,
            sampling_rate=sampling_rate,
            unit=unit,
            file_output_path=self.file_output_path,
        )

    # Multiplier to convert a value expressed in the given voltage unit to Volts
    VOLT_FACTORS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}

    def _get_target_unit(self, ch_name, channel_groups):
        """Get the configured target rescale unit for a given channel
        using the centralized group mapping."""
        for group_name, channels in channel_groups.items():
            if ch_name in channels:
                return self.config.rescale_unit.get(group_name)

        return self.config.rescale_unit.get("default")

    def _normalize_volt_unit(self, unit):
        """Normalize a unit string to canonical 'V'/'mV'/'uV' if it represents a voltage unit
        (case-insensitive; treats the unicode micro sign µ/μ the same as ascii 'u'), else None."""
        if not isinstance(unit, str):
            return None
        normalized = unit.strip().replace("µ", "u").replace("μ", "u").lower()
        return {"v": "V", "mv": "mV", "uv": "uV"}.get(normalized)

    def _rescale_signal(self, signal, unit, target_unit, ch_name):
        """Rescale signal amplitude to target_unit if both the channel's current unit
        and target_unit are voltage units (V/mV/uV); otherwise warn and keep the signal as is."""
        norm_unit = self._normalize_volt_unit(unit)
        norm_target = self._normalize_volt_unit(target_unit)
        if norm_unit is None or norm_target is None:
            self.logger.warning(
                "Channel %s: cannot rescale from unit '%s' to '%s' "
                "(both units must be one of V/mV/uV), keeping original unit.",
                ch_name, unit, target_unit
            )
            return signal, unit

        factor = self.VOLT_FACTORS[norm_unit] / self.VOLT_FACTORS[norm_target]
        if factor != 1.0:
            signal = signal * factor
            self.logger.info("Channel %s: rescaled amplitude from %s to %s.",
                             ch_name, unit, target_unit)

        return signal, target_unit

    def _apply_partial_epoch_offset(self, signal, fs, front_adjustment_sec):
        """Padding (positive) or crop (negative) at the raw signal front.

        Called after resampling/filtering so that filter edge effects are not
        introduced by padding, and the resampled fs is used for sample counting.
        """
        n_samples = int(abs(front_adjustment_sec) * fs)
        if front_adjustment_sec < 0:

            return signal[n_samples:]

        pad_val = np.float64(self.config.pad_values["signal"])
        return np.concatenate([np.full(n_samples, pad_val), signal])
