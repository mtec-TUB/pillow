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

from datasets.base import BaseDataset
from psg_processing.utils import LoggingManager, Alignment, SleepStage, SLEEP_STAGES, ProcessorConfig
from psg_processing.core.dataset_explorer import DatasetExplorer
from psg_processing.core.signal_processor import SignalProcessor

logging.captureWarnings(True)

@dataclass
class FileData:
    """Metadata shared while processing the channels of one PSG file."""

    psg_fname: str
    ann_fname: str | None
    start_datetime: datetime | None = None
    ann_start_datetime: datetime | int | float | Decimal | None = None
    file_duration: float = 0.0
    lights_off: datetime | time | int | float | None = None
    lights_on: datetime | time | int | float | None = None
    front_adjustment_sec: float = 0.0   # front alignment of the signal: < 0 padded, > 0 cropped
    selection_offset_sec: float = 0.0   # time from aligned start to the first selected epoch
    labels: np.ndarray | None = None

    @property
    def output_start_datetime(self) -> datetime | None:
        """Return the start datetime represented by the saved data.

        Returns:
            The original start datetime shifted by front alignment and epoch
            selection, or the unchanged value when it is not a datetime.
        """
        if not isinstance(self.start_datetime, datetime):
            return self.start_datetime
        return (self.start_datetime
                + timedelta(seconds=self.front_adjustment_sec)
                + timedelta(seconds=self.selection_offset_sec))


@dataclass
class ChannelData:
    """Metadata and data for a single channel of a PSG file.
    """

    ch_name_orig: str
    signal: np.ndarray
    sampling_rate: int | float
    unit: str
    file_output_path: str | None = None  # set at file level, after channel processing


class DatasetProcessor:
    """
    Main processor for PSG dataset preparation.

    This class orchestrates the dataset processing pipeline,
    by distributing the the processing of individual PSG files to worker processes.

    Proceesing stops if any file fails, and the error is logged for debugging.
    """

    def __init__(self, dataset, config):

        self.logging_manager = LoggingManager(console_level=config.logging_level)
        self.dataset = dataset
        self.config = config
        self.pipeline_logger = self.logging_manager.create_pipeline_logger()

    def process_files(self):
        """Process all PSG files according to the configured pipeline.

        Returns:
            None.
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

                            # Submit the file processing task to the executor
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

    def _build_annot_lookup(self, psg_fnames: list[str], ann_fnames: list[str]) -> dict[str, str]:
        """Match PSG files to annotation files by dataset-specific identifiers.

        Parameters:
            psg_fnames: PSG file paths to match.
            ann_fnames: Annotation file paths to match.

        Returns:
            A mapping from PSG paths to matching annotation paths.
        """
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

    def __init__(self, config: ProcessorConfig, dataset: BaseDataset, psg_fname: str, ann_fname: str):
        self.logging_manager = LoggingManager(console_level=config.logging_level)
        self.logger = None
        self.config = config
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.ann_fname = ann_fname

    def process_file(self):
        """Process all configured channels from one PSG file.

        Returns:
            None. Output files are written according to the configuration.
        """

        try:
            # Start buffering logs for this file with file-specific identifier
            file_id = Path(self.psg_fname).stem
            self.logger, buffer_handler = self.logging_manager.create_file_logger(file_id)

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


            # Check if there are dataset-level markers for Lights On/Off events.
            lights_off_ds, lights_on_ds = self.dataset.get_light_times(
                self.logger, self.psg_fname
            )
            file_data.lights_off = lights_off_ds
            file_data.lights_on = lights_on_ds

            # Annotations preparation
            if self.config.use_annot:
                labels = self._prepare_labels(file_data)
                if labels is not None:
                    file_data.labels = labels
                else:
                    return

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

            # Per-channel processing: raw signal -> filtered/resampled signal
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

                # Extract signal, resample and filter
                channel_processor = ChannelProcessor(
                    self.logger, self.config, self.dataset, self.psg_fname, channel
                )
                ch_result = channel_processor.process_channel()

                if ch_result is not None:
                    ch_result.file_output_path = file_output_path
                    all_channel_data[channel_harm] = ch_result
                self.logger.info("=" * 40)

            if not all_channel_data:
                return

            # Check if all channels have the same epoch count
            epoch_counts = [len(ch_data.signal) // self._n_epoch_samples(ch_data.sampling_rate)
                for _, ch_data in all_channel_data.items()]

            if len(set(epoch_counts)) > 1:
                raise ValueError(
                    "Channels have different epoch counts: "
                    + ", ".join(f"{name}: {count}" for name, count in zip(all_channel_data, epoch_counts))
                )

            ## Align signals and labels

            # Front alignment
            if file_data.labels is not None:
                self._front_align(file_data, all_channel_data)
                # Update epoch counts after front alignment
                epoch_counts = [len(ch_data.signal) // self._n_epoch_samples(ch_data.sampling_rate)
                    for _, ch_data in all_channel_data.items()]
                if any(count < 1 for count in epoch_counts):
                    self.logger.warning("No epochs left after front alignment, skipping file.")
                    return

            # Trim to full epochs (drops channels with less than one full epoch)
            self._trim_to_full_epochs(all_channel_data)
            if not all_channel_data:
                return

            # Memorize the number of epochs before end alignment (for epoch selection)
            n_signal_epochs = epoch_counts[0]

            # End alignment
            if file_data.labels is not None:
                self._end_align(file_data, all_channel_data, n_signal_epochs)

            ## Reshape channel signals into epochs
            self._signal_epoching(all_channel_data)

            ## Compute and apply selection of epochs to keep
            ret = self._epoch_selection(file_data, all_channel_data, n_signal_epochs)
            if ret is None:
                return

            ## Save processed data to file(s)
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

    def _setup_file_output(self):
        """Prepare paths and directories for a file-level output (edf/hdf5).

        Returns:
            A tuple containing the output file path and log file path.
        """
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

    def _prepare_labels(self, file_data: FileData) -> np.ndarray | None:
        """Parse and standardize annotations and lights marker for a PSG file.

        Parameters:
            file_data: File-level metadata to update with annotation markers.

        Returns:
            Standardized labels with shape ``(n_epochs,)`` or ``(n_epochs, 2)``,
            or ``None`` when the file has no usable annotations.
        """

        ann_events, ann_startdatetime, lights_off, lights_on = self.dataset.ann_parse(self.ann_fname)
        if isinstance(ann_startdatetime, datetime):
            ann_startdatetime = ann_startdatetime.replace(tzinfo=None)
        file_data.ann_start_datetime = ann_startdatetime
        # if Light markers are not yet stored from dataset-level, use the ones from annotation
        if file_data.lights_off is None:
            file_data.lights_off = lights_off
        if file_data.lights_on is None:
            file_data.lights_on = lights_on

        if ann_events == []:
            self.logger.warning("No sleep stage annotations found in %s, skipping file.",
                                os.path.basename(self.ann_fname))
            return None

        # Map dataset-labels to standardized labels and check consistency
        labels = self.dataset.ann_label(self.logger, ann_events,
                                        self.config.epoch_duration)
        # Check how many sleep epochs are in the file (first scorer)
        sleep_mask = np.isin(labels if labels.ndim == 1 else labels[:, 0], SLEEP_STAGES)
        if np.sum(sleep_mask) < self.config.min_sleep_epochs:
            self.logger.warning(
                "File contains less sleep epochs than required, skipping file."
            )
            return None

        return labels

    def _marker_to_sec(self, marker, start_datetime, name):
        """Resolve a light marker relative to the original signal start.

        Parameters:
            marker: Numeric, date-time, time-only, or ``None`` marker value.
            start_datetime: Original signal start datetime.
            name: Marker name used in validation errors.

        Returns:
            A ``(offset_seconds, time_of_day)`` tuple, or ``None`` when no
            marker is provided.
        """
        if marker is None:
            return None
        if isinstance(marker, (datetime, time)):
            if isinstance(marker, time):
                marker = datetime.combine(start_datetime.date(), marker)
            return (marker - start_datetime).total_seconds(), marker.time()
        if isinstance(marker, (int, float)):
            return marker, None
        raise ValueError(f"{name} time has unsupported format: {marker}")

    def _harmonize_channel_name(self, channel):
        """Map a channel name to its configured harmonized name.

        Parameters:
            channel: Original channel name.

        Returns:
            The harmonized channel name, or the original name if mapping is
            disabled.
        """
        if self.config.map_channel_names:
            channel_harm = self.dataset.map_channel(channel)
            self.logger.info(" Mapped channel name %s to %s", channel, channel_harm)
            return channel_harm

        return channel

    def _setup_channel_output(self, channel):
        """Prepare paths and directories for a channel-level output (npz).

        Parameters:
            channel: Channel name used to construct the output directory.

        Returns:
            A tuple containing the output file path and log file path.
        """

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

    def _n_epoch_samples(self, sampling_rate):
        """Calculate the number of samples in one epoch.

        Parameters:
            sampling_rate: Signal sampling rate in hertz.

        Returns:
            Number of samples in one configured epoch.
        """
        n_epoch_samples = self.config.epoch_duration * sampling_rate
        if not n_epoch_samples.is_integer():
            raise ValueError(
                f"Epoch duration {self.config.epoch_duration} sec with sampling rate {sampling_rate} Hz "
                "does not yield an integer number of samples per epoch."
            )
        return int(n_epoch_samples)

    def _front_align(self, file_data: FileData, all_channel_data: dict[str, ChannelData]):
        """Align signal and annotation starts at file level in place.

        Parameters:
            file_data: File metadata and labels to update.
            all_channel_data: Channel data whose signals may be cropped or
                padded.

        Raises:
            ValueError: If the dataset does not allow the required alignment.

        Returns:
            None.
        """

        # Offset between signal start and label start (> 0: labels start later)
        start_diff_sec = self._get_start_diff(file_data.ann_start_datetime, file_data.start_datetime)

        if start_diff_sec != 0:
            if not self.dataset.has_front_alignment:
                self.logger.error(
                    "Signal/annotation start-time mismatch (%.1f sec) but "
                    "dataset %s has has_front_alignment=False. Please verify.",
                    start_diff_sec, self.dataset.dset_name
                )
                raise ValueError(
                    f"Signal/annotation start-time mismatch ({start_diff_sec:.1f}s) but "
                    f"dataset {self.dataset.dset_name} has has_front_alignment=False. "
                    f"Please verify."
                )

            label_adjust_front, signal_adjust_front_sec = self._compute_front_alignment(
                start_diff_sec)
            file_data.labels = self._apply_front_alignment_labels(file_data.labels, label_adjust_front)
            file_data.front_adjustment_sec = signal_adjust_front_sec

            if signal_adjust_front_sec != 0.0:
                for ch_data in all_channel_data.values():
                    ch_data.signal = self._apply_front_alignment_signal(
                        ch_data.signal, ch_data.sampling_rate, signal_adjust_front_sec
                    )
                if isinstance(file_data.start_datetime, datetime):
                    self.logger.info("Start datetime after front alignment: %s",
                                     file_data.output_start_datetime)

    def _get_start_diff(self, ann_start_datetime, signal_start_datetime):
        """Calculate the annotation start offset relative to the signal start.

        Parameters:
            ann_start_datetime: Annotation start as a datetime or numeric
                offset.
            signal_start_datetime: Signal start datetime.

        Returns:
            Difference in seconds; positive means annotations start later.
        """
        diff = 0
        if ann_start_datetime is not None:
            # If annotation start datetime is given as a datetime object,
            # compare with signal start datetime and calculate start diff
            if isinstance(ann_start_datetime, datetime) and signal_start_datetime is not None:
                # Strip timezone info before comparison/subtraction
                ann_dt = ann_start_datetime
                sig_dt = signal_start_datetime
                if ann_dt.time() != sig_dt.time():
                    diff = (ann_dt - sig_dt).total_seconds()
                    if abs(diff) >= 24*3600:
                        direction = "after" if diff > 0 else "before"
                        raise ValueError(f"Annotation start datetime {ann_dt} is more than 24h "
                                         f"{direction} signal start datetime {sig_dt}. "
                                         f"Check implementation!")
                    if abs(diff) > 20*3600:
                        diff = diff % (-np.sign(diff)*24*3600)  # Wrap around to next day

            # If annotation start datetime is given as a numeric value,
            # it indicates a diff in seconds or samples (depending on dataset)
            elif isinstance(ann_start_datetime, (int, float, Decimal)):
                diff = ann_start_datetime
            else:
                raise ValueError(f"Unsupported format of annotation start datetime: "
                                 f"{ann_start_datetime}")
            if diff != 0:
                self.logger.info("Start of signal: %s, Start of labels: %s",
                                 signal_start_datetime, ann_start_datetime)

        return diff

    def _compute_front_alignment(self, diff_sec):
        """Compute front-alignment changes without modifying arrays.

        Parameters:
            diff_sec: Annotation start offset in seconds. Positive means
                annotations start later.

        Returns:
            A tuple ``(label_adjust_front, signal_adjust_front_sec)``. The
            label adjustment is positive for cropping and negative for padding;
            the signal adjustment is positive for cropping and negative for
            padding.
        """
        alignment = self.config.alignment
        epoch_duration = self.config.epoch_duration
        pad_values = self.config.pad_values

        label_adjust_front      = 0
        signal_adjust_front_sec = 0.0

        if diff_sec < 0:
            # Signal started after annotations
            self.logger.info("Signal started %.2f sec after label start", abs(diff_sec))
            if alignment in (Alignment.MATCH_SHORTER.value, Alignment.MATCH_SIGNAL.value):
                # labels can only be cropped in full epochs
                n_crop, remainder = divmod(abs(diff_sec), epoch_duration)
                label_adjust_front = int(n_crop)
                if remainder != 0:
                    # crop both labels and signals to the next epoch they both share
                    label_adjust_front += 1
                    signal_adjust_front_sec = (epoch_duration - remainder)  # positive = crop
                self.logger.info("Labels will be shortened at the front by %d epochs to match", label_adjust_front)
            elif alignment in (Alignment.MATCH_LONGER.value, Alignment.MATCH_ANNOT.value):
                signal_adjust_front_sec = diff_sec   # negative = pad
        else:
            self.logger.info("Labeling started %.2f min after signal start", diff_sec / 60)
            if alignment in (Alignment.MATCH_SHORTER.value, Alignment.MATCH_ANNOT.value):
                signal_adjust_front_sec = diff_sec # positive = crop
            elif alignment in (Alignment.MATCH_LONGER.value, Alignment.MATCH_SIGNAL.value):
                # labels can only be padded in full epochs
                n_pad, remainder = divmod(diff_sec, epoch_duration)
                label_adjust_front = -int(n_pad)    # negative = pad
                self.logger.info("Labels will be padded at the front with %d epochs of value: %s to match",
                            n_pad, pad_values['label'])
                if remainder != 0:
                    # crop both labels and signals to the next epoch they both share
                    signal_adjust_front_sec = remainder # positive = crop

        if signal_adjust_front_sec > 0:
            self.logger.info("Signal front crop: removing %.4f sec from signal start to align with labels.",
                        signal_adjust_front_sec)
        elif signal_adjust_front_sec < 0:
            self.logger.info("Signal front pad: prepending %.4f sec \
                        of value %s to signal start to align with labels.",
                        abs(signal_adjust_front_sec),
                        pad_values['signal'])

        return label_adjust_front, signal_adjust_front_sec

    def _apply_front_alignment_labels(self, labels, label_adjust_front):
        """Crop or pad labels at the front.

        Parameters:
            labels: Label array to adjust.
            label_adjust_front: Number of epochs to crop when positive or pad
                when negative.

        Returns:
            The adjusted label array.
        """
        if label_adjust_front == 0:
            return labels

        if label_adjust_front > 0:
            return labels[label_adjust_front:]

        pad_val = self.config.pad_values["label"]
        # Handles both 1-D and 2-D arrays (multiple scorer)
        pad = np.full((abs(label_adjust_front), *labels.shape[1:]), pad_val, dtype=labels.dtype)
        return np.concatenate([pad, labels])

    def _apply_front_alignment_signal(self, signal, fs, front_adjustment_sec):
        """Crop or pad a signal at the front.

        Parameters:
            signal: Continuous signal array to adjust.
            fs: Signal sampling rate in hertz.
            front_adjustment_sec: Seconds to crop when positive or pad when
                negative.

        Returns:
            The adjusted signal array.
        """
        n_samples = int(abs(front_adjustment_sec) * fs)
        if front_adjustment_sec > 0:
            # crop
            return signal[n_samples:]

        # pad
        pad_val = np.float64(self.config.pad_values["signal"])
        return np.concatenate([np.full(n_samples, pad_val), signal])

    def _trim_to_full_epochs(self, all_channel_data: dict[str, ChannelData]):
        """Trim each signal to complete epochs.

        Parameters:
            all_channel_data: Channel data to trim in place.

        Returns:
            None. Channels shorter than one epoch are removed.
        """
        for ch_name in list(all_channel_data):
            ch_data = all_channel_data[ch_name]
            n_epoch_samples = self._n_epoch_samples(ch_data.sampling_rate)

            n_epochs, remainder = divmod(len(ch_data.signal), n_epoch_samples)
            if n_epochs < 1:
                self.logger.warning(
                    "Channel %s: fewer than one full epoch of signal available, skipping.",
                    ch_data.ch_name_orig
                )
                del all_channel_data[ch_name]
                continue
            if remainder > 0:
                self.logger.info("Signal cropped to full epochs (%.2f sec removed).",
                                 remainder / ch_data.sampling_rate)
                ch_data.signal = ch_data.signal[:n_epochs * n_epoch_samples]

    def _end_align(self, file_data: FileData, all_channel_data: dict[str, ChannelData], n_signal_epochs: int):
        """Reconcile signal and label epoch counts.

        Parameters:
            file_data: File metadata containing labels to update.
            all_channel_data: Channel data whose signals may be cropped or
                padded.
            n_signal_epochs: Number of complete signal epochs before alignment.

        Raises:
            ValueError: If the dataset does not allow end alignment.

        Returns:
            None.
        """
        labels = file_data.labels
        n_label_epochs = len(labels)

        if n_signal_epochs != n_label_epochs:
            if not self.dataset.has_end_alignment:
                raise ValueError(
                    f"Signal ({n_signal_epochs} epochs) and labels "
                    f"({n_label_epochs} epochs) do not match. "
                    f"Dataset {self.dataset.dset_name} has no end alignment "
                    f"defined (has_end_alignment=False). Please verify."
                )

            self.logger.info(
                "End alignment needed: signal has %d epochs, labels have %d epochs.",
                n_signal_epochs, n_label_epochs
            )
            if n_signal_epochs > n_label_epochs:
                if self.config.alignment in (Alignment.MATCH_SHORTER.value,
                                            Alignment.MATCH_ANNOT.value):
                    self.logger.info("End alignment: cropping signal to %d epochs.", n_label_epochs)
                    # in-place modification
                    for ch in all_channel_data.values():
                        n_epoch_samples = self._n_epoch_samples(ch.sampling_rate)
                        ch.signal = ch.signal[:n_label_epochs * n_epoch_samples]
                elif self.config.alignment in (Alignment.MATCH_LONGER.value,
                                            Alignment.MATCH_SIGNAL.value): #pad labels
                    n_pad   = n_signal_epochs - n_label_epochs
                    pad_val = self.config.pad_values["label"]
                    self.logger.info("End alignment: padding labels by %d epochs.", n_pad)
                    pad = np.full((n_pad, *labels.shape[1:]), pad_val, dtype=labels.dtype)
                    labels = np.concatenate([labels, pad])
            else:
                if self.config.alignment in (Alignment.MATCH_SHORTER.value,
                                            Alignment.MATCH_SIGNAL.value):
                    self.logger.info("End alignment: cropping labels to %d epochs.", n_signal_epochs)
                    labels = labels[:n_signal_epochs]
                elif self.config.alignment in (Alignment.MATCH_LONGER.value,
                                            Alignment.MATCH_ANNOT.value):  # pad signal
                    n_pad   = n_label_epochs - n_signal_epochs
                    pad_val = np.float64(self.config.pad_values["signal"])
                    self.logger.info("End alignment: padding signal by %d epochs.", n_pad)
                    # in-place modification
                    for ch in all_channel_data.values():
                        n_epoch_samples = self._n_epoch_samples(ch.sampling_rate)
                        ch.signal = np.concatenate(
                            [ch.signal, np.full(n_pad * n_epoch_samples, pad_val)]
                        )
        file_data.labels = labels

    def _signal_epoching(self, all_channel_data: dict[str, ChannelData]):
        """Reshape continuous signals into epochs.

        Parameters:
            all_channel_data: Channel data to reshape in place.

        Returns:
            None.
        """
        for ch_data in all_channel_data.values():
            n_epoch_samples = self._n_epoch_samples(ch_data.sampling_rate)
            ch_data.signal = ch_data.signal.reshape(-1, n_epoch_samples)

    def _epoch_selection(self,
                         file_data: FileData,
                         all_channel_data: dict[str, ChannelData],
                         end_idx_orig_signal: int):
        """Select and apply the epochs to retain in the output.

        Parameters:
            file_data: File metadata and labels to update.
            all_channel_data: Channel data to select in place.
            end_idx_orig_signal: Epoch idx after which the original signal ends
                                 and padding begins.

        Returns:
            The selected epoch indices, or ``None`` when no valid epochs remain.
        """

        # If configured, use lights markers to get selection range
        if self.config.select_epochs == "lights":
            # Compute epoch indices for lights markers
            loff_epoch, lon_epoch = self._get_lights_epochs(file_data)
        else:
            loff_epoch, lon_epoch = None, None

        n_epochs = len(next(iter(all_channel_data.values())).signal)
        labels_sel = file_data.labels
        select_idx = self._compute_select_idx(labels_sel, loff_epoch, lon_epoch, n_epochs)
        if select_idx is None:
            return None

        # Skip file if selection falls entirely in a padded region with no real signal data.
        if not self._valid_selection_range(file_data, select_idx, end_idx_orig_signal):
            return None

        # Apply selection to labels and signals
        if file_data.labels is not None:
            file_data.labels = file_data.labels[select_idx]
            labels_shape = file_data.labels.shape

        for ch_data in all_channel_data.values():
            ch_data.signal = ch_data.signal[select_idx]

        self.logger.info(
            "Data after selection: signal %s, labels %s",
            next(iter(all_channel_data.values())).signal.shape,
            labels_shape if file_data.labels is not None else "None"
        )

        # Remember the selection offset for the start datetime in the output file metadata
        file_data.selection_offset_sec = float(select_idx[0] * self.config.epoch_duration)
        return select_idx

    def _get_lights_epochs(self, file_data: FileData):
        """Calculate epoch bounds from Lights Off and Lights On markers.

        Parameters:
            file_data: File metadata containing markers and front alignment.

        Returns:
            A tuple ``(lights_off_epoch, lights_on_epoch)``.
            Both are ``None`` when unavailable.
        """

        # front_adjustment_sec < 0: signal was padded at the front, > 0: signal was cropped
        front_adjustment_sec = file_data.front_adjustment_sec
        lights_off_epoch = None
        lights_on_epoch  = None

        off = self._marker_to_sec(file_data.lights_off, file_data.start_datetime, "Lights Off")
        if off is not None:
            off_sec, off_clock = off   # off_clock is None for numeric markers
            lights_off_sec = off_sec - front_adjustment_sec   # relative to aligned start
            if lights_off_sec == 0:
                self.logger.info("Lights Off time is at the start of the signal, "
                                 "no need of epoch selection based on Lights Off time.")
            elif -3600 < lights_off_sec < 0:
                # Between -3600 and 0, before signal start but in 1 hour range
                self.logger.warning("Lights Off time %s is before signal start "
                    "(marker is ignored for epoch selection).",
                    off_clock if off_clock is not None else lights_off_sec)
                # maybe padding with wake epochs until Lights Off time is reached?
                # For now, just keep all epochs and do not select
            else:
                if lights_off_sec <= -3600:
                    if off_clock is not None:
                        # Clock time without date: assume Lights Off is after midnight
                        lights_off_sec += 24*3600 # add 24h
                    else:
                        raise ValueError(f"Lights Off time ({lights_off_sec}) is at least 1 hour "
                                         f"before signal start")

                # Get epoch idx of Lights Off event
                lights_off_epoch = self._marker_epoch_idx("Lights Off", lights_off_sec, off_clock)
                self.logger.info("Select only epochs after Lights Off at second %s (epoch %d)",
                                 lights_off_sec, lights_off_epoch)
        else:
            self.logger.warning("Lights Off time not available, keeping all wake epochs at start.")

        on = self._marker_to_sec(file_data.lights_on, file_data.start_datetime, "Lights On")
        if on is not None:
            # A Lights On marker after the end of the signal is detected in _compute_select_idx,
            # where the number of epochs after alignment is known.
            on_sec, on_clock = on   # on_clock is None for numeric markers
            lights_on_sec = on_sec - front_adjustment_sec   # relative to aligned start
            if lights_on_sec < 0:
                if on_clock is not None:
                    # Clock time without date: assume Lights On is after midnight
                    lights_on_sec += 24 * 3600  # add 24h
                else:
                    raise ValueError(f"Lights On time ({lights_on_sec}) is before signal start")

            # Get epoch idx of Lights On event
            lights_on_epoch = self._marker_epoch_idx("Lights On", lights_on_sec, on_clock)
            if lights_on_sec % self.config.epoch_duration == 0:
                # Marker exactly on an epoch boundary: the epoch starting at Lights On is not included
                lights_on_epoch -= 1
            self.logger.info("Select only epochs before Lights On at second %s (epoch %d)",
                             lights_on_sec, lights_on_epoch)
        else:
            self.logger.warning("Lights On time not available.")

        return lights_off_epoch, lights_on_epoch

    def _marker_epoch_idx(self, marker, marker_sec, marker_time=None):
        """Convert a marker offset into its containing epoch index.

        Parameters:
            marker: Marker name used in log messages.
            marker_sec: Marker offset in seconds from signal start.
            marker_time: Optional marker time used in log messages.

        Returns:
            The zero-based epoch index containing the marker.
        """
        epoch_duration = self.config.epoch_duration
        if marker_time is None:
            marker_time = marker_sec
        if marker_sec % epoch_duration != 0:
            marker_epoch_idx = floor(marker_sec / epoch_duration)
            self.logger.info("%s event at %s corresponds to epoch %d (%dsec from signal start).",
                             marker, marker_time, marker_epoch_idx, marker_epoch_idx * epoch_duration)
        else:
            marker_epoch_idx = marker_sec / epoch_duration
        return int(marker_epoch_idx)

    def _compute_select_idx(self, labels, lights_off_epoch, lights_on_epoch, n_epochs):
        """Compute the epoch indices to retain.

        Parameters:
            labels: One- or two-dimensional labels, or ``None`` without
                annotations.
            lights_off_epoch: Optional first epoch to retain.
            lights_on_epoch: Optional last epoch boundary to retain.
            n_epochs: Total number of available epochs.

        Returns:
            A sorted array of selected epoch indices, or ``None`` if none
            survive.
        """
        if labels is not None:
            # only use first scorer column for selection
            labels  = labels if labels.ndim == 1 else labels[:, 0]
            sleep_mask = np.isin(labels, SLEEP_STAGES)
            sleep_indices  = np.where(sleep_mask)[0]
            if self.config.select_edges == "non-wake":
                # Anchor on any non-Wake epoch (incl. Movement/Unknown)
                # matching SleePyCo's reference behavior
                edges = np.where(labels != SleepStage.W)[0]
            else:
                edges = sleep_indices
            if edges.size > 0:
                edges = edges[[0, -1]]
            else:
                edges = np.array([], dtype=int)
        else:
            sleep_indices = np.array([], dtype=int)
            edges  = np.array([], dtype=int)

        start_idx = 0
        end_idx   = n_epochs

        if self.config.select_epochs == "lights":
            if lights_off_epoch is not None:
                start_idx = lights_off_epoch

            if lights_on_epoch is not None:
                if lights_on_epoch < n_epochs:
                    end_idx = lights_on_epoch + 1
                elif lights_on_epoch == n_epochs:
                    pass
                else:
                    # Maybe padding both signal and annotations until Lights On ??
                    self.logger.warning("Lights On is %d epochs after recording ends "
                                        "(marker is ignored for epoch selection).",
                                        lights_on_epoch - n_epochs + 1)
            else:
                if self.config.truncate_non_sleep_end and len(sleep_indices) > 0:
                    end_idx = sleep_indices[-1] + 1
                    self.logger.info("Removed %d Non-Sleep epochs at end of night.",
                                     n_epochs - end_idx)

        elif isinstance(self.config.select_epochs, int):
            if labels is None:
                self.logger.warning("select_epochs config is an integer but use_annot=False, "
                                    "no labels available to identify sleep stages, "
                                    "keeping all epochs.")
            elif len(edges) > 0:
                n_select  = self.config.select_epochs
                start_idx = max(0, edges[0] - n_select)
                end_idx   = min(n_epochs, edges[-1] + n_select + 1)
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

    def _valid_selection_range(self, file_data: FileData, select_idx, end_idx_orig_signal):
        """Check whether selected epochs contain original signal data.

        Parameters:
            file_data: File metadata containing alignment information.
            select_idx: Selected epoch indices.
            end_idx_orig_signal: Epoch idx after which the original signal ends
                                 and padding begins.

        Returns:
            ``True`` if the selection is valid; otherwise ``False``.
        """

        # Front padding: front_adjustment_sec < 0 means signal was padded with N epochs
        # Only full epochs that are entirely padding are counted (floor division).
        if file_data.front_adjustment_sec < 0:
            start_idx_orig_signal = int(abs(file_data.front_adjustment_sec) // self.config.epoch_duration)
            if (select_idx[-1] < start_idx_orig_signal):
                self.logger.warning(
                    "All selected epochs (%d-%d) fall entirely outside the signal region. "
                    "Skipping file.",
                    select_idx[0], select_idx[-1]
                )
                return False

        # End padding: signal was extended by end alignment and selection may start beyond real data.
        if (self.config.use_annot
                and select_idx[0] >= end_idx_orig_signal):
            self.logger.warning(
                "All selected epochs (%d-%d) fall entirely outside the signal region. "
                "Skipping file.",
                select_idx[0], select_idx[-1]
            )
            return False
        return True

    def _save_processed_data(
        self,
        file_data: FileData,
        all_channel_data: dict[str, ChannelData],
    ):
        """Save processed channels and labels in the configured format.

        Parameters:
            file_data: File metadata, labels, and output timing.
            all_channel_data: Processed channel data to save.

        Returns:
            None.
        """
        output_format = self.config.output_format
        labels = file_data.labels
        channels_sorted = sorted(all_channel_data.keys())
        channel_data = [all_channel_data[ch] for ch in channels_sorted]

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
                output_start = file_data.output_start_datetime
                if isinstance(output_start, datetime):
                    if output_start.date() > date(1985, 1, 1):
                        edf_start_datetime = output_start
                    else:
                        edf_start_datetime = datetime.combine(
                            date(1985, 1, 1), output_start.time()
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
    """Process a single channel: raw signal -> continuous (1-D) processed signal.

    - Read raw signal and sampling rate from file.
    - Rescale amplitude.
    - Resample / filter / clip (all fs-dependent).

    Alignment with the labels and epoching are done at file level
    (see FileProcessor._align_and_epoch).
    """

    def __init__(
        self,
        logger: logging.Logger,
        config: ProcessorConfig,
        dataset: BaseDataset,

        psg_fname: str,
        channel: str,
    ):
        self.logger  = logger
        self.config  = config
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.channel = channel

    def process_channel(self):
        """Read and process one channel into a continuous signal.

        Returns:
            Processed channel data, or ``None`` when the channel is unavailable
            or contains only invalid samples.
        """

        # Read raw signal
        psg_data = self.dataset.get_signal_data(self.logger,
                                                self.psg_fname,
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
        signal, unit = self._rescale_signal(signal, unit)

        # Resample / filter / clip (fs-dependent)
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

        # check if the channel contains only NaN values
        if np.isnan(signal).all():
            self.logger.warning("Channel %s contains only NaN values after processing. "
                                "Skipping.", self.channel)
            return None

        return ChannelData(
            ch_name_orig=self.channel,
            signal=signal,
            sampling_rate=sampling_rate,
            unit=unit
        )

    def _normalize_volt_unit(self, unit):
        """Normalize a voltage unit string.

        Parameters:
            unit: Unit string to normalize.

        Returns:
            Canonical ``"V"``, ``"mV"``, or ``"uV"``, or ``None`` for a
            non-voltage unit.
        """
        if not isinstance(unit, str):
            return None
        normalized = unit.strip().replace("µ", "u").replace("μ", "u").lower()
        return {"v": "V", "mv": "mV", "uv": "uV"}.get(normalized)

    # Multiplier to convert a value expressed in the given voltage unit to Volts
    VOLT_FACTORS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}

    def _rescale_signal(self, signal, unit):
        """Rescale signal amplitude to the configured channel unit.

        Parameters:
            signal: Signal values to rescale.
            unit: Current signal unit.

        Returns:
            A ``(signal, unit)`` tuple containing the possibly rescaled signal
            and its resulting unit.
        """
        target_unit = self.config.rescale_unit.get("default")

        for group_name, channels in self.dataset.channel_groups.items():
            if self.channel in channels:
                target_unit = self.config.rescale_unit.get(group_name)
                break

        if target_unit is not None:
            norm_unit = self._normalize_volt_unit(unit)
            norm_target = self._normalize_volt_unit(target_unit)
            if norm_unit is None or norm_target is None:
                self.logger.warning(
                    "Channel %s: cannot rescale from unit '%s' to '%s' "
                    "(both units must be one of V/mV/uV), keeping original unit.",
                    self.channel, unit, target_unit
                )
                return signal, unit

            factor = self.VOLT_FACTORS[norm_unit] / self.VOLT_FACTORS[norm_target]
            if factor != 1.0:
                signal = signal * factor
                self.logger.info("Channel %s: rescaled amplitude from %s to %s.",
                                self.channel, unit, target_unit)

            return signal, target_unit

        return signal, unit