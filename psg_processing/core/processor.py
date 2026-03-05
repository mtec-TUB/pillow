"""
Main dataset processor for PSG data preparation.
"""

import logging
import os
import re
import copy
from math import ceil, floor
from datetime import datetime, date
from pyedflib import EdfWriter
import h5py
import glob
from pathlib import Path
from decimal import Decimal
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from ..utils import LoggingManager
from .dataset_explorer import Dataset_Explorer
from .signal_processor import SignalProcessor

logging.captureWarnings(True)


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

    # Final sleep stage labels mapping (did not yet find a better place for this, maybe in config?)
    # labels will appear like this in output 
    STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6}

    def process_files(self):

        try:
            if self.config.overwrite:
                log_files = glob.glob(os.path.join(self.config.output_dir, "**", "*.log"), recursive=True)
                for f in log_files:
                    os.remove(f)

            self.pipeline_logger = self.logging_manager.create_pipeline_logger()

            self.pipeline_logger.info("Starting dataset processing")

            # Get all PSG and Annot files names
            psg_fnames, ann_fnames  = Dataset_Explorer(
                self.pipeline_logger,
                self.dataset,
                self.config.data_dir,
                self.config.ann_dir,
                log_level=self.config.logging_level,
            ).get_files()

            if self.config.use_annot:
                annotation_map = self._build_annot_lookup(psg_fnames, ann_fnames)
                if len(annotation_map) < len(psg_fnames):
                    self.pipeline_logger.info(f"{len(psg_fnames)-len(annotation_map)}/{len(psg_fnames)} PSG files have no matching annotation file and will be skipped.")
                n_files_to_process = len(annotation_map)
            else:
                n_files_to_process = len(psg_fnames)

            # Process psg files in parallel
            max_workers = self.config.num_workers or os.cpu_count() or 1
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = []
                psg_iter = iter(psg_fnames)

                # Progress Bar
                with tqdm(total=n_files_to_process, desc="Processing files") as pbar:
                    more_files_to_process = True
                    while True:
                        if not more_files_to_process and not tasks:
                           break
                        
                        # Fill up worker tasks until max_workers is reached or no more files to process
                        # Do not enqueue all files at once to handle Errors and stop processing immediatly
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
                                    continue # No matching annotation file found, skip this PSG file
                            else:
                                ann_fname = None

                            future = executor.submit(self._process_file, psg_fname, ann_fname)
                            tasks.append(future)
                            
                        # Get notified when at least one file is finished and resume this one
                        # Filling with new file will happen in the next loop iteration -> handle errors before enqueuing new files
                        # Maximal "n_worker" files will still get finished before stopping pipeline if an error occurs
                        # Ensures that capacity of workers is used as good as possible, even if some files take much longer to process than others
                        done,_ = wait(tasks, return_when=FIRST_COMPLETED)   

                        for future in done:
                            try:
                                future.result()  # Check for exceptions
                                pbar.update(1)
                            except Exception as e:
                                self.pipeline_logger.error(f"Processing failed: {e}")
                                executor.shutdown(cancel_futures=True)
                                raise
                            tasks.remove(future)
                    
            # Finalize processing
            self.pipeline_logger.info("=" * 60)
            self.pipeline_logger.info("DATASET PROCESSING COMPLETED")
        except KeyboardInterrupt:
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
            psg_base = str(Path(psg_fname).relative_to(self.config.data_dir))
            psg_id = self.dataset.get_file_identifier(psg_base, None)[0]
            match_ann_id = ann_lookup.get(psg_id, None)
            if match_ann_id:
                annotation_map[psg_fname] = match_ann_id
            # else:
            #     self.pipeline_logger.warning(
            #         f"No matching annotation file found for PSG: "
            #         f"{Path(psg_fname).relative_to(self.config.data_dir)}. Skipping file."
            #     )
        return annotation_map

    def _process_file(self, psg_fname, ann_fname):
        """Process a single PSG file for all specified channels."""

        # Initialize signal data dictionary which holds all necessary info for processing and saving
        file_data = {}
        file_data["psg_fname"] = psg_fname
        file_data["ann_fname"] = ann_fname

        # Start buffering logs for this file
        file_id = Path(psg_fname).stem
        file_logger, buffer_handler = self.logging_manager.create_file_logger(
            file_identifier=file_id,
        )
        log_paths = {}  # Store log paths for each channel/file

        try:
            if self.config.use_annot:
                # Parse annotations first (is same for all channels)
                ann_stage_events, ann_Startdatetime = self.dataset.ann_parse(ann_fname)
                file_data["ann_start_datetime"] = ann_Startdatetime

                if ann_stage_events == []:
                    file_logger.warning(
                        f"No sleep stage annotations found in {Path(ann_fname).relative_to(self.config.ann_dir)}, skipping file."
                    )
                    return
                
                # Map dataset-labels to standardized labels and check consistency
                labels = self.dataset.ann_label(
                    file_logger, ann_stage_events, self.config.epoch_duration
                )

                sleep_idx = np.where(
                    (labels != self.STAGE_DICT["W"])
                    & (labels != self.STAGE_DICT["MOVE"])
                    & (labels != self.STAGE_DICT["UNK"])
                )[0]

                if len(sleep_idx) <= self.config.min_sleep_epochs:
                    file_logger.warning(
                        "File contains less sleep epochs than required. Skipping"
                    )
                    return
                
                file_data["labels"] = labels
            else:
                file_data["labels"] = None
                file_data["ann_start_datetime"] = None          


            # List channels to process for this file (based on config and available channels in this file)
            channels = list(
                set(self.config.channels)
                & set(self.dataset.get_channels(file_logger, psg_fname))
            )
            if len(channels)==0:
                file_logger.info("No selected channels found in this file. Skipping.")
                return 

            # Process each channel
            all_file_data = []
            for channel in channels:
                # Set current channel for logging
                buffer_handler.set_channel(channel)

                channel_harm = self._harmonize_channel_name(file_logger,channel)
                file_data["ch_name"] = channel_harm
                
                file_output_path, log_path = self._setup_output(
                    channel_harm,
                    file_data["psg_fname"],
                )
                log_paths[channel] = log_path

                # Skip if file already exists and overwrite is False
                if not self.config.overwrite and os.path.exists(file_output_path):
                    if self.config.output_format == "npz":
                        file_logger.info(f"File already exists: {file_output_path}, skipping channel {channel} and continuing with next channel if available.")
                        continue
                    else:
                        file_logger.info(f"File already exists: {file_output_path}, skipping file and continuing with next file.")
                        # other channels are ignored as well because all channels in one file (that already exists)
                        break

                proc_file_data = self._process_channel(
                    file_logger,
                    copy.deepcopy(file_data),
                    channel,
                )

                # Save processed data if output is npz, else save it together with all other channels to save after
                if proc_file_data is not None:
                    if self.config.output_format == "npz":
                        self._save_processed_data(proc_file_data, file_output_path)
                        file_logger.info(f"Successfully saved: {file_output_path}")
                    elif self.config.output_format in ["edf", "hdf5"]:
                        # edf or h5 output
                        all_file_data.append(proc_file_data)
                
                file_logger.info("=" * 40)


            # Save multi-channel data if edf or hdf5 output specified (npz already saved per channel)
            if all_file_data and self.config.output_format in ["edf", "hdf5"]:
                self._save_processed_data(all_file_data, file_output_path)
                file_logger.info(f"Successfully saved: {file_output_path}")
        
        except Exception as e:
            # Log the exception
            file_logger.error(f"Error processing file {Path(psg_fname).name}: {str(e)}", exc_info=True)
            raise  # Re-raise after logging        
        finally:
            # Flush buffered logs to console and file(s), even if an exception occurred
            if self.config.output_format == "npz":
                # For npz, flush to each channel's log file with channel filtering
                for channel, log_path in log_paths.items():
                    buffer_handler.flush_to_console_and_file(log_path, channel=channel)
            elif self.config.output_format in ["edf", "hdf5"] and log_paths:
                # For edf/hdf5, flush all channels' logs to single file
                first_log_path = next(iter(log_paths.values()))
                buffer_handler.flush_to_console_and_file(first_log_path)

    def _process_channel(
        self,
        logger,
        file_data,
        channel,
    ):
        """Process a single channel from a single file."""

        logger.info(f"Signal file: {Path(file_data['psg_fname']).relative_to(self.config.data_dir)}")
        if self.config.use_annot:
            logger.info(f"Annotation file: {Path(file_data['ann_fname']).relative_to(self.config.ann_dir)}")

        file_data["ch_name_orig"] = channel
        logger.info(f"Channel selected: {file_data['ch_name_orig']}")

        # Extract data from psg file and add to file_data dictionary
        psg_data = self.dataset.get_signal_data(logger, file_data["psg_fname"], file_data["ch_name_orig"])
        file_data.update(psg_data)
        del psg_data  # free memory

        logger.info(f"Select channel samples: {len(file_data['signal'])}")
        logger.info(f"File duration: {file_data['file_duration']} sec, {file_data['file_duration']/3600:.2f} h")
        logger.info(f"Start datetime: {file_data['start_datetime']}")

        # Process the signal (resample, filter, clean)
        signal = file_data["signal"].astype(np.float64)
        labels = file_data["labels"]
        fs = file_data["sampling_rate"]

        if self.config.resample is not None or self.config.filter:
            signal_processor = SignalProcessor(logger, signal, file_data["ch_name_orig"], self.config.filter_freq, self.dataset.channel_types)

            if self.config.resample is not None:
                # Resample signal
                signal_processor.resample_signal(
                    fs,
                    self.config.resample,
                )
                fs = self.config.resample

            if self.config.filter:
                # Filter signal according to AASM
                signal_processor.filter_signal(
                    fs,
                    self.dataset.channel_groups,
                )
            
            signal_processor.clip_signal()
            signal = signal_processor.signal

        if self.config.use_annot:
            # Check if annotations and signal start at the same timestamp and pad/crop if necessary and configured
            signal, labels = self._handle_start_datetime(logger,signal, labels, fs, file_data["ann_start_datetime"], file_data["start_datetime"])

        # Reshape into epochs
        n_epoch_samples = self.config.epoch_duration * fs
        if not n_epoch_samples.is_integer():
            raise ValueError(
                f"Epoch duration {self.config.epoch_duration} sec with sampling rate {fs} Hz "
                "does not yield an integer number of samples per epoch."
            )
        n_epoch_samples = int(n_epoch_samples)

        # Check signal length (at least one epoch required)
        n_epochs, remainder = divmod(len(signal), n_epoch_samples)
        if n_epochs < 1:
            logger.warning(
                f"Channel does not hold at least one epoch, only {len(signal)} samples"
            )
            return

        logger.info(f"Seconds in unfilled (cropped) epoch: {remainder / fs:.2f} sec")

        signal_epoched = signal[: n_epochs * n_epoch_samples].reshape(n_epochs, -1)

        if self.config.use_annot:
            if len(signal_epoched) != len(labels):
                # Align end of signal and labels (some datasets have different length of signal and annotation data)
                signal_epoched, labels = self.dataset.align_end(
                    logger,
                    self.config.alignment,
                    self.config.pad_values,
                    file_data["psg_fname"],
                    file_data["ann_fname"],
                    signal_epoched,
                    labels,
                )

            assert len(signal_epoched) == len(labels), \
            f"Length mismatch: signal ({os.path.basename(file_data['psg_fname'])})={len(signal_epoched)}, labels({os.path.basename(file_data['ann_fname'])})={len(labels)} TODO: adapt alignment function"

            # Clean signal data based on annotations (e.g. remove movement/unknown epochs, select sleep periods)
            signal_epoched, labels = self._clean_signal(logger, signal_epoched, labels)

        # update signal, labels and sampling_rate after the processing
        file_data.update({"signal": signal_epoched, "labels": labels, "sampling_rate": fs})



        return file_data

    def _harmonize_channel_name(self, logger, channel):
        """Harmonize channel name based on dataset-specific mapping."""
        if self.config.map_channel_names:
            channel_harm = self.dataset.map_channel(channel)
            logger.info(f"Mapped channel name {channel} to {channel_harm}")
            return channel_harm
        else:
            return channel

    def _setup_output(self, channel, psg_fname):
        """Setup generic output parts shared across formats."""

        # Create output directory (including subfolders of original data structure if keep_folder_structure is True)
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(
                Path(psg_fname).relative_to(self.config.data_dir)
            )[0]
        else:
            relative_path = ""

        if self.config.output_format == "npz":
            # Output is generated per channel and sorted into channel folders with corresponding log file
            # replace slash in folder names to avoid nester output structure and colon because it is often not accepted in folder names
            channel_clean = re.sub(r"[:/]", "_", channel)
            output_dir = os.path.join(
                self.config.output_dir,
                relative_path,
                channel_clean,
            )
            log_dir = output_dir
            log_filename = channel_clean + ".log"

        elif self.config.output_format in ["edf", "hdf5"]:
            # Output is generated per PSG file containing all channels, log files are saved per PSG file separately
            output_dir = os.path.join(
                self.config.output_dir, relative_path
            )
            log_dir = os.path.join(output_dir, "log_files")
            log_filename = f"{Path(psg_fname).stem}.log"

        # Generate output file name
        filename = f"{Path(psg_fname).stem}.{self.config.output_format}"
        file_output_path = os.path.join(output_dir, filename)

        # Prepare log file path (will be written later after all channels processed)
        log_file_path = os.path.join(log_dir, log_filename)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        return file_output_path, log_file_path

    def _handle_start_datetime(self, logger,signal, labels, fs, ann_start_datetime, signal_start_datetime):
        # If annotation holds a start datetime, check if alignment is needed
        if ann_start_datetime != None:
            delay = 0

            # If annotation start datetime is given as a datetime object, compare with signal start datetime and calculate start delay
            if isinstance(ann_start_datetime, datetime) and signal_start_datetime is not None:
                if ann_start_datetime.time() != signal_start_datetime.time():
                    delay = (
                        ann_start_datetime - signal_start_datetime
                    ).total_seconds()
                    
            # If annotation start datetime is given as a numeric value, it indicates a delay in seconds or samples (depending on dataset)
            elif isinstance(
                ann_start_datetime, (int, float, Decimal)
            ):  # ann_Startdatetime can be in seconds or samples (depends on dataset)
                delay = ann_start_datetime

            if delay != 0:
                logger.info(
                    f"Start of signal: {signal_start_datetime}, Start of labels: {ann_start_datetime}"
                )
                # Align the start of signals and labels based on configuration
                signal, labels = self.dataset.align_front(
                    logger,
                    self.config.alignment,
                    self.config.pad_values,
                    self.config.epoch_duration,
                    delay,
                    signal,
                    labels,
                    fs,
                )
        return signal, labels

    def _clean_signal(self, logger, signal_epoched, labels):
        """
        Clean signal by removing movement/unknown epochs and selecting sleep periods.

        Args:
            signal_epoched: Signal epochs array
            labels: Stage labels array

        Returns:
            Tuple of (cleaned_signals, cleaned_labels) or (None, None) if no sleep detected
        """
        logger.info(
            f"Starting signal cleaning - Input shape: signal_epoched={signal_epoched.shape}, labels={labels.shape}"
        )

        # Remove movement and unknown epochs if configured
        if self.config.rm_move:
            move_idx = np.where(labels == self.STAGE_DICT["MOVE"])[0]
        else:
            move_idx = []
        if len(move_idx) > 0:
            logger.info(f"  Removing Movement epochs: {len(move_idx)}")

        if self.config.rm_unk:
            unk_idx = np.where(labels == self.STAGE_DICT["UNK"])[0]
        else:
            unk_idx = []
        if len(unk_idx) > 0:
            logger.info(f"  Removing Unknown epochs: {len(unk_idx)}")

        remove_idx = np.union1d(move_idx, unk_idx)

        sleep_idx = np.where(
            (labels != self.STAGE_DICT["W"])
            & (labels != self.STAGE_DICT["MOVE"])
            & (labels != self.STAGE_DICT["UNK"])
        )[0]

        if self.config.n_wake_epochs == "all":
            start_idx = 0
            end_idx = len(labels) - 1
        else:
            # Remove extensive wake epochs at start and end as given in config
            n_wake_epochs = self.config.n_wake_epochs
            start_idx = max(0, sleep_idx[0] - n_wake_epochs)
            end_idx = min(len(labels) - 1, sleep_idx[-1] + n_wake_epochs)

            if start_idx + (len(signal_epoched) - end_idx) - 1 > 0:
                logger.info(
                    f"  Outside {int(self.config.n_wake_epochs)/2}min wake epochs: {start_idx + (len(signal_epoched)-end_idx)-1}"
                )

        select_idx = np.setdiff1d(np.arange(start_idx, end_idx + 1), remove_idx)

        logger.info(
            f"  Total epochs to remove: {len(signal_epoched) - len(select_idx)}"
        )

        signal_epoched = signal_epoched[select_idx]
        labels = labels[select_idx]

        logger.info(f"Data after cleaning: {signal_epoched.shape}, {labels.shape}")

        return signal_epoched, labels

    def _save_processed_data(self, file_data, file_output_path):
        """Save processed data to file."""

        if self.config.output_format == "npz":
            save_dict = {
                "x": file_data["signal"],
                "fs": file_data["sampling_rate"],
                "ch_label": file_data["ch_name"],
                "ch_label_orig": file_data["ch_name_orig"],
                "file_duration": len(file_data["signal"])
                * self.config.epoch_duration,
                "epoch_duration": self.config.epoch_duration,
                "n_epochs": len(file_data["signal"]),  # after cleaning
            }

            # Write Annotations
            if self.config.use_annot:
                # Handle multiple scorers
                labels = file_data["labels"]
                if labels.ndim == 1:
                    save_dict["y"] = labels
                elif labels.ndim == 2:
                    save_dict["y"] = labels[:, 0]
                    save_dict["y2"] = labels[:, 1]

            # Write unit
            save_dict["unit"] = file_data["unit"] if "unit" in file_data else "a.u."

            np.savez(file_output_path, **save_dict)

        elif self.config.output_format == "edf":
            all_file_data = file_data

            with EdfWriter(
                file_output_path, n_channels=len(all_file_data)
            ) as edf_writer:
                # Set signal headers
                for i, file_data in enumerate(all_file_data):
                    signal = file_data["signal"].flatten()
                    scale = 10**3  # to get 3 decimals for physical min and max
                    phys_min = floor(np.nanmin(signal) * scale) / scale
                    phys_max = ceil(np.nanmax(signal) * scale) / scale
                    if phys_min == phys_max:
                        phys_min -= 1.0
                        phys_max += 1.0
                    channel_info = {
                        "label": file_data["ch_name"],
                        "dimension": (
                            file_data["unit"] if "unit" in file_data else "a.u."
                        ),
                        "sample_frequency": float(file_data["sampling_rate"]),
                        "physical_min": phys_min,
                        "physical_max": phys_max,
                        "digital_min": -32768,
                        "digital_max": 32767,
                        "transducer": "",
                        "prefilter": "",
                    }
                    edf_writer.setSignalHeader(i, channel_info)

                # Set start datetime (take first channel's datetime)
                if isinstance(all_file_data[0]["start_datetime"], datetime):
                    if all_file_data[0]["start_datetime"].date() > date(1985, 1, 1):
                        Startdatetime = all_file_data[0]["start_datetime"]
                    else:
                        Startdatetime = datetime.combine(
                            date(1985, 1, 1),
                            all_file_data[0]["start_datetime"].time(),
                        )
                else:
                    Startdatetime = datetime(1985, 1, 1, 0, 0, 0)

                edf_writer.setStartdatetime(Startdatetime)

                # Write signal samples
                all_signals = [
                    file_data["signal"].flatten() for file_data in all_file_data
                ]
                edf_writer.writeSamples(all_signals)

                # Write annotations
                if self.config.use_annot:
                    all_labels = np.array(
                        [file_data["labels"] for file_data in all_file_data]
                    )  # to check if they all have the same dimension
                    duration = self.config.epoch_duration
                    for epoch_idx, label in enumerate(
                        all_labels[0]
                    ):  # annotations are the same for all channels, take first one
                        edf_writer.writeAnnotation(
                            onset_in_seconds=epoch_idx * duration,
                            duration_in_seconds=duration,
                            description=str(label),
                        )

        elif self.config.output_format == "hdf5":
            all_file_data = file_data

            with h5py.File(file_output_path, "w") as h5f:
                # Metadata
                h5f.attrs["epoch_duration"] = self.config.epoch_duration
                h5f.attrs["file_duration"] = (
                    len(file_data[0]["signal"]) * self.config.epoch_duration
                )
                h5f.attrs["n_epochs"] = len(file_data[0]["signal"])  # after cleaning

                # Signals
                grp_signals = h5f.create_group("signals")

                for file_data in all_file_data:
                    signal = file_data["signal"].flatten()

                    # if channel_name came from h5 originally, keep only the last part after slash
                    if "h5" in self.dataset.file_extensions["psg_ext"] and "/" in file_data["ch_name"]:
                        group_name = file_data["ch_name"].split("/")[-1]
                    else:
                        group_name = file_data["ch_name"]
                    ch_grp = grp_signals.create_group(group_name)

                    ch_grp.create_dataset(
                        "data", data=signal, compression="gzip", shuffle=True
                    )
                    # Channel metadata
                    ch_grp.attrs["ch_label"] = file_data["ch_name"]
                    ch_grp.attrs["ch_label_orig"] = file_data["ch_name_orig"]
                    ch_grp.attrs["unit"] = file_data.get("unit", "a.u.")
                    ch_grp.attrs["sampling_rate"] = file_data["sampling_rate"]

                # Annotations
                if self.config.use_annot:
                    labels = np.asarray(
                        all_file_data[0]["labels"]
                    )  # annotations are the same for all channels, take first one
                    if labels.ndim == 1:
                        h5f.create_dataset("y", data=labels, compression="gzip")
                    elif labels.ndim == 2:
                        h5f.create_dataset("y", data=labels[:, 0], compression="gzip")
                        h5f.create_dataset("y2", data=labels[:, 1], compression="gzip")

