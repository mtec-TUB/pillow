"""
Main dataset processor for PSG data preparation.
"""

import logging
import os
import re
from math import ceil, floor
from datetime import datetime, date, time, timedelta
from pyedflib import EdfWriter
import h5py
import glob
from pathlib import Path
from decimal import Decimal
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from ..utils import LoggingManager, Alignment
from .dataset_explorer import Dataset_Explorer
from .signal_processor import SignalProcessor

logging.captureWarnings(True)

# Final sleep stage labels mapping (did not yet find a better place for this, maybe in config?)
# labels will appear like this in output 
STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6, "SLEEP": 7}

SLEEP_STAGES = [STAGE_DICT["N1"],STAGE_DICT["N2"],STAGE_DICT["N3"],STAGE_DICT["REM"]]


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
                self.config.psg_dir,
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

                            file_processor = FileProcessor(self.config, self.dataset, psg_fname, ann_fname)

                            future = executor.submit(file_processor._process_file)
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
                                # self.pipeline_logger.error(f"Processing failed: {e}")
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
    """Class to process a single PSG file with one or multiple channels and save the processed data to file."""

    def __init__(self, config, dataset, psg_fname, ann_fname):
        self.logging_manager = LoggingManager(console_level=config.logging_level)
        self.config = config
        self.dataset = dataset
        self.psg_fname = psg_fname
        self.ann_fname = ann_fname

    def _process_file(self):
        """Process a single PSG file for all specified channels."""

        try:
            # Start buffering logs for this file
            file_id = Path(self.psg_fname).stem
            self.logger, buffer_handler = self.logging_manager.create_file_logger(file_identifier=file_id)

            self.logger.info(f"Signal file: {Path(self.psg_fname).relative_to(self.config.psg_dir)}")
            if self.config.use_annot:
                self.logger.info(f"Annotation file: {Path(self.ann_fname).relative_to(self.config.ann_dir)}")

            # Initialize signal data dictionary which holds all necessary info for processing and saving
            file_data = {
                "psg_fname": self.psg_fname,
                "ann_fname": self.ann_fname,
            }

            if self.config.output_format in ["hdf5", "edf"]:
                file_output_path, log_path = self._setup_file_output()
                # Skip if file already exists and overwrite is False
                if os.path.exists(file_output_path) and not self.config.overwrite:
                    self.logger.info(f"File already exists: {file_output_path}, skipping file.")
                    return
            else:
                log_paths = {}  # Store log paths for each channel separately

            # Get Start datetime and duration of polysomnography data
            file_info = self.dataset.get_file_info(self.logger, self.psg_fname)
            if file_info == {}:
                return
            
            start_datetime = file_info["start_datetime"]
            if isinstance(start_datetime, datetime):
                start_datetime = start_datetime.replace(tzinfo=None)
            file_data["start_datetime"] = start_datetime
            self.logger.info(f"Start datetime: {start_datetime}")

            if file_info["file_duration"] // self.config.epoch_duration < 1:
                self.logger.warning(
                    f"File does not hold at least one epoch, only {file_info['file_duration']:.2f} seconds, skipping file.")
                return

            file_data["file_duration"] = file_info["file_duration"]
            self.logger.info(f"File duration: {file_data['file_duration']} sec, {file_data['file_duration']/3600:.2f} h")

            # Annotations and file-level label preparation
            if self.config.use_annot:
                ann_stage_events, ann_startdatetime, lights_off, lights_on = self.dataset.ann_parse(self.ann_fname)
                if isinstance(ann_startdatetime, datetime):
                    ann_startdatetime = ann_startdatetime.replace(tzinfo=None)
                file_data["ann_start_datetime"] = ann_startdatetime
                file_data.update({"lights_off": lights_off, "lights_on": lights_on})

                if ann_stage_events == []:
                    self.logger.warning(f"No sleep stage annotations found in {os.path.basename(self.ann_fname)}, skipping file.")
                    return
                
                # Map dataset-labels to standardized labels and check consistency
                labels = self.dataset.ann_label(self.logger, ann_stage_events, STAGE_DICT, self.config.epoch_duration)
                # Check how many sleep epochs are in the file
                sleep_mask = np.isin(labels if labels.ndim == 1 else labels[:, 0], SLEEP_STAGES)
                if np.sum(sleep_mask) < self.config.min_sleep_epochs:
                    self.logger.warning("File contains less sleep epochs than required, skipping file.")
                    return

                # ── Front alignment (file level) ─────────────────────────────────
                start_delay = self._get_start_delay(ann_startdatetime, start_datetime)
                file_data["start_delay"] = start_delay

                if start_delay != 0:
                    if not self.dataset.has_front_alignment:
                        raise Exception(
                            f"Signal/annotation start-time mismatch ({start_delay:.1f}s) but "
                            f"dataset {self.dataset.dset_name} has has_front_alignment=False. "
                            f"Please verify."
                        )
                    front_params = self.dataset.compute_front_alignment(
                        self.logger, self.config.alignment, self.config.pad_values,
                        self.config.epoch_duration, start_delay,
                    )
                    labels = self._apply_front_label_adjustment(labels, front_params["label_adjust_front"])
                    signal_adjust_front_sec = front_params["signal_adjust_front_sec"]
                    start_time_shift = -signal_adjust_front_sec
                    if start_time_shift and isinstance(start_datetime, datetime):
                        start_datetime = start_datetime + timedelta(seconds=start_time_shift)
                        file_data["start_datetime"] = start_datetime
                        self.logger.info(f"Adjusted start datetime after front alignment: {start_datetime}")
                else:
                    signal_adjust_front_sec = 0.0
                    start_time_shift        = 0.0

                file_data["signal_adjust_front_sec"] = signal_adjust_front_sec
                file_data["start_time_shift"]        = start_time_shift
                file_data["labels"] = labels  # adjusted labels, not yet epoch-selected

            else:
                file_data.update({
                    "labels":            None,
                    "ann_start_datetime": None,
                    "lights_off":        None,
                    "lights_on":         None,
                    "start_delay":       0,
                    "signal_adjust_front_sec": 0.0,
                    "start_time_shift":        0.0,
                })

            # ── Lights markers → epoch indices (uses possibly updated start_datetime) ─
            if self.config.select_epochs == "lights":
                lights_off_ds, lights_on_ds = self.dataset.get_light_times(self.logger, self.psg_fname)
                if lights_off_ds is not None:
                    file_data["lights_off"] = lights_off_ds
                if lights_on_ds is not None:
                    file_data["lights_on"] = lights_on_ds
                # Calculate the epochs to select later between lights off and lights on
                file_data["lights_off"], file_data["lights_on"] = self._get_lights_epochs(file_data)

            # Intersection of available channels in psg file and configured channels to process
            channels = list(
                set(self.config.channels) & set(self.dataset.get_channels(self.logger, self.psg_fname))
            )
            if len(channels) == 0:
                self.logger.warning("No selected channels found in this file. Skipping.")
                return

            # ── Per-channel processing: raw signal → epoched signal ───────────────
            # Each channel returns signal_epoched (n_epochs × n_samples), fs.
            # No label handling, no epoch selection inside channel processor.
            all_channel_data = {}
            for channel in sorted(channels):
                # Set current channel for logging
                buffer_handler.set_channel(channel)
                channel_harm = self._harmonize_channel_name(channel)
                if self.config.output_format == "npz":
                    file_output_path, log_path = self._setup_channel_output(channel_harm)
                    log_paths[channel] = log_path

                # all_channel_data["file_output_path"] = file_output_path
                # if os.path.exists(file_output_path):
                #     raise FileExistsError(f"Output file already exists: {file_output_path}.")
                # np.savetxt(file_output_path, [])
                # continue

                # Skip if file already exists and overwrite is False (only for output format npz on this level)
                if os.path.exists(file_output_path) and not self.config.overwrite:
                    self.logger.info(f" File already exists: {file_output_path}, skipping channel {channel}.")
                    continue

                channel_processor = ChannelProcessor(self.logger, self.config, self.dataset, channel)
                ch_result = channel_processor._process_channel(dict(file_data))

                if ch_result is not None:
                    all_channel_data[channel_harm] = {
                        **ch_result,
                        "ch_name": channel_harm,
                        "file_output_path": file_output_path,
                    }
                self.logger.info("=" * 40)

            if not all_channel_data:
                return

            # ── File-level post-processing ────────────────────────────────────────
            # 1. End alignment: reconcile signal epoch count with label count.
            #    All channels from the same file share the same n_epochs after reshaping.
            if self.config.use_annot:
                n_signal_epochs = len(next(iter(all_channel_data.values()))["signal_epoched"])
                target_n_epochs = len(labels)
                if n_signal_epochs != target_n_epochs:
                    if not self.dataset.has_end_alignment:
                        raise Exception(
                            f"Signal ({n_signal_epochs} epochs) and labels ({target_n_epochs} epochs) "
                            f"do not match. Dataset {self.dataset.dset_name} has no end alignment "
                            f"defined (has_end_alignment=False). Please verify."
                        )
                    labels = self._end_align(labels, all_channel_data, n_signal_epochs, target_n_epochs)

                # 2. Compute epoch selection on post-end_align labels, then apply.
                select_idx = self._compute_select_idx(
                    labels, file_data.get("lights_off"), file_data.get("lights_on")
                )
                if select_idx is None:
                    return
                labels_out = labels[select_idx]
                if isinstance(start_datetime, datetime) and len(select_idx) > 0:
                    start_datetime = start_datetime + timedelta(
                        seconds=float(select_idx[0] * self.config.epoch_duration)
                    )
                for ch_data in all_channel_data.values():
                    ch_data["signal"]       = ch_data["signal_epoched"][select_idx]
                    ch_data["labels"]       = labels_out
                    ch_data["start_datetime"] = start_datetime
            else:
                if self.config.select_epochs == "lights":
                    n_sig = len(next(iter(all_channel_data.values()))["signal_epoched"])
                    lo = file_data.get("lights_off") or 0
                    li = file_data.get("lights_on")
                    end_idx = min(li, n_sig) if li is not None else n_sig
                    select_idx = np.arange(lo, end_idx) if lo < end_idx else np.arange(n_sig)
                    for ch_data in all_channel_data.values():
                        ch_data["signal"]         = ch_data["signal_epoched"][select_idx]
                        ch_data["labels"]         = None
                        ch_data["start_datetime"] = start_datetime
                else:
                    for ch_data in all_channel_data.values():
                        ch_data["signal"]         = ch_data["signal_epoched"]
                        ch_data["labels"]         = None
                        ch_data["start_datetime"] = start_datetime

            labels_shape = labels_out.shape if self.config.use_annot else None
            self.logger.info(
                f"Data after selection: signal {next(iter(all_channel_data.values()))['signal'].shape}, "
                f"labels {labels_shape}"
            )

            self._save_processed_data(all_channel_data)
            self.logger.info("Successfully processed")

        except Exception as e:
            # Log the exception
            self.logger.error(f"Error processing file: {str(e)}", exc_info=True)
            raise  # Re-raise after logging        
        finally:

            # Flush buffered logs to console and file(s), even if an exception occurred
            if self.config.output_format == "npz":
                if log_paths == {}:
                    # Error happened before log paths could be set up, flush to console for this case
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
        else:
            if labels.ndim == 1:
                pad = np.full(label_adjust_front, pad_val, dtype=labels.dtype)
                return np.concatenate([pad, labels])
            else:
                pad = np.full((label_adjust_front, labels.shape[1]), pad_val, dtype=labels.dtype)
                return np.vstack([pad, labels])

    def _compute_select_idx(self, labels, lights_off_epoch, lights_on_epoch):
        """Compute the epoch indices to keep based on lights markers and MOVE/UNK removal.

        labels    : 1-D or 2-D label array, or None when use_annot=False.
                    When None, label-dependent modes (integer wake buffer, rm_move/rm_unk,
                    truncate_non_sleep_end) are skipped.
        n_epochs  : total epoch count; derived from labels when labels is not None.
        Uses the first scorer column when labels are 2-D.
        Returns None if no epochs survive.
        """
        label_1d = labels if labels.ndim == 1 else labels[:, 0]
        n_epochs  = len(label_1d)
        start_idx = 0
        end_idx   = n_epochs

        sleep_mask = np.isin(label_1d, SLEEP_STAGES)
        sleep_idx  = np.where(sleep_mask)[0]

        if self.config.select_epochs == "all":
            pass

        elif self.config.select_epochs == "lights":
            if lights_off_epoch is not None:
                start_idx = lights_off_epoch

            if lights_on_epoch is not None:
                if lights_on_epoch <= n_epochs:
                    end_idx = lights_on_epoch
                elif lights_on_epoch == n_epochs + 1:
                    pass  # lights on fell in the last cropped partial epoch -> keep all
                else:
                    # Maybe padding both signal and annotations until Lights On ??
                    self.logger.warning(f"Lights On is {lights_on_epoch - n_epochs} epochs after recording ends (no selection applied).")
            else:
                if self.config.truncate_non_sleep_end and len(sleep_idx) > 0:
                    end_idx = sleep_idx[-1] + 1
                    self.logger.info(f"Removed {n_epochs - end_idx} Non-Sleep epochs at end of night.")

        elif isinstance(self.config.select_epochs, int):
            n_select  = self.config.select_epochs
            start_idx = max(0, sleep_idx[0] - n_select)
            end_idx   = min(n_epochs, sleep_idx[-1] + n_select + 1)
            if start_idx > 0 or end_idx != n_epochs:
                n_crop = n_epochs - (end_idx - start_idx)
                self.logger.info(
                    f"Cropped {n_crop} epochs ({n_crop * self.config.epoch_duration / 60:.1f} min) of extensive wake"
                )

        move_idx   = np.where(label_1d == STAGE_DICT["MOVE"])[0] if self.config.rm_move else []
        unk_idx    = np.where(label_1d == STAGE_DICT["UNK"])[0]  if self.config.rm_unk  else []
        if len(move_idx) > 0:
            self.logger.info(f"Removing {len(move_idx)} Movement epochs")
        if len(unk_idx) > 0:
            self.logger.info(f"Removing {len(unk_idx)} Unknown epochs")

        remove_idx = np.union1d(move_idx, unk_idx)
        select_idx = np.setdiff1d(np.arange(start_idx, end_idx), remove_idx)

        if len(select_idx) == 0:
            self.logger.warning(
                f"No data left after epoch selection. "
                f"start_idx={start_idx}, end_idx={end_idx}, removed={len(remove_idx)}"
            )
            return None
        return select_idx

    def _end_align(self, labels, all_channel_data, n_signal_epochs, target_n_epochs):
        """Reconcile signal epoch count vs label epoch count using the alignment config.

        All channels share the same n_signal_epochs (same file, same duration).
        Mutates signal_epoched in-place inside all_channel_data; may also mutate labels.
        Returns the (possibly updated) labels array.
        """
        self.logger.info(
            f"End alignment needed: signal has {n_signal_epochs} epochs, labels have {n_labels_epochs} epochs."
        )
        if n_signal_epochs > target_n_epochs:
            if self.config.alignment in (Alignment.MATCH_SHORTER.value, Alignment.MATCH_ANNOT.value):
                self.logger.info(f"End alignment: cropping signal to {n_labels_epochs} epochs.")
                # in-place modification
                for ch in all_channel_data.values():
                    ch["signal_epoched"] = ch["signal_epoched"][:target_n_epochs]
            elif self.config.alignment in (Alignment.MATCH_LONGER.value, Alignment.MATCH_SIGNAL.value): #pad labels
                n_pad   = n_signal_epochs - target_n_epochs
                pad_val = self.config.pad_values["label"]
                self.logger.info(f"End alignment: padding labels by {n_pad} epochs.")
                if labels.ndim == 1:
                    labels = np.concatenate([labels, np.full(n_pad, pad_val, dtype=labels.dtype)])
                else:
                    labels = np.vstack([labels, np.full((n_pad, labels.shape[1]), pad_val, dtype=labels.dtype)])
        else:
            if self.config.alignment in (Alignment.MATCH_SHORTER.value, Alignment.MATCH_SIGNAL.value):
                self.logger.info(f"End alignment: cropping labels to {n_signal_epochs} epochs.")
                labels = labels[:n_signal_epochs]
            elif self.config.alignment in (Alignment.MATCH_LONGER.value, Alignment.MATCH_ANNOT.value):  # pad signal
                n_pad   = target_n_epochs - n_signal_epochs
                pad_val = np.float64(self.config.pad_values["signal"])
                self.logger.info(f"End alignment: padding signal by {n_pad} epochs.")
                # in-place modification
                for ch in all_channel_data.values():
                    sig = ch["signal_epoched"]
                    ch["signal_epoched"] = np.vstack(
                        [sig, np.full((n_pad, sig.shape[1]), pad_val)]
                    )
        return labels

    def _harmonize_channel_name(self, channel):
        """Harmonize channel name based on dataset-specific mapping."""
        if self.config.map_channel_names:
            channel_harm = self.dataset.map_channel(channel)
            self.logger.info(f" Mapped channel name {channel} to {channel_harm}")
            return channel_harm
        else:
            return channel
        
    def _setup_file_output(self):
        """ Setup output for edf and hdf5 output format where all channels of one file are stored together and share the same log file. """
        # Create output directory (including subfolders of original data structure if keep_folder_structure is True)
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(Path(self.psg_fname).relative_to(self.config.psg_dir))[0]
        else:
            relative_path = ""

        # Output is generated per PSG file containing all channels, log files are saved per PSG file separately
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
        """Setup output for npz output format where each channel is stored separately and has its own log file."""

        # Create output directory (including subfolders of original data structure if keep_folder_structure is True)
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(Path(self.psg_fname).relative_to(self.config.psg_dir))[0]
        else:
            relative_path = ""

        # Output is generated per channel and sorted into channel folders with corresponding log file
        # replace slash in folder names to avoid nested output structure and colon because it is often not accepted in folder names
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
        """ Check if annotation and signal start datetime are aligned and calculate the delay between annotation and signal start if necessary.
        """
        delay = 0
        if ann_start_datetime != None:
            # If annotation start datetime is given as a datetime object, compare with signal start datetime and calculate start delay
            if isinstance(ann_start_datetime, datetime) and signal_start_datetime is not None:
                # Strip timezone info before comparison/subtraction
                ann_dt = ann_start_datetime
                sig_dt = signal_start_datetime
                if ann_dt.time() != sig_dt.time():
                    delay = (ann_dt - sig_dt).total_seconds()
                    if abs(delay) >= 24*3600:
                        raise Exception(f"Annotation start datetime {ann_dt} is more than 24h after signal start datetime {sig_dt}. Check implementation!")
                    
            # If annotation start datetime is given as a numeric value, it indicates a delay in seconds or samples (depending on dataset)
            elif isinstance(ann_start_datetime, (int, float, Decimal)):
                delay = ann_start_datetime
            else:
                raise Exception(f"Unsupported format of annotation start datetime: {ann_start_datetime}")
            if delay != 0:
                self.logger.info(f"Start of signal: {signal_start_datetime}, Start of labels: {ann_start_datetime}")

        return delay
    
    def _get_lights_epochs(self, channel_data):
        """Get the epochs where lights off and lights on happen based on the configured lights marker in annotation or PSG data.
        - The epoch in which lights off event happens is return as lights_off_epoch
        - The epoch AFTER which lights on event happens is returned as lights_on_epoch, 
            meaning that the epoch in which lights on event happens is still included in the selected data. 

        Args:
            channel_data (_type_): Dictionary containing minimum the following keys: "psg_fname", "start_datetime", "lights_off", "lights_on"

        Returns:
            _type_: epochs of lights off and lights on
        """
        
        startdatetime    = channel_data["start_datetime"]
        start_time_shift = channel_data.get("start_time_shift", 0.0)
        lights_off_epoch = 0
        lights_on_epoch  = None

        if channel_data["lights_off"] is not None:
            lights_off = channel_data["lights_off"]
            if isinstance(lights_off, (datetime,time)):
                if isinstance(lights_off, time):
                    # if date of marker timestamp is not available, take date of signal start for easier comparison
                    lights_off = datetime.combine(startdatetime.date(),lights_off)

                lights_off_sec = (lights_off - startdatetime).total_seconds()

                if lights_off_sec != 0:
                    if lights_off_sec < 0 and lights_off_sec > -3600:       
                        # Between -3600 and 0, before signal start but in 1 hour range
                        self.logger.info(f"Lights Off time {lights_off} is before signal start time {startdatetime.time()}. No epoch selection applied.")
                        # maybe padding with wake epochs until lights Off time is reached? For now, just keep all epochs and do not select (keep lights_off_epoch at 0)
                    else:
                        if lights_off_sec < 0:
                            # Assume lights Off is after and signal start before midnight
                            lights_off_sec += 24*3600 # add 24h
                        
                        # Round to full epoch
                        lights_off_epoch = self._round_marker_time("lights_off", lights_off_sec, self.config.epoch_duration, lights_off.time())
                        
                        self.logger.info(f"Select only epochs after {(startdatetime + timedelta(seconds=lights_off_epoch * self.config.epoch_duration)).time()} (based on Lights Off marker in epoch {lights_off_epoch})")
                 
                else:
                    self.logger.info("Lights Off time is at the start of the signal, no need of epoch selection based on lights Off time.")

            elif isinstance(lights_off, (int,float)):
                # int/float is seconds from the ORIGINAL signal start; subtract start_time_shift
                # so the value becomes relative to the (possibly cropped/padded) start_datetime.
                lights_off_sec = lights_off - start_time_shift
                if lights_off_sec != 0:
                    if lights_off_sec < 0 and lights_off_sec > -3600:   # Lights Off probably starts before PSG Data (1 hour range before signal start)
                        self.logger.warning(f"Lights Off time {lights_off_sec} is before signal start time {startdatetime.time()}. No epoch selection applied.")
                        # front_padding = 
                        # maybe padding with wake epochs until lights Off time is reached? For now, just keep all epochs and do not select
                    elif lights_off_sec < 0:
                        raise Exception(f"Lights Off time ({lights_off_sec}) is more than 1 hour before signal start ({startdatetime.time()})")
                    else:
                        # Round to full epoch
                        lights_off_epoch = self._round_marker_time("lights_off", lights_off_sec, self.config.epoch_duration)
                        self.logger.info(f"Select only epochs after lights Off at second {lights_off_sec} (epoch {lights_off_epoch})")
                else:
                    self.logger.info("Lights Off time is at the start of the signal, no need of epoch selection based on lights Off time.")
            else:
                raise Exception(f"Lights Off time has unsupported format: {lights_off}")
        
        else:
            self.logger.warning(f"Lights Off time not available, keeping all wake epochs at start.")
            
        if channel_data["lights_on"] is not None:
            lights_on = channel_data["lights_on"]
            # Info: Lights On marker can happen to be after signal end, cannot check this here because signal length is not determined yet
            if isinstance(lights_on, (datetime,time)):
                if isinstance(lights_on, time):
                    # if date of marker timestamp is not available, take date of signal start for easier comparison
                    lights_on = datetime.combine(startdatetime.date(),lights_on)
                
                # seconds from recording start until lights On
                lights_on_sec = (lights_on - startdatetime).total_seconds()

                if lights_on_sec < 0:
                    # Assume lights On is after and signal start before midnight
                    lights_on_sec += 24 * 3600  # add 24h

                lights_on_epoch = self._round_marker_time("lights_on", lights_on_sec, self.config.epoch_duration, lights_on.time())
                self.logger.info(f"Select only epochs before {(startdatetime + timedelta(seconds=lights_on_epoch*self.config.epoch_duration)).time()} (based on Lights On marker in epoch {lights_on_epoch})")

            elif isinstance(lights_on, (int,float)):
                # int/float is seconds from the ORIGINAL signal start; subtract start_time_shift
                # so the value becomes relative to the (possibly cropped/padded) start_datetime.
                lights_on_sec = lights_on - start_time_shift

                if lights_on_sec < 0:
                    raise Exception(f"Lights On time ({lights_on_sec}) is before signal start ({startdatetime.time()})")
                        
                lights_on_epoch = self._round_marker_time("lights_on", lights_on_sec, self.config.epoch_duration)
                self.logger.info(f"Select only epochs before lights On at second {lights_on_sec} (epoch {lights_on_epoch})")
            else:
                raise Exception(f"Lights On time has unsupported format: {lights_on}")
        else:
            self.logger.warning(f"Lights On time not available.")

        return lights_off_epoch, lights_on_epoch
   
    def _round_marker_time(self, marker, marker_sec, epoch_duration, marker_time=None):
        """Helper function to round lights Off/On time to the next (or previous) epoch if it is not exactly at the start/end of an epoch and log this behavior.
        
        Args:
            marker (_type_): Type of marker which is processed ("lights_off" or "lights_on")
            marker_sec (_type_): timestamp of marker in seconds from signal start
            marker_time (_type_): timestamp of marker as datetime.time object for logging purposes (if not available, )
            epoch_duration (_type_): epoch duration in seconds to round the marker time to

        Returns:
            _type_: epoch number corresponding to the rounded marker time
        """
        if marker_time is None:
            marker_time = marker_sec
        if marker_sec % epoch_duration != 0:              
            if marker == "lights_off":
                round_marker_epoch = floor(marker_sec / epoch_duration)
                self.logger.info(f"Lights Off time {marker_time} is floored to {round_marker_epoch * epoch_duration}sec from signal start.")
            elif marker == "lights_on":
                round_marker_epoch = ceil(marker_sec / epoch_duration)
                self.logger.info(f"Lights On time {marker_time} is ceiled to  {round_marker_epoch * epoch_duration}sec from signal start.")
        else:
            round_marker_epoch = marker_sec / epoch_duration
        return int(round_marker_epoch)
 
    def _save_processed_data(self, all_channel_data):
        """Save processed data to file for npz, edf, hdf5 formats."""
        output_format = self.config.output_format
        channels_sorted = sorted(all_channel_data.keys())
        channel_dicts = [all_channel_data[ch] for ch in channels_sorted]
        n_epochs = [len(channel_data["signal"]) for channel_data in channel_dicts]
        if len(np.unique(n_epochs)) != 1:
            self.logger.error(f"All channels must have the same number of epochs after processing to be saved together, but got different number of epochs: {n_epochs}")
            raise ValueError(f"All channels must have the same number of epochs after processing to be saved together, but got different number of epochs: {n_epochs}")

        # check if there is a signal with only NaN values
        if any([np.isnan(channel_data["signal"]).all() for channel_data in channel_dicts]):
            self.logger.warning(f"One or more channels contain only NaN values after processing. Skipping this file.")
            return

        if output_format == "npz":
            for channel_data in channel_dicts:
                save_dict = {
                    "x": channel_data["signal"],
                    "fs": channel_data["sampling_rate"],
                    "ch_label": channel_data["ch_name"],
                    "ch_label_orig": channel_data["ch_name_orig"],
                    "file_duration": len(channel_data["signal"]) * self.config.epoch_duration,
                    "epoch_duration": self.config.epoch_duration,
                    "n_epochs": len(channel_data["signal"]),  # after cleaning
                }

                # Write Annotations
                if self.config.use_annot:
                    # Handle multiple scorers
                    labels = channel_data["labels"]
                    if labels.ndim == 1:
                        save_dict["y"] = labels
                    elif labels.ndim == 2:
                        save_dict["y"] = labels[:, 0]
                        save_dict["y2"] = labels[:, 1]
                save_dict["unit"] = channel_data.get("unit", "a.u.")
                np.savez(channel_data["file_output_path"], **save_dict)

        elif output_format == "edf":
            file_output_path = channel_dicts[0]["file_output_path"]
            with EdfWriter(file_output_path, n_channels=len(channel_dicts)) as edf_writer:
                for i, channel_data in enumerate(channel_dicts):
                    signal = channel_data["signal"].flatten()
                    scale = 10**3  # to get 3 decimals for physical min and max
                    phys_min = floor(np.nanmin(signal) * scale) / scale
                    phys_max = ceil(np.nanmax(signal) * scale) / scale
                    if phys_min == phys_max:
                        phys_min -= 1.0
                        phys_max += 1.0
                    channel_info = {
                        "label": channel_data["ch_name"],
                        "dimension": channel_data.get("unit", "a.u."),
                        "sample_frequency": float(channel_data["sampling_rate"]),
                        "physical_min": phys_min,
                        "physical_max": phys_max,
                        "digital_min": -32768,
                        "digital_max": 32767,
                        "transducer": "",
                        "prefilter": "",
                    }
                    edf_writer.setSignalHeader(i, channel_info)
                if isinstance(channel_dicts[0]["start_datetime"], datetime):
                    if channel_dicts[0]["start_datetime"].date() > date(1985, 1, 1):
                        Startdatetime = channel_dicts[0]["start_datetime"]
                    else:
                        Startdatetime = datetime.combine(date(1985, 1, 1), channel_dicts[0]["start_datetime"].time())
                else:
                    Startdatetime = datetime(1985, 1, 1, 0, 0, 0)

                # Write samples
                edf_writer.setStartdatetime(Startdatetime)
                edf_writer.writeSamples([channel_data["signal"].flatten() for channel_data in channel_dicts])

                # Write annotations
                if self.config.use_annot:
                    all_labels = np.array([channel_data["labels"] for channel_data in channel_dicts])
                    duration = self.config.epoch_duration
                    for epoch_idx, label in enumerate(all_labels[0]):
                        edf_writer.writeAnnotation(
                            onset_in_seconds=epoch_idx * duration,
                            duration_in_seconds=duration,
                            description=str(label),
                        )

        elif output_format == "hdf5":
            file_output_path = channel_dicts[0]["file_output_path"]
            with h5py.File(file_output_path, "w") as h5f:
                # Metadata
                h5f.attrs["epoch_duration"] = self.config.epoch_duration
                h5f.attrs["file_duration"] = len(channel_dicts[0]["signal"]) * self.config.epoch_duration
                h5f.attrs["n_epochs"] = len(channel_dicts[0]["signal"])
                grp_signals = h5f.create_group("signals")
                for channel_data in channel_dicts:
                    signal = channel_data["signal"].flatten()

                    # if channel_name came from h5 originally, keep only the last part after slash
                    if "h5" in self.dataset.file_extensions["psg_ext"] and "/" in channel_data["ch_name"]:
                        group_name = channel_data["ch_name"].split("/")[-1]
                    else:
                        group_name = channel_data["ch_name"]
                    ch_grp = grp_signals.create_group(group_name)
                    # Signal
                    ch_grp.create_dataset("data", data=signal, compression="gzip", shuffle=True)
                    ch_grp.attrs["ch_label"] = channel_data["ch_name"]
                    ch_grp.attrs["ch_label_orig"] = channel_data["ch_name_orig"]
                    ch_grp.attrs["unit"] = channel_data.get("unit", "a.u.")
                    ch_grp.attrs["sampling_rate"] = channel_data["sampling_rate"]

                # Annotations
                if self.config.use_annot:
                    labels = np.asarray(channel_dicts[0]["labels"])
                    if labels.ndim == 1:
                        h5f.create_dataset("y", data=labels, compression="gzip")
                    elif labels.ndim == 2:
                        h5f.create_dataset("y", data=labels[:, 0], compression="gzip")
                        h5f.create_dataset("y2", data=labels[:, 1], compression="gzip")


class ChannelProcessor:
    """Process a single channel: raw signal-> epoched signal array.

    - Read raw signal and sampling rate from file.
    - Resample / filter / clip (all fs-dependent).
    - Apply the per-channel raw-sample front offset (signal_adjust_front_sec × fs)
      AFTER resampling and filtering to avoid boundary artefacts.
    - Reshape into epochs.
    """

    def __init__(self, logger, config, dataset, channel):
        self.logger  = logger
        self.config  = config
        self.dataset = dataset
        self.channel = channel

    def _process_channel(self, data):
        """Read and process one channel; return dict with signal_epoched and fs.

        Returns None if the channel cannot be processed (missing signal, etc.).
        """
        data["ch_name_orig"] = self.channel

        # 1. Read raw signal
        psg_data = self.dataset.get_signal_data(self.logger, data["psg_fname"], data["ch_name_orig"])
        if psg_data == {}:
            return None
        signal = psg_data["signal"].astype(np.float64)
        fs     = psg_data["sampling_rate"]
        unit   = psg_data.get("unit", "a.u.")
        del psg_data

        self.logger.info(f" Channel {data['ch_name_orig']} has {len(signal)} samples ({fs:.2f} Hz)")

        # 2. Resample / filter / clip (fs-dependent, must come before raw-sample crop)
        if self.config.resample is not None or self.config.filter:
            signal_processor = SignalProcessor(
                self.logger, signal, data["ch_name_orig"], self.config, self.dataset.channel_types
            )
            if self.config.resample is not None:
                signal_processor.resample_signal(fs, self.config.resample)
                fs = self.config.resample
            if self.config.filter:
                signal_processor.filter_signal(fs, self.dataset.channel_groups)
            signal_processor.clip_signal()
            signal = signal_processor.signal

        # 3. Apply raw-sample front offset AFTER resample/filter
        signal_adjust_front_sec = data.get("signal_adjust_front_sec", 0.0)
        if signal_adjust_front_sec != 0.0:
            signal = self._apply_partial_epoch_offset(signal, fs, signal_adjust_front_sec)

        # 4. Reshape into epochs
        n_epoch_samples = self.config.epoch_duration * fs
        if not n_epoch_samples.is_integer():
            raise ValueError(
                f"Epoch duration {self.config.epoch_duration} sec with sampling rate {fs} Hz "
                "does not yield an integer number of samples per epoch."
            )
        n_epoch_samples = int(n_epoch_samples)

        n_epochs, remainder = divmod(len(signal), n_epoch_samples)
        if n_epochs < 1:
            self.logger.warning(
                f"Channel {data['ch_name_orig']}: fewer than one full epoch of signal available, skipping."
            )
            return None
        if remainder > 0:
            self.logger.info(f" Signal cropped to full epochs ({remainder / fs:.2f} sec removed).")

        signal_epoched = signal[:n_epochs * n_epoch_samples].reshape(n_epochs, -1)

        return {
            "signal_epoched": signal_epoched,
            "sampling_rate":  fs,
            "unit":           unit,
            "ch_name_orig":   self.channel,
        }

    def _apply_partial_epoch_offset(self, signal, fs, signal_adjust_front_sec):
        """Padding (positive) or crop (negative) at the raw signal front.

        Called after resampling/filtering so that filter edge effects are not
        introduced by padding, and the resampled fs is used for sample counting.
        """
        n_samples = int(abs(signal_adjust_front_sec) * fs)
        if signal_adjust_front_sec < 0:

            return signal[n_samples:]
        else:
            pad_val = np.float64(self.config.pad_values["signal"])
            return np.concatenate([np.full(n_samples, pad_val), signal])

