"""
Main dataset processor for PSG data preparation.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from decimal import Decimal
import numpy as np

from ..utils import LoggingManager
from .dataset_explorer import Dataset_Explorer
from .signal_processor import SignalProcessor

# Sleep stage labels mapping
STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6}


# def _process_channel_worker(args):
#     """
#     Worker for processing a single channel. Args is a tuple:
#     (psg_fname, ann_fname, channel, data_dir, output_dir, resample,
#      dataset_name, channel_types, channel_groups, file_extensions, alias_mapping,
#      keep_folder_structure, epoch_duration, overwrite, ann_stage_events, ann_Startdatetime)
#     """
#     (
#         psg_fname,
#         ann_fname,
#         channel,
#         data_dir,
#         output_dir,
#         resample,
#         dataset_name,
#         channel_types,
#         channel_groups,
#         file_extensions,
#         alias_mapping,
#         keep_folder_structure,
#         epoch_duration,
#         overwrite,
#         ann_stage_events,
#         ann_Startdatetime,
#     ) = args

#     # Reconstruct dataset processor class
#     from datasets.registry import get_dataset

#     Dataset = get_dataset(dataset_name)
#     try:
#         dataset= Dataset(dataset_name)
#     except Exception:
#         dataset = Dataset()

#     # Inject attributes if needed
#     dataset.channel_types = channel_types
#     dataset.channel_groups = channel_groups
#     dataset.file_extensions = file_extensions
#     dataset.alias_mapping = alias_mapping
#     dataset.keep_folder_structure = keep_folder_structure
#     # Create local processor and logger
#     local_processor = DatasetProcessor(overwrite=overwrite)
#     local_processor.logger = local_processor.logging_manager.setup_logger(output_dir, local_processor.overwrite)

#     # Create file handler factory
#     handler = get_handler(dataset_name, file_extensions['psg_ext'])

#     try:
#         ret = local_processor._process_single_channel(
#             psg_fname,
#             ann_fname,
#             channel,
#             handler,
#             data_dir,
#             output_dir,
#             resample,
#             dataset,
#             ann_stage_events,
#             ann_Startdatetime,
#             epoch_duration,
#         )
#     except Exception as e:
#         print(f"Error processing channel {channel} in {psg_fname}: {e}")
#         raise

#     return ret



class DatasetProcessor:
    """
    Main processor for PSG dataset preparation and signal processing.

    This class orchestrates the entire dataset processing pipeline,
    handling file processing, signal cleaning, and output generation.
    """

    def __init__(self, dataset, config):

        self.logging_manager = LoggingManager(level=logging.INFO)
        self.logger = None
        self.dataset = dataset
        self.config = config
        

    def prepare_files(self):

        try:
            # Set up logger and initialize components
            self.logger = self.logging_manager.setup_logger(self.config.output_dir, self.config.overwrite)

            # Get files using dataset-specific extensions
            explorer = Dataset_Explorer(
                self.logger,
                self.dataset.psg_file_handler,
                self.config.data_dir,
                self.config.ann_dir,
                **self.dataset.file_extensions,
            )
            psg_fnames, ann_fnames = explorer.get_files(self.config.allow_missing)

            # Process each file
            ann_idx = 0
            for psg_idx, psg_fname in enumerate(psg_fnames):
                print(f"\n--- Processing file {psg_idx+1}/{len(psg_fnames)} ---")
                if ann_fnames is not None:
                    ann_fname, ann_idx = self._find_matching_annotation(
                        psg_fname, ann_fnames, ann_idx, self.config.allow_missing
                    )
                    if ann_fname is None:
                        self.logger.warning(
                            f"No matching annotation found for PSG: "
                            f"{Path(psg_fname).relative_to(self.data_dir)}. Skipping file."
                        )
                        continue
                self._process_single_file(
                    psg_fname,
                    ann_fname if ann_fnames is not None else None,
                    self.config.num_jobs,
                )

            # Finalize processing
            self.logging_manager.cleanup_file_handlers(self.logger)
            self.logger.info("\n" + "=" * 60)
            self.logger.info("DATASET PREPARATION COMPLETED")
        except KeyboardInterrupt:
            self.logging_manager.cleanup_file_handlers(self.logger)
            self.logger.info("Stopped processing")

    def _find_matching_annotation(self, psg_fname, ann_fnames, start_idx, allow_missing=False):
        """
        Scan annotation files from start_idx forward and return the first match.
        Return (ann_fname, new_index).
        If no match is found:
            - If allow_missing=True -> return (None, start_idx)
            - else -> raise
        """
        psg_base = str(Path(psg_fname).relative_to(self.config.data_dir))

        for i in range(start_idx, len(ann_fnames)):
            ann_base = str(Path(ann_fnames[i]).relative_to(self.config.ann_dir))
            psg_id, ann_id = self.dataset.get_file_identifier(psg_base, ann_base)  # adapt as needed

            if psg_id == ann_id:
                return ann_fnames[i], i + 1  # move annotation pointer past this match

        # No match found
        if allow_missing:
            return None, start_idx
        else:
            raise Exception(
                f"No matching annotation found for PSG: {psg_base}"
            )

    def _process_single_file(
        self,
        psg_fname,
        ann_fname,
        num_workers=None,
    ):
        """Process a single PSG file for all specified channels."""

        # Load annotations before (same for all channels)
        ann_stage_events, ann_Startdatetime = self.dataset.ann_parse(ann_fname)

        if ann_stage_events == []:
            return

        # ann_stage_events = dataset.check_labels(self.logger, ann_stage_events, epoch_duration)

        # List channels to process for this file
        channels = list(set(self.dataset.channel_names) & set(self.dataset.psg_file_handler.get_channels(self.logger,psg_fname)))

        if num_workers and num_workers != 1:
            raise NotImplementedError("Multiprocessing not implemented in this version.")
            # # Build channel tasks for multiprocessing
            # tasks = []
            # for channel in channels:
            #     tasks.append(
            #         (
            #             psg_fname,
            #             ann_fname,
            #             channel,
            #             self.data_dir,
            #             self.output_dir,
            #             resample,
            #             dataset.dset_name,
            #             dataset.channel_types,
            #             dataset.channel_groups,
            #             dataset.file_extensions,
            #             dataset.alias_mapping,
            #             dataset.keep_folder_structure,
            #             epoch_duration,
            #             self.overwrite,
            #             ann_stage_events,
            #             ann_Startdatetime,
            #         )
            #     )

            # from multiprocessing import Pool

            # with Pool(processes=num_workers) as pool:
            #     for ret in pool.imap_unordered(_process_channel_worker, tasks):
            #         if ret is False:
            #             # if a worker indicates no more channels needed, break
            #             break
        else:
            # Sequential per-channel processing
            for channel in channels:
                ret = self._process_single_channel(
                    psg_fname,
                    ann_fname,
                    channel,
                    ann_stage_events,
                    ann_Startdatetime,
                )
                if not ret:  # marks that other channels of this file don't have to be processed
                    break

    def _process_single_channel(
        self,
        psg_fname,
        ann_fname,
        channel,
        ann_stage_events,
        ann_Startdatetime,
    ):
        """Process a single channel from a single file."""

        # Setup channel processing environment
        file_output_dir, file_output_path = self._setup_channel_output(
            channel,
            psg_fname,
        )

        # Skip if file already exists
        if not self.config.overwrite and self._output_file_exists(file_output_path):
            return True

        # Setup logging for this channel
        self.logging_manager.setup_channel_file_logging(self.logger, file_output_dir)

        self.logger.info(f"Signal file: {Path(psg_fname).relative_to(self.config.data_dir)}")
        self.logger.info(f"Annotation file: {Path(ann_fname).relative_to(self.config.ann_dir)}")

        # Extract and process signal
        signal_data = self.dataset.psg_file_handler.get_signal_data(self.logger,psg_fname, channel)

        if signal_data is None:
            return True

        self.logger.info(
            f"File duration: {signal_data['file_duration']} sec, {signal_data['file_duration']/3600:.2f} h"
        )

        # Add more information to signal_data annotations
        signal_data["ann_stage_events"] = ann_stage_events
        signal_data["psg_fname"] = psg_fname
        signal_data["ann_fname"] = ann_fname

        # Handle start datetime (take time from annotation file)
        if signal_data["start_datetime"] is None:
            signal_data["start_datetime"] = ann_Startdatetime
        
        # If annotations holds a start datetime, check if alignment is needed
        if ann_Startdatetime != None:

            if (isinstance(ann_Startdatetime, datetime) and
                signal_data["start_datetime"].time() != ann_Startdatetime.time()):
                start_time = (ann_Startdatetime - psg_start_datetime).total_seconds()

            elif isinstance(ann_Startdatetime, (int, float, Decimal)):  # ann_Startdatetime can be in seconds or samples (depends on dataset)
                start_time = ann_Startdatetime

                print(
                    f"Start of signal: {signal_data['start_datetime']} \nStart of labels: {ann_Startdatetime}"
                )

            if start_time != 0:
                # Shorten signal if annotations start later or align front to first common epoch if annotations start before
                signal, labels = self.dataset.align_front(
                    self.logger,
                    self.config.alignment,
                    self.config.pad_values,
                    self.config.epoch_duration,
                    start_time,
                    signal_data["signal"],
                    signal_data["ann_stage_events"],
                    signal_data["sampling_rate"],
                )
                signal_data["signal"] = signal
                signal_data["ann_stage_events"] = labels

        self.logger.info(f"Start datetime: {signal_data['start_datetime']}")

        # Print and log labels for every file and channel if desired
        # if self.log_labels():
        #     self._log_labels(signal_data["ann_stage_events"])
        signal_data["ann_stage_events"] = self.dataset.ann_label(
            self.logger, signal_data["ann_stage_events"], self.config.epoch_duration
        )

        # Process the signal (resample, filter, clean)
        ch_type = self._get_channel_type(channel)

        processed_data, continue_processing = self._process_signal_data(
            signal_data, channel, ch_type
        )

        if continue_processing is False:
            return False
        if processed_data is None:
            return True

        # replace x,y and n_epochs after the processing
        for key in processed_data:
            signal_data[key] = processed_data[key]

        # Save processed data
        self._save_processed_data(
            signal_data, file_output_path, channel
        )

        self.logger.info("=" * 40)
    
        return True

    # def _log_labels(self, ann_stage_events):
    #     for event in ann_stage_events:
    #         onset_sec = int(event["Start"])
    #         duration_sec = int(event["Duration"])
    #         ann_str = event["Stage"]

    #         self.logger.info("Include onset:{}, duration:{}, label:{} ({})".format(onset_sec, duration_sec, label, ann_str))

    def _get_channel_type(self, channel):
        """Get the type (analog/digital) for a specific channel."""
        for ch_type, channels in self.dataset.channel_types.items():
            if channel in channels:
                return ch_type

        raise Exception(f"channel {channel} not listed in channel_types")

    def _setup_channel_output(
        self,
        channel,
        psg_fname,
    ):
        """Setup output directory and filename for a channel."""

        # Handle channel name aliasing
        ch_name_path = channel
        if self.dataset.alias_mapping:
            alias_checking = [
                key for key, aliases in self.dataset.alias_mapping.items() if channel in aliases
            ]
            if alias_checking:
                ch_name_path = alias_checking[0]

        # replace slash in folder names to avoid nester output structure and colon because it is often not accepted in folder names
        ch_name_path = re.sub(r"[:/]", "_", ch_name_path)

        # Create output directory
        if self.dataset.keep_folder_structure:
            relative_path = os.path.split(Path(psg_fname).relative_to(self.config.data_dir))[0]
        else:
            relative_path = ""

        file_output_dir = os.path.join(self.config.output_dir, relative_path, "npz", ch_name_path)
        os.makedirs(file_output_dir, exist_ok=True)

        # Generate safe file and folder name
        base_filename = os.path.splitext(os.path.basename(psg_fname))[0] + ".npz"
        ch_name_safe = re.sub(r"[^a-zA-Z0-9._\-\s]", "_", channel)
        filename = f"{ch_name_safe}_{base_filename}"

        file_output_path = os.path.join(file_output_dir, filename)

        return file_output_dir,file_output_path

    def _output_file_exists(self, file_output_path):
        """Check if output file already exists."""
        if os.path.exists(file_output_path):
            print(f"File already exists: {file_output_path}")
            return True
        return False

    def _process_signal_data(
        self,
        signal_data,
        channel,
        ch_type,
    ):
        """Process signal data through the complete pipeline."""

        signal = signal_data["signal"]
        labels = signal_data["ann_stage_events"]
        sampling_rate = signal_data["sampling_rate"]

        # Check signal length
        if len(signal) // n_epoch_samples <= 1:
            self.logger.info(f"Signal too short, only {len(signal)} samples")
            return None, True

        signal_processor = SignalProcessor(self.logger)
        if self.config.resample is not None:
            # Resample signal
            signal, sampling_rate = signal_processor.resample_signal(
                signal,
                ch_type,
                sampling_rate,
                self.config.resample,
            )
        if self.config.filter:
            # Filter signal according to AASM
            signal = signal_processor.filter_signal(
                signal,
                sampling_rate,
                channel,
                self.dataset.channel_groups,
                ch_type,
            )

        # Reshape into epochs based on annotation start
        n_epoch_samples = int(self.config.epoch_duration * sampling_rate)
        n_epochs = len(signal) // n_epoch_samples
        print(
            f"Seconds in unfilled epoch: {len(signal)/sampling_rate - (n_epochs * self.config.epoch_duration):.4f} sec"
        )
        signals = signal[0 : int(n_epochs * self.config.epoch_duration * sampling_rate)].reshape(
            -1, n_epoch_samples
        )

        # if self.config.resample is None:
        #     # zero pad last eventually not full epoch
        #     last_epoch = signal[n_epochs * self.config.epoch_duration * sampling_rate :]
        #     n_last_epoch = len(last_epoch)
        #     if n_last_epoch > 0:
        #         last_epoch = np.pad(
        #             last_epoch,
        #             pad_width=(0, n_epoch_samples - n_last_epoch),
        #             constant_values=0,
        #         ).reshape(1, -1)
        #         signals = np.append(signals, last_epoch, axis=0)

        # Align labels (some datasets have different length of signal and annotation data)
        signals, labels = self.dataset.align_end(
            self.logger,
            self.config.alignment,
            self.config.pad_values,
            signal_data["psg_fname"],
            signal_data["ann_fname"],
            signals,
            labels,
        )
        # Clean signal data
        signals, labels, select_start = self._clean_signal(signals, labels)

        if signals is None:
            return None, False

        x, y = signals.astype(np.float32), labels.astype(np.int32)

        return {
            "x": x,
            "y": y,
            "sampling_rate": sampling_rate,
            "n_all_epochs": n_epochs,
            "rm_start_epochs": select_start,
        }, True

    def _clean_signal(self, x, y):
        """
        Clean signal by removing movement/unknown epochs and selecting sleep periods.

        Args:
            x: Signal epochs array
            y: Stage labels array

        Returns:
            Tuple of (cleaned_x, cleaned_y) or (None, None) if no sleep detected
        """
        self.logger.info(
            f"Starting signal cleaning - Input shape: x={x.shape}, y={y.shape}"
        )

        # Remove movement and unknown epochs if configured
        if self.config.rm_move:
            move_idx = np.where(y == STAGE_DICT["MOVE"])[0]  
        else:
            move_idx = []
        if len(move_idx) > 0:
            self.logger.info(f"  Removing Movement epochs: {len(move_idx)}")

        if self.config.rm_unk:
            unk_idx = np.where(y == STAGE_DICT["UNK"])[0]
        else:
            unk_idx = []
        if len(unk_idx) > 0:
            self.logger.info(f"  Removing Unknown epochs: {len(unk_idx)}")

        remove_idx = np.union1d(move_idx, unk_idx)

        sleep_idx = np.where(
            (y != STAGE_DICT["W"])
            & (y != STAGE_DICT["MOVE"])
            & (y != STAGE_DICT["UNK"])
        )[0]

        if len(sleep_idx) <= self.config.min_sleep_epochs:
            self.logger.warning("File contains less sleep epochs than required. Skipping")
            return None, None, None

        if self.config.n_wake_epochs == "all":
            start_idx = 0
            end_idx = len(y) - 1
        else:
            # Remove extensive wake epochs at start and end as given in config
            n_wake_epochs = int(self.config.n_wake_epochs)
            start_idx = max(0, sleep_idx[0] - n_wake_epochs)
            end_idx = min(len(y) - 1, sleep_idx[-1] + n_wake_epochs)

            self.logger.info(
                f"  Outside {int(self.config.n_wake_epochs)/2}min wake epochs: {start_idx + (len(x)-end_idx)-1}"
            )

        select_idx = np.setdiff1d(np.arange(start_idx, end_idx + 1), remove_idx)

        self.logger.info(f"  Total epochs to remove: {len(x) - len(select_idx)}")
        self.logger.info(f"Removed {select_idx[0]} epochs at beginning of signal")

        x = x[select_idx]
        y = y[select_idx]

        self.logger.info(f"  Data after cleaning: {x.shape}, {y.shape}")

        return x, y, select_idx[0]

    def _save_processed_data(
        self, signal_data, file_output_path, channel
    ):
        """Save processed data to file."""

        save_dict = {
            "x": signal_data["x"],
            "fs": signal_data["sampling_rate"],
            "ch_label": channel,
            "start_datetime": signal_data["start_datetime"],
            "file_duration": signal_data["file_duration"],
            "epoch_duration": self.config.epoch_duration,
            "n_all_epochs": signal_data["n_all_epochs"],
            "n_epochs": len(signal_data["x"]),
            "rm_start_epochs": signal_data["rm_start_epochs"],
        }

        # Handle multiple scorers
        y = signal_data["y"]
        if y.ndim == 1:
            save_dict["y"] = y
        elif y.ndim == 2:
            save_dict["y"] = y[:, 0]
            save_dict["y2"] = y[:, 1]

        # Include unit if existing
        if "unit" in signal_data:
            save_dict["unit"] = signal_data["unit"]

        np.savez(file_output_path, **save_dict)
        self.logger.info(f"Successfully saved: {file_output_path}")
