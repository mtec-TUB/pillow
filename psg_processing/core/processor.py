"""
Main dataset processor for PSG data preparation.
"""

import logging
import os
import re
from math import ceil, floor
from datetime import datetime
from pyedflib import EdfWriter
import h5py
from pathlib import Path
from decimal import Decimal
import numpy as np

from ..utils import LoggingManager
from .dataset_explorer import Dataset_Explorer
from .signal_processor import SignalProcessor


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

    # Sleep stage labels mapping
    STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6}

    def process_files(self):
        
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
            psg_fnames, ann_fnames = explorer.get_files()

            # Process each file
            ann_idx = 0
            for psg_idx, psg_fname in enumerate(psg_fnames):
                print(f"\n--- Processing file {psg_idx+1}/{len(psg_fnames)} ---")
                if ann_fnames is not None and self.config.use_annot:
                    ann_fname, ann_idx = self._find_matching_annotation(
                        psg_fname, ann_fnames, ann_idx)
                    if ann_fname is None:
                        self.logger.warning(
                            f"No matching annotation found for PSG: "
                            f"{Path(psg_fname).relative_to(self.data_dir)}. Skipping file."
                        )
                        continue
                self._process_single_file(
                    psg_fname,
                    ann_fname if ann_fnames is not None else None
                )

            # Finalize processing
            self.logging_manager.cleanup_file_handlers(self.logger)
            self.logger.info("\n" + "=" * 60)
            self.logger.info("DATASET PREPARATION COMPLETED")
        except KeyboardInterrupt:
            self.logging_manager.cleanup_file_handlers(self.logger)
            self.logger.info("Stopped processing")

    def _find_matching_annotation(self, psg_fname, ann_fnames, start_idx):
        """
        Scan annotation files from start_idx forward and return the first match.
        Return (ann_fname, new_index).
        Only works if annotation files are ordered in the same way as PSG files.
        """
        psg_base = str(Path(psg_fname).relative_to(self.config.data_dir))

        for i in range(start_idx, len(ann_fnames)):
            ann_base = str(Path(ann_fnames[i]).relative_to(self.config.ann_dir))
            psg_id, ann_id = self.dataset.get_file_identifier(psg_base, ann_base)  # adapt as needed

            if psg_id == ann_id:
                return ann_fnames[i], i + 1  # move annotation pointer past this match

        return None, start_idx  # no match found

    def _process_single_file(
        self,
        psg_fname,
        ann_fname
    ):
        """Process a single PSG file for all specified channels."""
        
        if self.config.use_annot:
            # Load annotations before (same for all channels)
            ann_stage_events, ann_Startdatetime = self.dataset.ann_parse(ann_fname)

            if ann_stage_events == []:
                return
        else:
            ann_stage_events = []
            ann_Startdatetime = None

        # List channels to process for this file
        channels = list(set(self.dataset.channel_names) & set(self.dataset.psg_file_handler.get_channels(self.logger,psg_fname)))

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

        if self.config.use_annot:
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

            if isinstance(ann_Startdatetime, datetime):
                if signal_data["start_datetime"].time() != ann_Startdatetime.time():
                    start_time = (ann_Startdatetime - signal_data["start_datetime"]).total_seconds()
                else:
                    start_time=0

            elif isinstance(ann_Startdatetime, (int, float, Decimal)):  # ann_Startdatetime can be in seconds or samples (depends on dataset)
                start_time = ann_Startdatetime

                print(
                    f"Start of signal: {signal_data['start_datetime']} \nStart of labels: {ann_Startdatetime}"
                )
             
            if start_time != 0:
                # Shorten signal if annotations start later or align front to first common epoch if annotations start before
                signal, ann_stage_events = self.dataset.align_front(
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
                signal_data["ann_stage_events"] = ann_stage_events

        self.logger.info(f"Start datetime: {signal_data['start_datetime']}")
        
        if self.config.use_annot:
            signal_data["labels"] = self.dataset.ann_label(
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

        # replace signal, labels and sampling_rate after the processing
        for key in processed_data:
            signal_data[key] = processed_data[key]

        # Save processed data
        self._save_processed_data(
            signal_data, file_output_path, channel
        )

        self.logger.info("=" * 40)
    
        return True

    def _get_channel_type(self, channel):
        """Get the type (analog/digital) for a specific channel."""
        for ch_type, channels in self.dataset.channel_types.items():
            if channel in channels:
                return ch_type

        raise Exception(f"channel {channel} not listed in channel_types")

    def _setup_channel_output(
        self,
        channel,
        psg_fname
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

        file_output_dir = os.path.join(self.config.output_dir, relative_path, self.config.output_format, ch_name_path)
        os.makedirs(file_output_dir, exist_ok=True)

        # Generate safe file and folder name
        base_filename = os.path.splitext(os.path.basename(psg_fname))[0] + "." + self.config.output_format
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
        labels = signal_data["labels"]
        sampling_rate = signal_data["sampling_rate"]
        n_epoch_samples = self.config.epoch_duration * sampling_rate
        if not n_epoch_samples.is_integer():
            raise Exception(
                f"Epoch duration {self.config.epoch_duration} sec with sampling rate {sampling_rate} Hz "
                f"does not result in integer number of samples per epoch."
            )
        else:
            n_epoch_samples = int(n_epoch_samples)

        # Check signal length (at least one epoch required)
        n_epochs = int(len(signal) // n_epoch_samples)
        if n_epochs < 1:
            self.logger.info(f"File does not hold at least one epoch, only {len(signal)} samples")
            return None, True

        signal_processor = SignalProcessor(self.logger, self.config.filter_freq)
        if self.config.resample is not None:
            # Resample signal
            signal = signal_processor.resample_signal(
                signal,
                ch_type,
                sampling_rate,
                self.config.resample,
            )
            sampling_rate = self.config.resample
        if self.config.filter:
            # Filter signal according to AASM
            signal = signal_processor.filter_signal(
                signal,
                sampling_rate,
                channel,
                self.dataset.channel_groups,
                ch_type,
            )

        # Reshape into epochs
        print(
            f"Seconds in unfilled (cropped) epoch: {len(signal)/sampling_rate - (n_epochs * self.config.epoch_duration):.4f} sec"
        )
        signal_epoched = signal[:n_epochs * int(self.config.epoch_duration * sampling_rate)].reshape(n_epochs, -1)

        # Not sure about this part
        # if self.config.alignment == Alignment.MATCH_ANNOT:
        #     # zero pad last eventually not full epoch
        #     last_epoch = signal[n_epochs * self.config.epoch_duration * sampling_rate :]
        #     n_last_epoch = len(last_epoch)
        #     if n_last_epoch > 0:
        #         last_epoch = np.pad(
        #             last_epoch,
        #             pad_width=(0, n_epoch_samples - n_last_epoch),
        #             constant_values=0,
        #         ).reshape(1, -1)
        #         signal_epoched = np.append(signal_epoched, last_epoch, axis=0)

        if self.config.use_annot and len(signal_epoched) != len(labels):
            # Align labels (some datasets have different length of signal and annotation data)
            signal_epoched, labels = self.dataset.align_end(
                self.logger,
                self.config.alignment,
                self.config.pad_values,
                signal_data["psg_fname"],
                signal_data["ann_fname"],
                signal_epoched,
                labels,
            )
            
            assert len(signal_epoched) == len(labels), f"Length mismatch: signal ({os.path.basename(signal_data['psg_fname'])})={len(signal_epoched)}, labels({os.path.basename(signal_data['ann_fname'])})={len(labels)} TODO: implement alignment function"
        
        if self.config.use_annot:
            # Clean signal data based on annotations
            signal_epoched, labels = self._clean_signal(signal_epoched, labels)

        if signal_epoched is None:
            return None, False

        signal_epoched, labels = signal_epoched.astype(np.float32), labels.astype(np.int32)

        return {
            "signal": signal_epoched,
            "labels": labels,
            "sampling_rate": sampling_rate
        }, True

    def _clean_signal(self, signal_epoched, labels):
        """
        Clean signal by removing movement/unknown epochs and selecting sleep periods.

        Args:
            signal_epoched: Signal epochs array
            labels: Stage labels array

        Returns:
            Tuple of (cleaned_signals, cleaned_labels) or (None, None) if no sleep detected
        """
        self.logger.info(
            f"Starting signal cleaning - Input shape: signal_epoched={signal_epoched.shape}, labels={labels.shape}"
        )

        # Remove movement and unknown epochs if configured
        if self.config.rm_move:
            move_idx = np.where(labels == self.STAGE_DICT["MOVE"])[0]  
        else:
            move_idx = []
        if len(move_idx) > 0:
            self.logger.info(f"  Removing Movement epochs: {len(move_idx)}")

        if self.config.rm_unk:
            unk_idx = np.where(labels == self.STAGE_DICT["UNK"])[0]
        else:
            unk_idx = []
        if len(unk_idx) > 0:
            self.logger.info(f"  Removing Unknown epochs: {len(unk_idx)}")

        remove_idx = np.union1d(move_idx, unk_idx)

        sleep_idx = np.where(
            (labels != self.STAGE_DICT["W"])
            & (labels != self.STAGE_DICT["MOVE"])
            & (labels != self.STAGE_DICT["UNK"])
        )[0]

        if len(sleep_idx) <= self.config.min_sleep_epochs:
            self.logger.warning("File contains less sleep epochs than required. Skipping")
            return None, None, None

        if self.config.n_wake_epochs == "all":
            start_idx = 0
            end_idx = len(labels) - 1
        else:
            # Remove extensive wake epochs at start and end as given in config
            n_wake_epochs = int(self.config.n_wake_epochs)
            start_idx = max(0, sleep_idx[0] - n_wake_epochs)
            end_idx = min(len(labels) - 1, sleep_idx[-1] + n_wake_epochs)

            if start_idx + (len(signal_epoched)-end_idx)-1 > 0:
                self.logger.info(
                    f"  Outside {int(self.config.n_wake_epochs)/2}min wake epochs: {start_idx + (len(signal_epoched)-end_idx)-1}"
                )

        select_idx = np.setdiff1d(np.arange(start_idx, end_idx + 1), remove_idx)

        self.logger.info(f"  Total epochs to remove: {len(signal_epoched) - len(select_idx)}")

        signal_epoched = signal_epoched[select_idx]
        labels = labels[select_idx]

        self.logger.info(f"  Data after cleaning: {signal_epoched.shape}, {labels.shape}")

        return signal_epoched, labels

    def _save_processed_data(
        self, signal_data, file_output_path, channel
    ):
        """Save processed data to file."""
        
        if self.config.output_format == "npz":
            save_dict = {
                "x": signal_data["signal"],
                "fs": signal_data["sampling_rate"],
                "ch_label": channel,
                "file_duration": len(signal_data["signal"]) * self.config.epoch_duration,
                "epoch_duration": self.config.epoch_duration,
                "n_epochs": len(signal_data["signal"]),    # after cleaning
            }

            # Handle multiple scorers
            labels = signal_data["labels"]
            if labels.ndim == 1:
                save_dict["y"] = labels
            elif labels.ndim == 2:
                save_dict["y"] = labels[:, 0]
                save_dict["y2"] = labels[:, 1]

            # Include unit if existing
            if "unit" in signal_data:
                save_dict["unit"] = signal_data["unit"]

            np.savez(file_output_path, **save_dict)
            
        elif self.config.output_format == "edf":
            signal = signal_data["signal"].flatten()
            
            with EdfWriter(file_output_path, n_channels=1) as edf_writer:
                scale = 10**3   # to get 3 decimals for physical min and max
                channel_info = {
                    'label': channel,
                    'dimension': signal_data["unit"] if "unit" in signal_data else "a.u.",
                    'sample_frequency': signal_data["sampling_rate"],
                    'physical_min': floor(np.min(signal) *  scale)/scale,
                    'physical_max': ceil(np.max(signal) * scale)/scale,
                    'digital_min': -32768,
                    'digital_max': 32767,
                    'transducer': '',
                    'prefilter': ''
                }
                edf_writer.setSignalHeader(0, channel_info)
                if isinstance(signal_data["start_datetime"], datetime):
                    edf_writer.setStartdatetime(signal_data["start_datetime"])
                else:
                    edf_writer.setStartdatetime(datetime(1985, 1, 1, 0, 0, 0))
                edf_writer.update_header()

                edf_writer.writeSamples(signal.reshape(1,-1))

                duration=self.config.epoch_duration
                for epoch_idx, label in enumerate(signal_data["labels"]):
                    edf_writer.writeAnnotation(
                        onset_in_seconds=epoch_idx * duration,
                        duration_in_seconds=duration,
                        description=str(label)
                    )
        elif self.config.output_format == "hdf5":
            raise NotImplementedError("HDF5 output not yet implemented")
            # with h5py.File(file_output_path, 'w') as h5f:
            #     h5f.create_dataset('x', data=signal_data["signal"], compression="gzip")
            #     h5f.attrs['sampling_rate'] = signal_data["sampling_rate"]
            #     h5f.attrs['channel'] = channel
            #     h5f.attrs['epoch_duration'] = self.config.epoch_duration
            #     h5f.attrs['n_epochs'] = len(signal_data["signal"])  # after cleaning

            #     # Handle multiple scorers
            #     labels = signal_data["y"]
            #     if labels.ndim == 1:
            #         h5f.create_dataset('y', data=labels, compression="gzip")
            #     elif labels.ndim == 2:
            #         h5f.create_dataset('y', data=labels[:, 0], compression="gzip")
            #         h5f.create_dataset('y2', data=labels[:, 1], compression="gzip")
        
        
        self.logger.info(f"Successfully saved: {file_output_path}")