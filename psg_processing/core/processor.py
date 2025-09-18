"""
Main dataset processor for PSG data preparation.
"""

import os
import re
import numpy as np
from pathlib import Path
import logging

from ..file_handlers import FileHandlerFactory
from ..utils.logging_manager import LoggingManager
from .signal_processor import SignalProcessor
from .dataset_explorer import Dataset_Explorer

# Sleep stage labels mapping
STAGE_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "MOVE": 5, "UNK": 6}


class DatasetProcessor:
    """
    Main processor for PSG dataset preparation and signal processing.

    This class orchestrates the entire dataset processing pipeline,
    handling file processing, signal cleaning, and output generation.
    """

    def __init__(self, overwrite=False):
        self.logging_manager = LoggingManager(level=logging.INFO)
        self.logger = None
        self.overwrite = overwrite

    def prepare_files(
        self,
        dataset_processor,
        data_dir,
        ann_dir,
        output_dir,
        resample,
        epoch_duration=30,
    ):
        """
        Main function to prepare dataset files for processing.

        Args:
            dataset_processor: Instance of BaseDatasetProcessor containing all dataset-specific logic
            data_dir: PSG data directory
            ann_dir: PSG Annotation directory
            output_dir: Output directory for processed files
            resample: Resample frequency (given in Hz, or None)
            epoch_duration: Duration of each epoch in seconds (default: 30)
        """
        try:
            # Set up logger and initialize components
            self.logger = self.logging_manager.setup_logger(output_dir)
            file_factory = FileHandlerFactory(dataset_processor.dataset_name)

            # Get files using dataset-specific extensions
            explorer = Dataset_Explorer(
                self.logger, dataset_processor.dataset_name, data_dir, ann_dir, **dataset_processor.file_extensions
            )
            psg_fnames, ann_fnames = explorer.get_files()

            # Process each file
            for i, psg_fname in enumerate(psg_fnames):
                print(f"\n--- Processing file {i+1}/{len(psg_fnames)} ---")
                self._process_single_file(
                    psg_fname,
                    ann_fnames[i] if ann_fnames is not None else None,
                    data_dir,
                    output_dir,
                    resample,
                    dataset_processor,
                    file_factory,
                    epoch_duration,
                )

            # Finalize processing
            self.logging_manager.cleanup_file_handlers(self.logger)
            self.logger.info("\n" + "=" * 60)
            self.logger.info("DATASET PREPARATION COMPLETED")
        except KeyboardInterrupt:
            self.logging_manager.cleanup_file_handlers(self.logger)
            self.logger.info("Stopped processing")

    def _process_single_file(
        self,
        psg_fname,
        ann_fname,
        data_dir,
        output_dir,
        resample,
        dataset_processor,
        file_factory,
        epoch_duration,
    ):
        """Process a single PSG file for all specified channels."""

        # Get file handler and validate format
        handler = file_factory.get_handler(self.logger, psg_fname)
        if not handler:
            self.logger.warning(f"Unsupported file format: {psg_fname}")
            return

        # Load annotations before (same for all channels)
        ann_stage_events, ann_Startdatetime = dataset_processor.ann_parse(
            ann_fname, epoch_duration
        )

        if ann_stage_events == []:
            return

        # Process each channel for this file
        for channel in dataset_processor.channel_names:
            self._process_single_channel(
                psg_fname,
                ann_fname,
                channel,
                handler,
                data_dir,
                output_dir,
                resample,
                dataset_processor,
                ann_stage_events,
                ann_Startdatetime,
                epoch_duration,
            )

    def _process_single_channel(
        self,
        psg_fname,
        ann_fname,
        channel,
        handler,
        data_dir,
        output_dir,
        resample,
        dataset_processor,
        ann_stage_events,
        ann_Startdatetime,
        epoch_duration,
    ):
        """Process a single channel from a single file."""

        # Setup channel processing environment
        ch_type = self._get_channel_type(channel, dataset_processor.channel_types)
        output_dir, filename = self._setup_channel_output(
            dataset_processor.keep_folder_structure,
            channel,
            psg_fname,
            data_dir,
            output_dir,
            dataset_processor.alias_mapping,
        )

        # Skip if file already exists
        if not self.overwrite and self._output_file_exists(output_dir, filename):
            return

        # Setup logging for this channel
        self.logging_manager.setup_channel_file_logging(self.logger, output_dir)

        self.logger.info(f"Signal file: {psg_fname}")

        # Extract and process signal
        signal_data = self._extract_signal_data(
            handler,
            psg_fname,
            channel,
            epoch_duration,
            ann_stage_events,
            ann_Startdatetime,
        )

        if signal_data is None:
            return

        signal_data["psg_fname"] = psg_fname
        signal_data["ann_fname"] = ann_fname

        # Process the signal (resample, filter, clean)
        processed_data = self._process_signal_data(
            signal_data, channel, ch_type, resample, dataset_processor, epoch_duration
        )

        if processed_data is None:
            return
        
        # replace x,y and n_epochs after the processing
        for key in processed_data:
            signal_data[key] = processed_data[key]

        # Save processed data
        self._save_processed_data(
            signal_data, output_dir, filename, channel, epoch_duration
        )

        self.logger.info("=" * 40)

    def _get_channel_type(self, channel, channel_types):
        """Get the type (analog/digital) for a specific channel."""
        for ch_type, channels in channel_types.items():
            if channel in channels:
                return ch_type
        raise Exception

    def _setup_channel_output(
        self,
        keep_folder_structure,
        channel,
        psg_fname,
        data_dir,
        output_dir,
        alias_mapping,
    ):
        """Setup output directory and filename for a channel."""

        # Handle channel name aliasing
        ch_name_path = channel
        if alias_mapping:
            alias_checking = [
                key for key, aliases in alias_mapping.items() if channel in aliases
            ]
            if alias_checking:
                ch_name_path = alias_checking[0]

        # replace slash in folder names to avoid nester output structure
        ch_name_path = re.sub(r"[\/]", "_", ch_name_path)

        # Create output directory
        if keep_folder_structure:
            relative_path = os.path.split(Path(psg_fname).relative_to(data_dir))[0]
        else:
            relative_path = ""

        output_dir = os.path.join(output_dir, relative_path, "npz", ch_name_path)
        os.makedirs(output_dir, exist_ok=True)

        # Generate safe filename
        base_filename = os.path.splitext(os.path.basename(psg_fname))[0] + ".npz"
        ch_name_safe = re.sub(r"[^a-zA-Z0-9._\-\s]", "_", channel)
        filename = f"{ch_name_safe}_{base_filename}"

        return output_dir, filename

    def _output_file_exists(self, output_dir, filename):
        """Check if output file already exists."""
        full_path = os.path.join(output_dir, filename)
        if os.path.exists(full_path):
            print(f"File already exists: {full_path}")
            return True
        return False

    def _extract_signal_data(
        self,
        handler,
        psg_fname,
        channel,
        epoch_duration,
        ann_stage_events,
        ann_Startdatetime,
    ):
        """Extract signal data using the appropriate handler."""

        signal_data = handler.get_signal_data(
            psg_fname, epoch_duration, channel
        )

        if signal_data is None:
            return None

        # Handle start datetime fallback (take time from annotation file)
        if signal_data["start_datetime"] is None:
            signal_data["start_datetime"] = ann_Startdatetime

        # Add pre-loaded annotations
        signal_data["ann_stage_events"] = ann_stage_events

        self.logger.info(f"Start datetime: {signal_data['start_datetime']}")
        self.logger.info(
            f"File duration: {signal_data['file_duration']} sec, {signal_data['file_duration']/3600:.2f} h"
        )

        return signal_data

    def _process_signal_data(
        self,
        signal_data,
        channel,
        ch_type,
        resample_freq,
        dataset_processor,
        epoch_duration,
    ):
        """Process signal data through the complete pipeline."""

        signal = signal_data["signal"]
        sampling_rate = signal_data["sampling_rate"]
        n_epoch_samples = signal_data["n_epoch_samples"]

        channel_groups = dataset_processor.channel_groups
        ann_label = dataset_processor.ann_label

        # Check signal length
        if len(signal) // n_epoch_samples <= 1:
            self.logger.info(f"Signal too short, only {len(signal)} samples")
            return None

        if resample_freq != "None":
            # Resample and filter
            signal_processor = SignalProcessor(self.logger)
            signal, sampling_rate = signal_processor.resample_filter_signal(
                signal, channel, ch_type, channel_groups, sampling_rate, resample_freq
            )

        # Reshape into epochs
        n_epoch_samples = int(epoch_duration * sampling_rate)
        n_epochs = len(signal) // n_epoch_samples
        signals = signal[0 : n_epochs * epoch_duration * sampling_rate].reshape(-1, n_epoch_samples)

        if resample_freq != "None":
            # zero pad last eventually not full epoch
            last_epoch = signals[n_epochs * epoch_duration * sampling_rate:]
            if last_epoch:
                n_last_epoch = len(last_epoch)
                last_epoch = np.pad(last_epoch,pad_width=n_epoch_samples-n_last_epoch,constant_values=0)
                signals = np.append(signals, last_epoch, axis=0)

        # Generate labels 
        labels = ann_label(self.logger, signal_data["ann_stage_events"], epoch_duration)

        if resample_freq != "None":
            # Align labels (some datasets handle different length of signal and label data)
            signals, labels = dataset_processor.alignment(
                self.logger,
                signal_data["psg_fname"],
                signal_data["ann_fname"],
                signals,
                labels,
            )

        x, y = signals.astype(np.float32), labels.astype(np.int32)

        if resample_freq != "None":
            # Clean signal data
            x, y = self._clean_signal(x, y, STAGE_DICT)

        if x is None:
            return None

        return {
            "x": x,
            "y": y,
            "sampling_rate": sampling_rate,
            "n_all_epochs": n_epochs,
        }

    def _clean_signal(self, x, y, stage_dict):
        """
        Clean signal by removing movement/unknown epochs and selecting sleep periods.

        Args:
            x: Signal epochs array
            y: Stage labels array
            stage_dict: Dictionary mapping stage names to integers

        Returns:
            Tuple of (cleaned_x, cleaned_y) or (None, None) if no sleep detected
        """
        if self.logger:
            self.logger.info(
                f"Starting signal cleaning - Input shape: x={x.shape}, y={y.shape}"
            )

        # Remove movement and unknown epochs
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]

        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            if self.logger:
                self.logger.info("Removing irrelevant stages:")
                if len(move_idx) > 0:
                    self.logger.info(f"  Movement epochs: {len(move_idx)}")
                if len(unk_idx) > 0:
                    self.logger.info(f"  Unknown epochs: {len(unk_idx)}")
                self.logger.info(f"  Total epochs to remove: {len(remove_idx)}")
                self.logger.info(f"  Data before removal: {x.shape}, {y.shape}")

            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]

        if self.logger:
            self.logger.info(f"  Data after removal: {x.shape}, {y.shape}")

        # Select only sleep periods (30 min buffer around sleep)
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]

        if len(nw_idx) == 0:
            if self.logger:
                self.logger.warning("File contains no sleep stages (only Wake)")
            return None, None

        # Calculate sleep period boundaries with buffer
        start_idx = max(0, nw_idx[0] - (w_edge_mins * 2))
        end_idx = min(len(y) - 1, nw_idx[-1] + (w_edge_mins * 2))

        select_idx = np.arange(start_idx, end_idx + 1)

        if self.logger:
            self.logger.info(f"  Data before sleep selection: {x.shape}, {y.shape}")

        x = x[select_idx]
        y = y[select_idx]

        if self.logger:
            self.logger.info(f"  Data after sleep selection: {x.shape}, {y.shape}")

        return x, y

    def _save_processed_data(
        self, signal_data, output_dir, filename, channel, epoch_duration
    ):
        """Save processed data to file."""

        save_dict = {
            "x": signal_data["x"],
            "fs": signal_data["sampling_rate"],
            "ch_label": channel,
            "start_datetime": signal_data["start_datetime"],
            "file_duration": signal_data["file_duration"],
            "epoch_duration": epoch_duration,
            "n_all_epochs": signal_data["n_all_epochs"],
            "n_epochs": len(signal_data["x"]),
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

        output_path = os.path.join(output_dir, filename)
        
        np.savez(output_path, **save_dict)
        self.logger.info(f"Successfully saved: {filename}")
