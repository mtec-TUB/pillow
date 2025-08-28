"""
Main dataset processor for PSG data preparation.
"""

import os
import re
import numpy as np

from ..file_handlers import FileHandlerFactory
from ..utils.logging_manager import LoggingManager
from .signal_processor import SignalProcessor

# Sleep stage labels mapping
STAGE_DICT = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "MOVE": 5,
    "UNK": 6
}

class DatasetProcessor:
    """
    Main processor for PSG dataset preparation and signal processing.
    
    This class orchestrates the entire dataset processing pipeline,
    handling file processing, signal cleaning, and output generation.
    """
    
    def __init__(self):
        self.logging_manager = LoggingManager()
        self.logger = None
        
    def prepare_files(self, args, channels, channel_types, psg_fnames, ann_fnames, 
                     ann_parse, ann_label, get_filter_freq, alias_mapping=None, epoch_duration=30):
        """
        Main function to prepare dataset files for processing.
        
        Args:
            args: Command line arguments
            channels: List of channel names to process
            channel_types: Dictionary mapping channels to types (analog/digital)
            psg_fnames: List of signal file paths
            ann_fnames: List of annotation file paths
            ann_parse: Function to parse annotations
            ann_label: Function to generate labels
            get_filter_freq: Function to get filter frequencies
            alias_mapping: Optional channel name mapping
            epoch_duration: Duration of each epoch in seconds (default: 30)
        """        
        # Set up logger and initialize components
        self.logger = self.logging_manager.setup_logger()
        file_factory = FileHandlerFactory()
        
        # Process each file
        for i, psg_fname in enumerate(psg_fnames):
            self._process_single_file(
                i, psg_fname, ann_fnames[i] if ann_fnames is not None else None,
                args, channels, channel_types, ann_parse, ann_label, 
                get_filter_freq, file_factory, alias_mapping, epoch_duration
            )
        
        # Finalize processing
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DATASET PREPARATION COMPLETED")
        self.logging_manager.cleanup_file_handlers(self.logger)
    
    def _process_single_file(self, file_index, psg_fname, ann_fname, args, channels, 
                           channel_types, ann_parse, ann_label, get_filter_freq, 
                           file_factory, alias_mapping, epoch_duration):
        """Process a single PSG file for all specified channels."""
        
        # Get file handler and validate format
        handler = file_factory.get_handler(psg_fname)
        if not handler:
            self.logger.warning(f"Unsupported file format: {psg_fname}")
            return
        
        # Load annotations before (same for all channels)
        ann_stage_events, ann_Startdatetime = ann_parse(ann_fname, epoch_duration)

        if ann_stage_events == []:
            return
        
        # Process each channel for this file
        for channel in channels:
            self._process_single_channel(
                file_index, psg_fname, channel, handler, args,
                channel_types, ann_stage_events, ann_Startdatetime,
                ann_label, get_filter_freq, alias_mapping, epoch_duration
            )
    
    def _process_single_channel(self, file_index, psg_fname, channel, handler, args,
                              channel_types, ann_stage_events, ann_Startdatetime,
                              ann_label, get_filter_freq, alias_mapping, epoch_duration):
        """Process a single channel from a single file."""
        
        # Setup channel processing environment
        ch_type = self._get_channel_type(channel, channel_types)
        output_dir, filename = self._setup_channel_output(
            channel, psg_fname, args, alias_mapping
        )
        
        # Skip if file already exists
        if self._output_file_exists(output_dir, filename):
            return
        
        # Setup logging for this channel
        self.logging_manager.setup_channel_file_logging(
            self.logger, output_dir, args.log_file
        )
        
        self.logger.info(f"\n--- Processing file {file_index+1} ---")
        self.logger.info(f"Signal file: {psg_fname}")
        
        try:
            # Extract and process signal
            signal_data = self._extract_signal_data(
                handler, psg_fname, channel, epoch_duration, ann_stage_events, ann_Startdatetime
            )
            
            if signal_data is None:
                return
            
            # Process the signal (resample, filter, clean)
            processed_data = self._process_signal_data(
                signal_data, channel, ch_type, args.resample, 
                get_filter_freq, ann_label, epoch_duration
            )
            
            if processed_data is None:
                return
            
            # Save processed data
            self._save_processed_data(
                processed_data, output_dir, filename, channel, epoch_duration
            )
            
        except (ValueError, KeyboardInterrupt):
            raise
        except Exception as e:
            self.logger.error(f"Error processing {channel} in {psg_fname}: {e}")
            return
        
        self.logger.info("=" * 40)
    
    def _get_channel_type(self, channel, channel_types):
        """Get the type (analog/digital) for a specific channel."""
        for ch_type, channels in channel_types.items():
            if channel in channels:
                return ch_type
        return "digital"  # Default fallback
    
    def _setup_channel_output(self, channel, psg_fname, args, alias_mapping):
        """Setup output directory and filename for a channel."""
        
        # Handle channel name aliasing
        ch_name_path = channel
        if alias_mapping:
            alias_checking = [key for key, aliases in alias_mapping.items() if channel in aliases]
            if alias_checking:
                ch_name_path = alias_checking[0]
        
        # Create output directory
        output_dir = os.path.join(args.output_dir, ch_name_path)
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
    
    def _extract_signal_data(self, handler, psg_fname, channel, epoch_duration, 
                           ann_stage_events, ann_Startdatetime):
        """Extract signal data using the appropriate handler."""
               
        signal_info = handler.get_signal_info(self.logger, psg_fname, epoch_duration, channel)
        
        if signal_info is None:
            return None
        
        # Handle start datetime fallback
        if signal_info['start_datetime'] is None:
            signal_info['start_datetime'] = ann_Startdatetime
        
        # Add pre-loaded annotations
        signal_info['ann_stage_events'] = ann_stage_events
        
        self.logger.info(f"Start datetime: {signal_info['start_datetime']}")
        self.logger.info(f"File duration: {signal_info['file_duration']} sec, {signal_info['file_duration']/3600:.2f} h")
        
        return signal_info
    
    def _process_signal_data(self, signal_data, channel, ch_type, resample_freq, 
                           get_filter_freq, ann_label, epoch_duration):
        """Process signal data through the complete pipeline."""
        
        signal = signal_data['signal']
        sampling_rate = signal_data['sampling_rate']
        n_epoch_samples = signal_data['n_epoch_samples']
        
        # Check signal length
        n_epochs = len(signal) // n_epoch_samples
        if n_epochs <= 1:
            self.logger.info(f"Signal too short, only {len(signal)} samples")
            return None
        
        # Truncate to whole epochs
        signal = signal[0:n_epochs * epoch_duration * sampling_rate]
        
        # Resample and filter
        signal_processor = SignalProcessor(self.logger)
        signal, sampling_rate = signal_processor.resample_filter_signal(
            signal, channel, ch_type, sampling_rate, resample_freq, get_filter_freq
        )
        
        # Reshape into epochs
        n_epoch_samples = int(epoch_duration * sampling_rate)
        n_epochs = len(signal) // n_epoch_samples
        signals = signal.reshape(-1, n_epoch_samples)
        
        # Generate and align labels
        labels = ann_label(self.logger, signal_data['ann_stage_events'], epoch_duration)
        signals, labels = self._align_signals_and_labels(signals, labels, signal_data['file_duration'])
        
        # Clean signal data
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)
        
        assert len(x) == len(y), f"Length mismatch: signal={len(x)}, labels={len(y)}"
        
        x, y = self._clean_signal(x, y, STAGE_DICT)
        if x is None:
            return None
        
        return {
            'x': x,
            'y': y,
            'sampling_rate': sampling_rate,
            'start_datetime': signal_data['start_datetime'],
            'file_duration': signal_data['file_duration'],
            'n_all_epochs': n_epochs
        }
    
    def _align_signals_and_labels(self, signals, labels, file_duration):
        """Align signals and labels, handling dataset-specific issues."""
        
        # Remove annotations longer than signals
        labels = labels[:len(signals)]
        
        # Handle dataset-specific label/signal mismatches
        # Note: This logic could be moved to dataset-specific handlers in the future
        if len(labels) < len(signals):
            signals = signals[:len(labels)]
        
        return signals, labels
    
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
            self.logger.info(f"Starting signal cleaning - Input shape: x={x.shape}, y={y.shape}")
        
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
            self.logger.info(f"  Data before selection: {x.shape}, {y.shape}")
        
        x = x[select_idx]
        y = y[select_idx]
        
        if self.logger:
            self.logger.info(f"  Data after selection: {x.shape}, {y.shape}")
        
        return x, y

    def _save_processed_data(self, processed_data, output_dir, filename, channel, epoch_duration):
        """Save processed data to file."""
        
        save_dict = {
            "x": processed_data['x'],
            "fs": processed_data['sampling_rate'],
            "ch_label": channel,
            "start_datetime": processed_data['start_datetime'],
            "file_duration": processed_data['file_duration'],
            "epoch_duration": epoch_duration,
            "n_all_epochs": processed_data['n_all_epochs'],
            "n_epochs": len(processed_data['x']),
        }
        
        # Handle multiple scorers
        y = processed_data['y']
        if y.ndim == 1:
            save_dict["y"] = y
        elif y.ndim == 2:
            save_dict["y"] = y[:, 0]
            save_dict["y2"] = y[:, 1]
        
        output_path = os.path.join(output_dir, filename)
        np.savez(output_path, **save_dict)
        self.logger.info(f"Successfully saved: {filename}")
