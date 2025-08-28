"""
Dataset exploration and analysis tools for PSG data.
"""

import os
import glob
import numpy as np
from numba import njit

from ..file_handlers import FileHandlerFactory

# Amplitude resolution threshold (2^7)
DIGITAL_SIGNAL_MAX_UNIQUE_VALUES = 128

class Dataset_Explorer:
    """
    A class for exploring and analyzing polysomnography (PSG) datasets.
    
    This class helps discover available files, analyze channel information,
    and determine signal characteristics across different file formats.
    """

    def __init__(self):
        """Initialize the Dataset_Explorer with empty containers."""
        self.psg_fnames = []
        self.ann_fnames = []
        self.ch_names = []
        self.get_channel_types = []
        self.file_factory = FileHandlerFactory()

    def get_files(self, args, psg_ext="*.edf", ann_ext="*.xml", ann_ext2=None):
        """
        Discover and collect PSG signal files and annotation files.
        
        Args:
            args: Arguments object containing data_dir and ann_dir paths
            psg_ext (str): File extension pattern for PSG files (default: "*.edf")
            ann_ext (str): File extension pattern for annotation files (default: "*.xml")  
            ann_ext2 (str, optional): Second annotation file extension pattern
            
        Returns:
            tuple: (psg_filenames, annotation_filenames) arrays
        """
        # Discover PSG signal files
        print(f"Finding all signal files in {os.path.join(args.data_dir, psg_ext)} ...")
        self.psg_fnames = glob.glob(os.path.join(args.data_dir, psg_ext), recursive=True)
        self.psg_fnames.sort()
        
        # Return early if no annotation files needed
        if ann_ext is None:
            return self.psg_fnames, None
        
        # Discover annotation files
        print(f"Finding all annotation files in {os.path.join(args.ann_dir, ann_ext)} ...")
        self.ann_fnames = glob.glob(os.path.join(args.ann_dir, ann_ext), recursive=True)
        
        # Add second annotation extension if provided
        if ann_ext2:
            ann_fnames2 = glob.glob(os.path.join(args.ann_dir, ann_ext2), recursive=True)
            self.ann_fnames.extend(ann_fnames2)

        self.ann_fnames.sort()

        # Convert to numpy arrays for consistency
        self.psg_fnames = np.asarray(self.psg_fnames)
        self.ann_fnames = np.asarray(self.ann_fnames)
        
        # Validate that we have matching numbers of files
        if ann_ext != "":
            assert len(self.ann_fnames) == len(self.psg_fnames), (
                f"\nAnnotation files: {len(self.ann_fnames)} "
                f"\nPSG files: {len(self.psg_fnames)} "
                f"\n-> Counts don't match"
            )

        return self.psg_fnames, self.ann_fnames

    def get_all_channels(self):
        """
        Discover all available channel names and frequencies across all PSG files.
        
        Supports multiple file formats: EDF, H5, CSV, and WFDB.
        
        Returns:
            set: Set of tuples containing (channel_name, frequency) pairs for EDF files,
                 or just channel names for other formats.
        """
        print("Getting all available channel names ...")
        
        self.ch_names = set()
        
        for psg_fname in self.psg_fnames:
            handler = self.file_factory.get_handler(psg_fname)
            
            if handler:
                channels = handler.get_channels(psg_fname)
                
                # Handle different return types
                if handler.file_extension == ".edf":
                    # EDF returns (label, freq) tuples
                    self.ch_names.update(channels)
                else:
                    # Other formats return just labels
                    self.ch_names.update(channels)
                    
                print(f"{os.path.basename(psg_fname)}: {channels}")
            else:
                print(f"Unsupported file format for {psg_fname}")

        return self.ch_names

    def get_channel_type(self):
        """
        Analyze channels to determine if they contain digital or analog signals.
        
        Returns:
            dict: Dictionary with 'analog' and 'digital' keys containing lists of channels
        """
        print("Analyzing channel types (digital vs analog)...")
        
        channel_types = {'analog': [], 'digital': []}
        
        for channel in self.ch_names:
            try:
                print(f"Analyzing channel: {channel}")
                print("=" * 50)
                
                # Check multiple files to determine channel type
                is_analog_found = False
                
                for psg_fname in self.psg_fnames:
                    handler = self.file_factory.get_handler(psg_fname)
                    
                    if not handler:
                        continue
                    
                    signal = handler.read_signal(psg_fname, channel)
                    
                    if signal is None:
                        continue  # Channel not found in this file
                    
                    print(f"Checking {os.path.basename(psg_fname)}")
                    
                    if not self._is_digital(signal):
                        # If any file shows analog signal, classify as analog
                        channel_types["analog"].append(channel)
                        is_analog_found = True
                        break
                
                # If no analog signals found, classify as digital
                if not is_analog_found:
                    channel_types["digital"].append(channel)
                    
            except KeyboardInterrupt:
                # Handle user interruption gracefully
                if channel not in channel_types["analog"]:
                    channel_types["digital"].append(channel)
                continue
            except Exception as e:
                print(f"Error analyzing channel {channel}: {e}")
                channel_types["digital"].append(channel)  # Default to digital on error

        return channel_types

    @staticmethod
    @njit
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
        # Quick check: if signal is constant, it's likely digital
        if signal.max() == signal.min():
            return True

        # Count unique values efficiently
        unique_values = set()
        for value in signal:
            unique_values.add(value)
            if len(unique_values) > DIGITAL_SIGNAL_MAX_UNIQUE_VALUES:
                return False  # Too many unique values, likely analog
        
        print(f"Number of unique values in signal: {len(unique_values)}")
        return True
