"""
Signal processing utilities for PSG data.
"""

from typing import Dict, List, Union

import numpy as np
from scipy.interpolate import interp1d
from mne.filter import resample, filter_data


class SignalProcessor:
    """
    A class for processing PSG signals including resampling and filtering operations.
    
    This class handles both digital and analog signal processing with appropriate
    methods for each signal type, preserving important signal characteristics
    during processing.
    """
    
    def __init__(self, logger):
        """
        Initialize the SignalProcessor.
        
        Args:
            logger: Logger instance for operation tracking
        """
        self.logger = logger
        
    # Standard filter frequency mappings for different signal types
    FILTER_FREQUENCIES = {
        'eeg_eog': [0.3, 35],      # EEG and EOG channels: 0.3-35 Hz
        'emg': [10, None],     # EMG channels: 10+ Hz (high-pass)
        'ecg': [0.3, None],    # ECG channels: 0.3+ Hz (high-pass)
        'respiratory': [0.1, 15],  # Respiratory signals: 0.1-15 Hz
        'nasal_pressure': [0.03, None],  # Nasal pressure: 0.03+ Hz (high-pass)
        'snoring': [10, None], # Snoring: 10+ Hz (high-pass)
        'default': [None, None]  # Default: no filtering
    }
        
    def get_filt_freq(self, ch_name: str, channel_groups: Dict[str, List[str]]) -> List[Union[float, None]]:
        """
        Get filter frequencies for a given channel using the centralized mapping.
        
        Args:
            ch_name: Channel name
            
        Returns:
            List of [low_freq, high_freq] where None means no filtering
        """
        # Look up which group this channel belongs to
        for group_name, channels in channel_groups.items():
            if ch_name in channels:
                return self.FILTER_FREQUENCIES.get(group_name, self.FILTER_FREQUENCIES['default'])
        
        # If channel not found in any group, return default (no filtering)
        return self.FILTER_FREQUENCIES['default']

    def resample_filter_signal(self, signal, select_ch, ch_type, channel_groups, sampling_rate, resample_freq):
        """
        Resample and filter signal based on channel type and parameters.
        
        Args:
            signal: Input signal data
            select_ch: Channel name
            ch_type: Channel type ('analog' or 'digital')
            sampling_rate: Current sampling rate
            resample_freq: Target sampling rate
            get_filter_freq: Function to get filter frequencies
            
        Returns:
            tuple: (processed_signal, final_sampling_rate)
        """
        self.logger.info(f"Sample rate before: {sampling_rate}")

        # perform resampling if necessary
        if (resample_freq != 'None'):
            resample_freq = int(resample_freq)

            if resample_freq != sampling_rate:

                if ch_type == "digital":
                    signal = self.resample_dig(signal, sampling_rate, resample_freq)

                elif ch_type == "analog":
                    signal = self.resample_ana(signal, sampling_rate, resample_freq)

            # Filter signal according to AASM Manual
            [low, high] = self.get_filt_freq(select_ch, channel_groups)
            if not (low is None and high is None):
                self.logger.info(f"Filter signal with low: {low} Hz and high: {high} Hz bandpass")
                if (high is None or sampling_rate >= high*2) and (ch_type == "analog"):
                    signal = filter_data(signal, resample_freq, low, high, method='fir',verbose="WARNING")
                elif (sampling_rate < high*2) and (ch_type == "analog"):
                    signal = filter_data(signal, resample_freq, low, sampling_rate/2, method='fir',verbose="WARNING")

            sampling_rate = resample_freq

            self.logger.info(f"Sample rate after: {sampling_rate}")

        return signal, sampling_rate


    def resample_dig(self, signal, input_rate, output_rate):
        """
        Resample digital signal using 1D interpolation with nearest neighbor.
        
        Args:
            signal: Input digital signal
            input_rate: Current sampling rate
            output_rate: Target sampling rate
            
        Returns:
            numpy.ndarray: Resampled signal
        """
        input_times = np.arange(0, len(signal), 1)
        output_times = np.arange(0, len(signal), input_rate/output_rate)
        f = interp1d(input_times, signal, kind="nearest", fill_value="extrapolate")
        return f(output_times)

    def resample_ana(self, signal, input_rate, output_rate):
        """
        Resample analog signal using polyphase filtering with clipping preservation.
        
        This method preserves clipping artifacts that may be important for signal analysis
        while using high-quality polyphase resampling for the main signal content.
        
        Args:
            signal: Input analog signal
            input_rate: Current sampling rate
            output_rate: Target sampling rate
            
        Returns:
            numpy.ndarray: Resampled signal with preserved clipping characteristics
        """
        # Find clipping thresholds
        signal_min, signal_max = np.min(signal), np.max(signal)

        # Create clipping masks
        clipping_mask_min = signal == signal_min
        clipping_mask_max = signal == signal_max
        clipping_mask = clipping_mask_min | clipping_mask_max
        
        # Remove isolated clipping points (likely not true clipping)
        lonely_clippers = (np.diff(clipping_mask.astype(np.int32), n=2, append=[0, 0]) == -2)
        indices_lonely_clippers = np.argwhere(lonely_clippers) + 1 
        clipping_mask_min[indices_lonely_clippers] = False
        clipping_mask_max[indices_lonely_clippers] = False

        # Resample clipping masks to match new sampling rate
        resample_factor = input_rate / output_rate
        clipping_mask_min_resampled = np.round(np.argwhere(clipping_mask_min) * resample_factor).astype(int)
        clipping_mask_max_resampled = np.round(np.argwhere(clipping_mask_max) * resample_factor).astype(int)
        
        # Perform high-quality polyphase resampling
        signal_resampled = resample(signal.astype(np.float64), up=output_rate, down=input_rate, method="polyphase")

        # Ensure mask indices are within bounds
        clipping_mask_min_resampled = clipping_mask_min_resampled[clipping_mask_min_resampled < len(signal_resampled)]
        clipping_mask_max_resampled = clipping_mask_max_resampled[clipping_mask_max_resampled < len(signal_resampled)]

        # Restore clipping values to prevent resampling artifacts
        signal_resampled[clipping_mask_min_resampled] = signal_min
        signal_resampled[clipping_mask_max_resampled] = signal_max

        # Final clipping to original signal range
        signal_resampled = np.clip(signal_resampled, a_min=signal_min, a_max=signal_max)

        return signal_resampled
