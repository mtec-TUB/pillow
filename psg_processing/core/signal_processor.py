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
        self.signal_min = None
        self.signal_max = None

    # Standard filter frequency mappings for different signal groups
    FILTER_FREQUENCIES = {
        "eeg_eog": [0.3, 35],  # EEG and EOG channels: 0.3-35 Hz
        "emg": [10, None],  # EMG channels: 10+ Hz (high-pass)
        "ecg": [0.3, None],  # ECG channels: 0.3+ Hz (high-pass)
        "thoraco_abdo_resp": [0.1, 15],  # thoraco_abdo_resp signals: 0.1-15 Hz
        "nasal_pressure": [0.03, None],  # Nasal pressure: 0.03+ Hz (high-pass)
        "snoring": [10, None],  # Snoring: 10+ Hz (high-pass)
        "default": [None, None],  # Default: no filtering
    }        

    def get_filt_freq(
        self, ch_name: str, channel_groups: Dict[str, List[str]]
    ) -> tuple[str, List[Union[float, None]]]:
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
                return group_name, self.FILTER_FREQUENCIES.get(group_name)

        # If channel not found in any group, return default (no filtering)
        return "default", self.FILTER_FREQUENCIES["default"]

    def resample_filter_signal(
        self, signal, select_ch, ch_type, channel_groups, sampling_rate, resample_freq
    ):
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

        resample_freq = int(resample_freq)

        channel_group, [low, high] = self.get_filt_freq(select_ch, channel_groups)

        # center signals, that will be filtered later on, around zero
        if channel_group != "default":
            signal = signal - np.mean(signal)
        
        # Store clipping threshold
        self.signal_min, self.signal_max = np.min(signal), np.max(signal)

        # if fs not already desired resample fs -> resample
        if resample_freq != sampling_rate:

            if ch_type == "digital":
                signal = self.resample_dig(signal, sampling_rate, resample_freq)

            elif ch_type == "analog":
                signal = self.resample_ana(signal, sampling_rate, resample_freq)

        # Filter signal according to AASM Manual
        if not (low is None and high is None) and (ch_type == "analog"):
                
            self.logger.info(
                f"Filter signal with low: {low} Hz and high: {high} Hz bandpass"
            )
            
            signal = filter_data(
                signal,
                resample_freq,
                low,
                high if (high and high < resample_freq/2) else None,
                method="fir",
                verbose="WARNING",
            )

        # Final clipping to original signal range
        signal = np.clip(signal, a_min=self.signal_min, a_max=self.signal_max)

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
        output_times = np.arange(0, len(signal), input_rate / output_rate)
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
        # Perform high-quality polyphase resampling
        signal_resampled = resample(
            signal.astype(np.float64),
            up=output_rate,
            down=input_rate,
            method="polyphase",
        )

        return signal_resampled
