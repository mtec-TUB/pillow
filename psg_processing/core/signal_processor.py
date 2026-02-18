"""
Signal processing utilities for PSG data.
"""

from typing import Dict, List, Union

import numpy as np
from scipy.interpolate import interp1d
from mne.filter import resample, filter_data
import cupy


class SignalProcessor:
    """
    A class for processing PSG signals including resampling and filtering operations.

    This class handles both digital and analog signal processing with appropriate
    methods for each signal type, preserving important signal characteristics
    during processing.
    """

    def __init__(self, logger, ch_name, filter_freq, ch_types):
        """
        Initialize the SignalProcessor.

        Args:
            logger: Logger instance for operation tracking
        """
        self.logger = logger
        self.signal_min = None
        self.signal_max = None
        self.filter_freq = filter_freq
        self.select_ch = ch_name
        self.ch_type = self._get_channel_type(ch_name, ch_types)

    def get_filt_freq(
        self, ch_name: str, channel_groups: Dict[str, List[str]]
    ) -> List[Union[float, None]]:
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
                return self.filter_freq.get(group_name)

        # If channel not found in any group, return default (no filtering)
        return self.filter_freq["default"]
    
    def _get_channel_type(self, channel, channel_types):
        """Get the type (analog/digital) for a specific channel."""
        for ch_type, channels in channel_types.items():
            if channel in channels:
                return ch_type

        raise Exception(f"channel {channel} not listed in channel_types")

    def resample_signal(self, signal, sampling_rate, resample_freq):
        """
        Resample signal based on channel type and parameters.

        Args:
            signal: Input signal data
            ch_type: Channel type ('analog' or 'digital')
            sampling_rate: Current sampling rate
            resample_freq: Target sampling rate

        Returns:
            tuple: (processed_signal, final_sampling_rate)
        """
        self.logger.info(f"Sample rate before: {sampling_rate}")

        # Store clipping threshold
        self.signal_min = np.nanmin(signal) - np.nanmean(signal)
        self.signal_max = np.nanmax(signal) - np.nanmean(signal)

        # if fs not already desired resample fs -> resample
        if resample_freq != sampling_rate:

            if self.ch_type == "digital":
                signal = self.resample_dig(signal, sampling_rate, resample_freq)

            elif self.ch_type == "analog":
                signal = self.resample_ana(signal, sampling_rate, resample_freq)

        self.logger.info(f"Sample rate after: {resample_freq}")

        return signal

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
            signal,
            up=output_rate,
            down=input_rate,
            method="polyphase",
            n_jobs=-1,
        )

        return signal_resampled

    def filter_signal(self, signal, fs, channel_groups):
        """
        Filter signal based on channel type and parameters.

        Args:
            signal: Input signal data
            fs: Sampling rate of signal
            ch_type: Channel type ('analog' or 'digital')
            sampling_rate: Current sampling rate
            resample_freq: Target sampling rate

        Returns:
            tuple: (processed_signal, final_sampling_rate)
        """

        # Store clipping threshold if no resampling has been done yet
        if self.signal_max is None or self.signal_min is None:
            self.signal_min = np.nanmin(signal) - np.nanmean(signal)
            self.signal_max = np.nanmax(signal) - np.nanmean(signal)

        [low, high] = self.get_filt_freq(self.select_ch, channel_groups)

        # Filter signal according to AASM Manual
        if not (low is None and high is None) and (self.ch_type == "analog"):

            if low and low >= fs / 2:
                self.logger.warning(
                    f"Low cutoff frequency {low} Hz is >= Nyquist frequency {fs/2} Hz. Skipping lowpass filter."
                )
                low = None  # Avoid invalid low cutoff frequency

            if high and high >= fs / 2:
                self.logger.warning(
                    f"High cutoff frequency {high} Hz is >= Nyquist frequency {fs/2} Hz. Skipping highpass filter."
                )
                high = None  # Avoid invalid high cutoff frequency

            self.logger.info(
                f"Filter signal with low: {low} Hz and high: {high} Hz bandpass"
            )

            signal = filter_data(
                signal,
                fs,
                low,
                high,
                method="fir",
                n_jobs="cuda" if cupy.cuda.is_available() else -1,
                verbose="WARNING",
            )
            # Final clipping to original signal range if highpass filter was applied
            if low is not None and low != 0:
                signal = np.clip(signal, a_min=self.signal_min, a_max=self.signal_max)

        elif not (low is None and high is None) and self.ch_type == "digital":
            self.logger.error("Digital channels cannot be filtered.")
            raise Exception("Digital channels cannot be filtered.")

        return signal
