import logging
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Union

class Alignment(Enum):
    """Options for aligning signal and annotation lengths at front and/or end."""

    MATCH_SHORTER = "match_shorter"  # no padding, but cropping if necessary
    MATCH_LONGER = "match_longer"  # no cropping, but padding with custom values
    MATCH_SIGNAL = "match_signal"  # pad/crop to signal length
    MATCH_ANNOT = "match_annot"  # pad/crop to annotation length

class ConfigError(ValueError):
    """Raised when configuration validation fails."""
    pass

class ProcessorConfig:
    """Configuration dataclass for dataset processing.
    See config.yaml for detailed explanations of each parameter."""

    # Define valid options for configuration parameters
    VALID_OUTPUT_FORMATS = {"npz", "edf", "hdf5"}
    VALID_ACTIONS = {"process", "get_channel_names", "get_channel_types"}
    VALID_ALIGNMENT = [a.value for a in Alignment]
    VALID_FILTER_GROUPS = {
        "eeg_eog", "emg", "ecg",
        "thoraco_abdo_resp", "nasal_pressure",
        "snoring", "default"
    }

    def __init__(self, **kwargs):
        # validate and set all required parameters

        self.dataset = kwargs.get("dataset")

        # Path parameters
        self.base_data_dir: Path = self._validate_path(
            kwargs.get("base_data_dir"),
        )

        self.data_dir: Optional[Path] = self._validate_path(
            kwargs.get("data_dir")
        )

        self.ann_dir: Optional[Path] = self._validate_path(
            kwargs.get("ann_dir")
        )

        self.output_dir: Optional[Path] = self._validate_path(
            kwargs.get("output_dir")
        )

        # Enum parameters
        self.output_format: str = self._validate_enum(
            kwargs.get("output_format"),
            self.VALID_OUTPUT_FORMATS,
            "output_format"
        )

        self.logging_level: str = self._validate_enum(
            kwargs.get("logging_level"),
            {name for name, lvl in logging._nameToLevel.items()},
            "logging_level"
        )

        self.action: str = self._validate_enum(
            kwargs.get("action"),
            self.VALID_ACTIONS,
            "action"
        )

        self.alignment: str = self._validate_enum(
            kwargs.get("alignment"),
            self.VALID_ALIGNMENT,
            "alignment"
        )

        # Boolean parameters
        self.overwrite: bool = self._validate_bool(kwargs.get("overwrite"))
        self.filter: bool = self._validate_bool(kwargs.get("filter"))
        self.map_channel_names: bool = self._validate_bool(
            kwargs.get("map_channel_names")
        )
        self.rm_move: bool = self._validate_bool(kwargs.get("rm_move"))
        self.rm_unk: bool = self._validate_bool(kwargs.get("rm_unk"))
        self.use_annot: bool = self._validate_bool(kwargs.get("use_annot"))

        # Other parameters
        self.resample: Optional[int] = self._validate_resample(
            kwargs.get("resample"),
            "resample"
        )

        self.epoch_duration: int = self._validate_epoch_duration(
            kwargs.get("epoch_duration")
        )

        self.min_sleep_epochs: int = self._validate_min_sleep_epochs(
            kwargs.get("min_sleep_epochs"),
            "min_sleep_epochs"
        )

        self.channels: List[str] = self._validate_channels(
            kwargs.get("channels"),
            "channels"
        )

        self.n_wake_epochs: Union[int, str] = \
            self._validate_n_wake_epochs(kwargs.get("n_wake_epochs"))

        self.filter_freq: Dict[str, List[Optional[float]]] = \
            self._validate_filter_freq(kwargs.get("filter_freq"))

        self.pad_values = self._validate_pad_values(
            kwargs.get("pad_values")
        )

        # ---------- Cross-checks ----------
        self._validate_consistency()

    def _validate_enum(self, value, valid_set, name):
        if value not in valid_set:
            raise ConfigError(
                f"{name} must be one of {valid_set}, got {value}"
            )
        return value

    def _validate_bool(self, value):
        if not isinstance(value, bool):
            raise ConfigError(f"Expected bool, got {value}")
        return value

    def _validate_path(self, value):
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"Invalid path type: {value}")
        return Path(value)

    def _validate_resample(self, value):
        if value is None:
            return None
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"resample must be positive int.")
        return value

    def _validate_min_sleep_epochs(self, value, name):
        return self._validate_non_negative_int(value, name)

    def _validate_epoch_duration(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ConfigError("epoch_duration must be positive integer.")
        if 30 % value == 0:
            return value
        raise ConfigError(
            "Epoch_duration must divide 30."
        )

    def _validate_channels(self, value):
        if value is None:
            return []
        if not isinstance(value, list) or \
           not all(isinstance(v, str) for v in value):
            raise ConfigError(f"channels must be list of strings.")
        return value

    def _validate_n_wake_epochs(self, value):
        if value == "all":
            return value
        if isinstance(value, int) and value >= 0:
            return value
        raise ConfigError(
            "n_wake_epochs must be non-negative int or 'all'."
        )

    def _validate_filter_freq(self, value):
        if not isinstance(value, dict):
            raise ConfigError("filter_freq must be a dictionary.")

        for key in value:
            if key not in self.VALID_FILTER_GROUPS:
                raise ConfigError(
                    f"Invalid filter group: {key}"
                )
            freq = value[key]
            if (not isinstance(freq, list)) or len(freq) != 2:
                raise ConfigError(
                    f"{key} must be [low, high]"
                )

            low, high = freq
            if low is not None and (not isinstance(low, (int, float)) or low < 0):
                raise ConfigError(f"{key}: invalid low cutoff")
            if high is not None and (not isinstance(high, (int, float)) or high <= 0):
                raise ConfigError(f"{key}: invalid high cutoff")
            if low and high and low >= high:
                raise ConfigError(
                    f"{key}: low cutoff must be < high cutoff"
                )

        return value

    def _validate_pad_values(self, value):
        if not isinstance(value, dict):
            raise ConfigError("pad_values must be dict.")
        signal = value.get("signal")
        label = value.get("label")

        if label is None or not isinstance(label, int):
            raise ConfigError("pad_values['label'] must be int.")

        return {"signal": signal, "label": label}

    def _validate_consistency(self):
        
        has_base = self.base_data_dir is not None
        has_specific = (
            self.data_dir is not None and
            self.ann_dir is not None
        )

        if not (has_base or has_specific):
            raise ConfigError(
                "Either 'base_data_dir' must be provided OR "
                "both 'data_dir' and 'ann_dir' must be provided."
            )

        # Optional: forbid ambiguous configuration
        if has_base and has_specific:
            raise ConfigError(
                "Provide either 'base_data_dir' OR "
                "'data_dir' and 'ann_dir', not both."
            )
    
        # Filtering requires resampling or original fs known
        if self.filter and not self.filter_freq:
            raise ConfigError("filter=True but no filter_freq defined.")

        if not self.use_annot:
            if self.rm_move or self.rm_unk:
                raise ConfigError(
                    "rm_move/rm_unk require use_annot=True."
                )
            if self.min_sleep_epochs > 0:
                raise ConfigError(
                    "min_sleep_epochs requires use_annot=True."
                )

