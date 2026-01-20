import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import logging
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

from psg_processing.utils import Alignment
from datasets.file_handlers import get_handler

class BaseDataset(ABC):
    """
    Abstract base class for datasets.
    Each dataset should inherit from this class and implement the required methods.
    """

    def __init__(self, dset_name: str, dataset_name_long: str, keep_folder_structure=True):
        self.dset_name = dset_name
        self.dataset_name = dataset_name_long
        self.keep_folder_structure = keep_folder_structure  # save files in the same subfolder structure as they were found recursively
        self.data_dir = None
        self.ann_dir = None

        # Dataset-specific configurations - to be set by subclasses
        self.ann2label = {}
        self.alias_mapping = {}
        self.channel_names = []
        self.channel_types = {}
        self.channel_groups = {}
        self.file_extensions = {}

        # Call setup method that subclasses must implement
        self._setup_dataset_config()

        self._file_handler = get_handler(self.dset_name, self.file_extensions['psg_ext'])()


    # Delegate file handler methods to the dataset
    def get_channels(self, logger, filepath):
        """Extract channel information from PSG file."""
        return self._file_handler.get_channels(logger, filepath)
    
    def read_signal(self, logger, filepath, channel):
        """Read signal data for a specific channel."""
        return self._file_handler.read_signal(logger, filepath, channel)
    
    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for processing."""
        return self._file_handler.get_signal_data(logger, filepath, channel)


    @abstractmethod
    def _setup_dataset_config(self):
        """
        Configure dataset-specific settings.
        Subclasses must implement this to set:
        - self.ann2label: Dict[str, int]
        - self.channel_names: List[str]
        - self.channel_types: Dict[str, List[str]]
        - self.file_extensions: Dict[str, str]
        - self.channel_groups: Dict[str, List[str]]
        - self.alias_mapping: Dict[str, List[str]] (optional)
        """
        pass

    def get_file_identifier(self, psg_fname, ann_fname):
        psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
        ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
        return psg_fname.split(psg_ext)[0], ann_fname.split(ann_ext)[0]

    def dataset_paths(self) -> List[str]:
        """
        The folder paths where this dataset is stored.
        """
        data_dir = os.path.join(self.dataset_name, "polysomnography", "edfs")
        ann_dir = os.path.join(self.dataset_name, "polysomnography", "annotations-events-nsrr")
        return data_dir, ann_dir

    def ann_parse(self, ann_fname: str) -> Tuple[List[Dict], datetime]:
        """
        Generic parse annotation file and extract sleep stage events.
        This works for datasets like
        ABC, MESA, SOF, Sleep-EDF, FDCSR, CFS, MROS, etc.

        Args:
            ann_fname: Path to annotation file

        Returns:
            Tuple of (sleep_stage_events, start_datetime)
            where sleep_stage_events is a list of dictionaries with keys:
            - 'Stage': sleep stage name
            - 'Start': start time in seconds
            - 'Duration': duration in seconds
        """

        ann_f = ET.parse(ann_fname)
        ann_root = ann_f.getroot()

        ann_stage_events = []
        ann_startdatetime = None

        for event in ann_root.iter("ScoredEvent"):
            event_concept = event.find("EventConcept").text

            # Look for recording start time
            if event_concept == "Recording Start Time":
                datetime_str = event.find("ClockTime").text
                try:
                    ann_startdatetime = datetime.strptime(datetime_str, "%d.%m.%y %H.%M.%S")
                except ValueError:
                    try:
                        ann_startdatetime = datetime.strptime(datetime_str.split(" ")[1], "%H.%M.%S")
                    except IndexError:
                        ann_startdatetime = datetime.strptime(datetime_str, "%H.%M.%S")
                continue

            # Process sleep stage events
            event_type = event.find("EventType").text
            if event_type == "Stages|Stages":
                start = float(event.find("Start").text)
                duration = float(event.find("Duration").text)
                ann_stage_events.append(
                    {
                        "Stage": event_concept.split("|")[0],
                        "Start": start,
                        "Duration": duration,
                    }
                )

        return ann_stage_events, ann_startdatetime

    def ann_label(
        self,
        logger: logging.Logger,
        ann_stage_events: List[Dict],
        epoch_duration: int,
    ) -> np.ndarray:
        """
        Generic annotation labeling function for sleep datasets with one annotation per psg file.
        This does not work for dataset ISRUC

        Args:
            logger: Logger instance
            ann_stage_events: List of annotation events with 'Start', 'Duration', 'Stage' keys
            epoch_duration: Duration of each epoch in seconds

        Returns:
            Array of sleep stage labels
        """
        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0

        for event in ann_stage_events:
            onset_sec = int(event["Start"])
            ann_duration = int(event["Duration"])
            ann_str = event["Stage"]

            # Sanity check
            assert onset_sec == total_duration, f"Onset sec of epoch is {onset_sec} but last epoch ended at {total_duration}"

            # Get label value
            if ann_str in self.ann2label:
                label = self.ann2label[ann_str]
            else:
                logger.error(f"Something unexpected: label {ann_str} not found")
                raise Exception(f"Something unexpected: label {ann_str} not found")

            # Compute # of epoch for this stage
            if ann_duration % epoch_duration != 0:
                logger.error(f"Something wrong: {ann_duration} {epoch_duration}")
                raise Exception(f"Something wrong: {ann_duration} {epoch_duration}")
            n_epochs = int(ann_duration / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(n_epochs, dtype=np.int32) * label
            labels.append(label_epoch)

            total_duration += ann_duration

            # logger.info("Include onset:{}, duration:{}, label:{} ({})".format(onset_sec, duration_sec, label, ann_str))

        return np.concatenate(labels)

    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs) -> Tuple[bool, float]:
        """ Align front part of signals and labels, in some datasets annotations start after signal recording"""

        raise NotImplementedError("Subclass has no front alignment implemented")

    def base_align_front(self, logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels, fs):
        if delay_sec < 0:
            logger.error(f"Annotations start before signal start, which is not supported in the base align front method")
            raise Exception("Annotations start before signal start, which is not supported in the base align front method")
        
        if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_ANNOT.value:
            logger.info(f"Labeling started {delay_sec/60:.2f} min after signal start, signal will be shortened at the front to match")
            signal = signal[int(delay_sec*fs):]
        elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_SIGNAL.value:
            logger.info(f"Labeling started {delay_sec/60:.2f} min after signal start, labels will be padded at the front with full epochs of value:{pad_values["label"]} to match")
            n_pad = int(delay_sec // epoch_duration)
            # adapt start times of all existing labels
            for event in labels:
                event['Start'] += n_pad * epoch_duration
            # create epochs to pad at the front
            new_labels = []
            for i in range(n_pad):
                new_labels.append({
                'Stage': pad_values["label"],
                'Start': i * epoch_duration,
                'Duration': epoch_duration
                })
            labels = new_labels + labels
            if delay_sec % epoch_duration != 0:
                logger.info(f"Partial epoch detected at start, signal will be shortened at the front to match")
                signal = signal[int((delay_sec % epoch_duration)*fs):]
        else:
            raise ValueError(f"Unknown alignment option: {alignment}")
        return signal, labels


    def align_end(
        self,
        logger,
        alignment,
        pad_values,
        psg_fname: str,
        ann_fname: str,
        signals: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclass has no end alignment implemented but is required")
    
    def base_align_end_labels_longer(self, logger, alignment, pad_values, signals, labels):
        if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_SIGNAL.value:
            logger.info(f"Labels (len:{len(labels)}) are shortend to match signal (len:{len(signals)})")
            labels = labels[:len(signals)]
        elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_ANNOT.value:
            n_pad = (len(labels) - len(signals))
            logger.info(f"Signal (len:{len(signals)}) will be padded at the end with {n_pad} epochs of constant value:{pad_values["signal"]} to match labels length (len:{len(labels)})")
            signals = np.vstack((signals, np.full((n_pad, signals.shape[1]), pad_values["signal"])))
        else:
            raise ValueError(f"Unknown alignment option: {alignment}")
        return signals,labels

    def base_align_end_signals_longer(self, logger, alignment, pad_values, signals, labels):
        if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_ANNOT.value:
            logger.info(f"Signal (len:{len(signals)}) is shortend to match label (len:{len(labels)})")
            signals = signals[:len(labels)]
        elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_SIGNAL.value:
            n_pad = int((len(signals) - len(labels)))
            logger.info(f"Labels (len:{len(labels)}) will be padded at the end with {n_pad} epochs of value:{pad_values["label"]} to match signals (len:{len(signals)}))")
            labels = np.hstack((labels, np.full((n_pad,),pad_values["label"])))
        else:
            raise ValueError(f"Unknown alignment option: {alignment}")
        return signals, labels


    def preprocess(self, data_dir, ann_dir, output_dir):
        """
        Preprocess files before they can be processed
        Can include new sorting and copying/renaming
        """
        pass
