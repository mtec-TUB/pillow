import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

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
                logger.info(f"Something unexpected: label {ann_str} not found")
                raise Exception(f"Something unexpected: label {ann_str} not found")

            # Compute # of epoch for this stage
            if ann_duration % epoch_duration != 0:
                logger.info(f"Something wrong: {ann_duration} {epoch_duration}")
                raise Exception(f"Something wrong: {ann_duration} {epoch_duration}")
            n_epochs = int(ann_duration / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(n_epochs, dtype=np.int32) * label
            labels.append(label_epoch)

            total_duration += ann_duration

            # logger.info("Include onset:{}, duration:{}, label:{} ({})".format(onset_sec, duration_sec, label, ann_str))

        return np.concatenate(labels)

    def align_front(self, logger, ann_Startdatetime, psg_fname, ann_fname: str, signal: np.ndarray, labels, fs) -> Tuple[np.ndarray, np.ndarray]:
        """ Align front part of signals and labels, in some datasets annotations start after signal recording"""
        return False, signal, labels

    def align_end(
        self,
        logger,
        psg_fname: str,
        ann_fname: str,
        signals: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align end part of signals and labels, handling dataset-specific issues."""

        # if len(labels) > len(signals):
        #     logger.info(f"Labels (len: {len(labels)}) are shortend to match signal ({len(signals)})")
        #     labels = labels[:len(signals)]

        assert len(signals) == len(labels), f"Length mismatch: signal ({os.path.basename(psg_fname)})={len(signals)}, labels({os.path.basename(ann_fname)})={len(labels)} TODO: implement alignment function"

        return signals, labels
    
    def preprocess(self, data_dir, ann_dir, output_dir):
        """
        Preprocess files before they can be processed
        Can include new sorting and copying/renaming
        """
        pass
