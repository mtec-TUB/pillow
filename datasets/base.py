import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

from psg_processing import Dataset_Explorer, DatasetProcessor
from psg_processing.file_handlers.factory import get_handler


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

    def dataset_paths(self) -> List[str]:
        """
        The folder paths where this dataset is stored.
        """
        data_dir = os.path.join(self.dataset_name, "polysomnography", "edfs")
        ann_dir = os.path.join(self.dataset_name, "polysomnography", "annotations-events-nsrr")
        return data_dir, ann_dir

    def ann_parse(self, ann_fname: str, epoch_duration: Optional[int] = None) -> Tuple[List[Dict], datetime]:
        """
        Generic parse annotation file and extract sleep stage events.
        This works for datasets like
        ABC, MESA, SOF, Sleep-EDF, FDCSR, CFS, MROS, etc.

        Args:
            ann_fname: Path to annotation file
            epoch_duration: Duration of each epoch in seconds (optional)

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
            duration_sec = int(event["Duration"])
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
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int32) * label
            labels.append(label_epoch)

            total_duration += duration_sec

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
        #     logger.info(f"Labels (len: {len(labels)}) are shortend to match signal length ({len(signals)})")
        #     labels = labels[:len(signals)]

        assert len(signals) == len(labels), f"Length mismatch: signal ({os.path.basename(psg_fname)})={len(signals)}, labels({os.path.basename(ann_fname)})={len(labels)} TODO: implement alignment function"

        return signals, labels
    
    def preprocess(self, data_dir, ann_dir, output_dir):
        """
        Preprocess files before they can be processed
        Can include new sorting and copying/renaming
        """
        pass

    def process(self, action, data_dir, ann_dir, output_dir, resample, channels, num_jobs, overwrite):
        """
        Main processing entry point.
        This calls the prepare_files function with dataset-specific parameters.
        """
        ret = self.preprocess(data_dir, ann_dir, output_dir)

        if ret is False:
            return

        # Add the parent directory to the path so we can import psg_processing
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        self.psg_file_handler = get_handler(self.dset_name, self.file_extensions['psg_ext'])

        if action == "prepare":

            # all channels will be processed if empty list
            if channels != []:
                self.channel_names = channels

            # Initialize a new DatasetProcessor
            processor = DatasetProcessor(self,data_dir,ann_dir,output_dir,overwrite=overwrite)

            # Use the new prepare_files method with dataset-specific parameters
            processor.prepare_files(
                resample,
                epoch_duration=30,
                num_jobs=num_jobs
            )

        elif action == "get_channel_names":
            explorer = Dataset_Explorer(None, self.psg_file_handler, data_dir, ann_dir, **self.file_extensions)
            channels = list(explorer.get_all_channels())
            print(f"Available channels in {self.dset_name}: {(channels)}")

        elif action == "get_channel_types":
            explorer = Dataset_Explorer(None, self.psg_file_handler, data_dir, ann_dir, **self.file_extensions)
            explorer.get_all_channels()
            channel_types = explorer.get_channel_type()
            print(f"Channel types in {self.dset_name}: {channel_types}")

        else:
            raise ValueError(f"Unknown action: {action}")
