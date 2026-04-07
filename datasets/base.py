import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import logging
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum, auto

from psg_processing.utils import Alignment
from .file_handlers import EDFHandler

class BaseDataset(ABC):
    """Abstract base class for datasets.
    Holds all dataset-specific information and methods for loading PSG files, parsing annotations, and harmonizing channel names.
    Each dataset should inherit from this class and implement the required methods.
    """

    def __init__(self, dset_name: str, dataset_name_long: str, keep_folder_structure=True):
        self.dset_name = dset_name
        self.dataset_name = dataset_name_long
        self.keep_folder_structure = keep_folder_structure  # save files in the same subfolder structure as they were found

        # Dataset-specific configurations - to be set by subclasses (in _setup_dataset_config)
        self.ann2label = {}
        self.intra_dataset_mapping = {}
        self.inter_dataset_mapping = {}
        self.channel_names = []
        self.channel_types = {}
        self.channel_groups = {}
        self.file_extensions = {}

        # Call setup method that subclasses must implement
        self._setup_dataset_config()

        # Default way to load psg files is with the EDF-Handler, datasets that do not use EDF files override this
        self._file_handler = EDFHandler()


    # Delegate file handler methods to the dataset or use dataset-specific functions here
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
        """Configure dataset-specific settings.
        Subclasses must implement this to set:
        - self.ann2label: Dict[str, int]
        - self.channel_names: List[str]
        - self.channel_types: Dict[str, List[str]]
        - self.inter_dataset_mapping: Dict[str, Mapping]
        - self.file_extensions: Dict[str, str]
        - self.channel_groups: Dict[str, List[str]]
        - self.intra_dataset_mapping: Dict[str, List[str]]
        """
        pass

    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        """Used to find corresponding PSG and annotation files based on filename patterns
        By default, it removes the file extensions and returns the base name
        Datasets with more complex naming conventions can override this method
        """
        psg_id, ann_id = None, None
        if psg_fname:
            psg_ext = self.file_extensions['psg_ext'].split('*')[-1]
            psg_id = psg_fname.split(psg_ext)[0]
        if ann_fname:
            ann_ext = self.file_extensions['ann_ext'].split('*')[-1]
            ann_id = ann_fname.split(ann_ext)[0]
        return psg_id, ann_id

    def dataset_paths(self) -> List[str]:
        """Paths where PSG and Annotations are stored.
        Are used to construct full paths if no specific dataset location is given in config.
        Default: NSRR structure
        """
        data_dir = os.path.join("polysomnography", "edfs")
        ann_dir = os.path.join("polysomnography", "annotations-events-nsrr")
        return data_dir, ann_dir
    
    def map_channel(self, channel):
        """Harmonize channel name using intra- and inter-dataset mappings.
        Intra-dataset mapping maps channel names within the same dataset to a common name (e.g., 'ECG R' and 'ECGR' both map to 'ECGR').
        Inter-dataset mapping maps common names to a harmonized naming convention across datasets.
        """

        harm_channel = channel

        # Intra-dataset channel name harmonization
        if self.intra_dataset_mapping:
            for key, aliases in self.intra_dataset_mapping.items():
                if channel in aliases:
                    harm_channel = key
                    break

        # Inter-dataset channel name harmonization
        mapping = self.inter_dataset_mapping
        if harm_channel in mapping:
            chnl = mapping[harm_channel] # change name to harmonized one
            harm_channel = chnl.get_mapping()      
        
        return harm_channel
    
    class Mapping:
        def __init__(self, ref1, ref2):
            self.ref1 = ref1
            self.ref2 = ref2
        
        def __eq__(self, other):
            return (self.ref1, self.ref2) == (other.ref1, other.ref2)
        
        def get_mapping(self):
            return f"{self.ref1}-{self.ref2}" if self.ref2 is not None else f"{self.ref1}"
        
    class TTRef(Enum):        
        # 10-10 EEG system for scalp PSG

        """
        "MCN system renames four electrodes of the 10–20 system:
        T3 is now T7
        T4 is now T8
        T5 is now P7
        T6 is now P8"
        
        Source: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
        """
        
        Nz = auto()
        Fpz = auto()
        Fp1 = auto()
        Fp2 = auto()
        AF7 = auto()
        AF3 = auto()
        AFz = auto()
        AF4 = auto()
        AF8 = auto()
        F9 = auto()
        F7 = auto()
        F5 = auto()
        F3 = auto()
        F1 = auto()
        Fz = auto()
        F2 = auto()
        F4 = auto()
        F6 = auto()
        F8 = auto()
        F10 = auto()
        FT9 = auto()
        FT7 = auto()
        FC5 = auto()
        FC3 = auto()
        FC1 = auto()
        FCz = auto()
        FC2 = auto()
        FC4 = auto()
        FC6 = auto()
        FT8 = auto()
        FT10 = auto()
        T7 = auto() # Same as T3 in 10-20 system
        C5 = auto()
        C3 = auto()
        C1 = auto()
        Cz = auto()
        C2 = auto()
        C4 = auto()
        C6 = auto()
        T8 = auto() # Same as T4 in 10-20 system
        TP9 = auto()
        TP7 = auto()
        CP5 = auto()
        CP3 = auto()
        CP1 = auto()
        CPz = auto()
        CP2 = auto()
        CP4 = auto()
        CP6 = auto()
        TP8 = auto()
        TP10 = auto()
        P9 = auto()
        P7 = auto() # Same as T5 in 10-20 system
        P5 = auto()
        P3 = auto()
        P1 = auto()
        Pz = auto()
        P2 = auto()
        P4 = auto()
        P6 = auto()
        P8 = auto() # Same as T6 in 10-20 system
        P10 = auto()
        PO7 = auto()
        PO3 = auto()
        POz = auto()
        PO4 = auto()
        PO8 = auto()
        O1 = auto()
        Oz = auto()
        O2 = auto()
        Iz = auto()
        LPA = auto() # Same as A1 in 10-20 system
        RPA = auto() # Same as A2 in 10-20 system
        LRPA = auto() # Linked mastoids
        
        EL = auto() # LOC, E1
        ER = auto() # ROC, E2
        
        # Computed linked Ear and Linked Ear Reference. May be rare, and so far is only in MASS. Can only find this article describing it: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5479869/
        CLE = auto()
        LER = auto()

        # Other PSG Channels
        ECG = auto()
        EMG_CHIN = auto()
        EMG_LCHIN = auto()
        EMG_RCHIN = auto()
        EMG_LLEG = auto()
        EMG_RLEG = auto()
        AIRFLOW = auto()
        THORACIC = auto()
        ABDOMINAL = auto()
        SPO2 = auto()
        CPAP = auto()
        SNORE = auto()
        POSITION = auto()
        HR = auto() # Heart Rate
        
        def __str__(self):
            return self.name

    def get_light_times(self, logger, psg_fname):
        """Get lights on and lights off time, if available in dataset
        Used to extract only epochs between these markers
        """
        lights_off = None
        lights_on = None
        return lights_off, lights_on

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
                stage = event_concept.split("|")[0]
                if stage == "Beginning of analysis period" or stage == "End of analysis period":
                    continue
                start = float(event.find("Start").text)
                duration = float(event.find("Duration").text)
                ann_stage_events.append(
                    {
                        "Stage": stage,
                        "Start": start,
                        "Duration": duration,
                    }
                )

        lights_off, lights_on = None, None

        return ann_stage_events, ann_startdatetime, lights_off, lights_on

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

            logger.debug("Include onset:{}, duration:{}, label:{} ({})".format(onset_sec, ann_duration, label, ann_str))

        return np.concatenate(labels)

    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs) -> Tuple[bool, float]:
        """ Align front part of signals and labels, in some datasets annotations start after signal recording"""
        logger.error("Signal and Annotations do not start at the same time. TODO: implement alignment function")
        raise NotImplementedError("Subclass has no front alignment implemented")

    def base_align_front(self, logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels, fs):
        start_time_shift = 0
        if delay_sec < 0:
            advance_sec = -delay_sec
            if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_SIGNAL.value:
                logger.info(f"Signal started {advance_sec:.2f} sec after label start, labels will be shortened at the front to match")
                n_crop = int(advance_sec//epoch_duration)
                labels = labels[n_crop:]
                if advance_sec % epoch_duration != 0:
                    logger.info(f"Partial epoch detected at start, signal ({epoch_duration-advance_sec} sec) and labels (one epoch) will be shortened at the front to match")
                    labels = labels[1:]
                    signal = signal[int((epoch_duration-(advance_sec % epoch_duration))*fs):]
                    start_time_shift = epoch_duration - (advance_sec % epoch_duration) 
            elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_ANNOT.value:
                logger.info(f"Signal started {advance_sec:.2f} sec after label start, signal will be padded with constant value:{np.float64(pad_values["signal"])} at the front to match")
                n_pad_samples = int(advance_sec*fs)
                signal = np.hstack((np.full((n_pad_samples,), np.float64(pad_values["signal"])), signal))
                start_time_shift = delay_sec
        else:
            if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_ANNOT.value:
                logger.info(f"Labeling started {delay_sec/60:.2f} min after signal start, signal will be shortened at the front to match")
                signal = signal[int(delay_sec*fs):]
                start_time_shift = delay_sec
            elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_SIGNAL.value:
                n_pad = int(delay_sec // epoch_duration)
                logger.info(f"Labeling started {delay_sec/60:.2f} min after signal start, labels will be padded at the front with {n_pad} epochs of value:{pad_values["label"]} to match")
                labels = np.hstack((np.full((n_pad,), pad_values["label"]), labels))
                if delay_sec % epoch_duration != 0:
                    logger.info(f"Partial epoch detected at start, signal will be shortened at the front to match")
                    signal = signal[int((delay_sec % epoch_duration)*fs):]
                    start_time_shift = delay_sec - n_pad*epoch_duration
        return start_time_shift, signal, labels

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
        logger.error(f"Length mismatch: signal ({os.path.basename(psg_fname)})={len(signals)}, labels({os.path.basename(ann_fname)})={len(labels)}")
        raise NotImplementedError("Subclass has no end alignment implemented but is required")
    
    def base_align_end_labels_longer(self, logger, alignment, pad_values, signals, labels):
        if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_SIGNAL.value:
            logger.info(f"Labels (len:{len(labels)}) are shortend to match signal (len:{len(signals)})")
            labels = labels[:len(signals)]
        elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_ANNOT.value:
            n_pad = (len(labels) - len(signals))
            logger.info(f"Signal (len:{len(signals)}) will be padded at the end with {n_pad} epochs of constant value:{np.float64(pad_values["signal"])} to match labels length (len:{len(labels)})")
            signals = np.vstack((signals, np.full((n_pad, signals.shape[1]), np.float64(pad_values["signal"]))))
        return signals,labels

    def base_align_end_signals_longer(self, logger, alignment, pad_values, signals, labels):
        if alignment == Alignment.MATCH_SHORTER.value or alignment == Alignment.MATCH_ANNOT.value:
            logger.info(f"Signal (len:{len(signals)}) is shortend to match label (len:{len(labels)})")
            signals = signals[:len(labels)]
        elif alignment == Alignment.MATCH_LONGER.value or alignment == Alignment.MATCH_SIGNAL.value:
            n_pad = int((len(signals) - len(labels)))
            logger.info(f"Labels (len:{len(labels)}) will be padded at the end with {n_pad} epochs of value:{pad_values["label"]} to match signals (len:{len(signals)}))")
            labels = np.hstack((labels, np.full((n_pad,),pad_values["label"])))
        return signals, labels


    def preprocess(self, data_dir, ann_dir, output_dir):
        """
        Preprocess files before they can be processed
        Can include new sorting and copying/renaming
        """
        pass
