import pandas as pd

from datasets.base import BaseDataset
from datasets.registry import register_dataset

@register_dataset("BOAS")
class BOAS(BaseDataset):
    """BOAS - Bitbrain Open Access Sleep (ds005555) dataset"""

    def __init__(self):
        super().__init__("BOAS","BOAS - Bitbrain Open Access Sleep (ds005555)", keep_folder_structure=False)
        self.has_end_alignment = True

        # File "sub-72_task-Sleep_acq-psg_eeg.edf" is shorter than the corresponding headband file
        # We will truncate all channels to the shortest signal length to ensure alignment across channels
        self.shortest_signal_length = None
    
    def _setup_dataset_config(self):
        self.ann2label = {0: "W",
                          1: "N1",
                          2: "N2",
                          3: "N3",   
                          4: "REM",
                          8: "UNK",
                          -2: "UNK" 
        }        
        
        self.channel_names = ['HB_1', 'HB_2', 'PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG', 'PSG_EMG', 'PSG_THER', 'PSG_THOR', 
                              'PSG_ABD', 'HB_IMU_1', 'HB_IMU_2', 'HB_IMU_3', 'HB_IMU_4', 'HB_IMU_5', 'HB_IMU_6', 'HB_PULSE', 'PSG_PULSE', 'PSG_BEAT', 
                              'PSG_SPO2', 'PSG_EOGL', 'PSG_EOGR', 'PSG_CAN']
        
        self.hb_channel_names = ['HB_1', 'HB_2','HB_IMU_1', 'HB_IMU_2', 'HB_IMU_3', 'HB_IMU_4', 'HB_IMU_5', 'HB_IMU_6', 'HB_PULSE']
        self.psg_channel_names = ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG', 'PSG_EMG', 'PSG_THER', 'PSG_THOR', 
                                 'PSG_ABD','PSG_PULSE', 'PSG_BEAT', 'PSG_SPO2', 'PSG_EOGL', 'PSG_EOGR', 'PSG_CAN' ]
        
        # see README
        self.inter_dataset_mapping = {
            'PSG_F3': self.Mapping(self.TTRef.F3, None),
            'PSG_F4': self.Mapping(self.TTRef.F4, None),
            'PSG_C3': self.Mapping(self.TTRef.C3, None),
            'PSG_C4': self.Mapping(self.TTRef.C4, None),
            'PSG_O1': self.Mapping(self.TTRef.O1, None),
            'PSG_O2': self.Mapping(self.TTRef.O2, None),
            'PSG_EOG': self.Mapping(self.TTRef.EL, self.TTRef.ER),
            'PSG_EOGL': self.Mapping(self.TTRef.EL, self.TTRef.LPA),
            'PSG_EOGR': self.Mapping(self.TTRef.ER, self.TTRef.LPA),
            'PSG_EMG': self.Mapping(self.TTRef.EMG_CHIN, None),
            'PSG_THER': self.Mapping(self.TTRef.AIRFLOW, None), # both PSG_THER and PSG_CAN measure AIRFLOW, which to choose?
            'PSG_THOR': self.Mapping(self.TTRef.THORACIC, None),
            'PSG_ABD': self.Mapping(self.TTRef.ABDOMINAL, None),
            'PSG_BEAT': self.Mapping(self.TTRef.HR, None),
            'PSG_SPO2': self.Mapping(self.TTRef.SPO2, None),
            'HB_1': self.Mapping(self.TTRef.AF7, None),
            'HB_2': self.Mapping(self.TTRef.AF8, None),
        }
        
        self.channel_types = {'analog': ['HB_1', 'HB_2', 'PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG', 'PSG_EMG', 'PSG_THER', 
                                         'PSG_THOR', 'PSG_ABD', 'HB_IMU_1', 'HB_IMU_2', 'HB_IMU_3', 'HB_IMU_4', 'HB_IMU_5', 'HB_IMU_6', 'HB_PULSE', 
                                         'PSG_PULSE', 'PSG_BEAT', 'PSG_SPO2', 'PSG_EOGL', 'PSG_EOGR', 'PSG_CAN'], 
                              'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG','HB_1', 'HB_2','PSG_EOGL', 'PSG_EOGR', ],
                                'emg': ['PSG_EMG',],
                                'thoraco_abdo_resp': ['PSG_CAN','PSG_THER', 'PSG_THOR', 'PSG_ABD',],
                                }
                
        self.file_extensions = {'psg_ext': '**/*psg_eeg.edf',
                                'ann_ext': '**/*psg_events.tsv'} 
        

    # Use custom file handlers because channels are split into two separate EDF files (PSG and Headband)
    def get_channels(self, logger, filepath):
        """Extract channel information from PSG file."""
        psg_channels = self._file_handler.get_channels(logger, filepath)

        headband_filepath = filepath.replace('psg_eeg.edf', 'headband_eeg.edf')
        headband_channels = self._file_handler.get_channels(logger, headband_filepath)
        all_channels = psg_channels + headband_channels
        if len(all_channels) > len(set(all_channels)):
            raise ValueError(f"Duplicate channel names found in PSG and Headband files. Please check the files and ensure unique channel names.")
        
        # Find the shortest signal length across all channels
        channel_lengths = [len(self.read_signal(logger, filepath, ch)) for ch in all_channels]
        self.shortest_signal_length = min(channel_lengths)

        return all_channels
    
    def read_signal(self, logger, filepath, channel):
        """Read signal data for a specific channel."""
        if channel in self.psg_channel_names:
            return self._file_handler.read_signal(logger, filepath, channel)
        elif channel in self.hb_channel_names:
            headband_filepath = filepath.replace('psg_eeg.edf', 'headband_eeg.edf')
            return self._file_handler.read_signal(logger, headband_filepath, channel)
        else:
            raise ValueError(f"Channel {channel} not found in either PSG or Headband channel list")
    
    def get_file_info(self, logger, filepath):
        """Get start datetime and file duration."""
        psg_info = self._file_handler.get_file_info(logger, filepath)
        psg_start_datetime = psg_info["start_datetime"].replace(tzinfo=None)
        psg_duration = psg_info["file_duration"]

        headband_filepath = filepath.replace('psg_eeg.edf', 'headband_eeg.edf')
        headband_info = self._file_handler.get_file_info(logger, headband_filepath)
        hb_start_datetime = headband_info["start_datetime"].replace(tzinfo=None)
        hb_duration = headband_info["file_duration"]
        
        if psg_duration != hb_duration:
            logger.warning(f"Duration of PSG file ({psg_duration} sec) and Headband file ({hb_duration} sec) do not match. They will be aligned using the shorter duration.")
            psg_info["file_duration"] = min(psg_duration, hb_duration)

        if psg_start_datetime != hb_start_datetime:
            logger.error(f"Start datetimes of PSG ({psg_start_datetime}) and Headband ({hb_start_datetime}) files do not match.")
            raise ValueError(f"Start datetimes of PSG ({psg_start_datetime}) and Headband ({hb_start_datetime}) files do not match.")
        else:
            return psg_info            
    
    def get_signal_data(self, logger, filepath, channel):
        """Get complete signal information for processing."""
        if channel in self.psg_channel_names:
            signal_data = self._file_handler.get_signal_data(logger, filepath, channel)
        elif channel in self.hb_channel_names:
            headband_filepath = filepath.replace('psg_eeg.edf', 'headband_eeg.edf')
            signal_data = self._file_handler.get_signal_data(logger, headband_filepath, channel)
        else:
            logger.error(f"Channel {channel} not found in either PSG or Headband channel list")
            raise ValueError(f"Channel {channel} not found in either PSG or Headband channel list")
        
        if len(signal_data["signal"]) > self.shortest_signal_length:
            signal_data["file_duration"] = self.shortest_signal_length / signal_data["sampling_rate"]
            logger.warning(f"Truncating signal for channel {channel} to shortest signal ({signal_data["file_duration"]} sec) to ensure alignment across channels.")
            signal_data["signal"] = signal_data["signal"][:self.shortest_signal_length]

        return signal_data

    def dataset_paths(self):
        return ['', '']
    
    def ann_parse(self, ann_fname):
        # Reads only the scorings of human experts (in *_psg_events.tsv), AI scores in *_psg_events.tsv and *_headband_events.tsv are ignored
        annot = pd.read_csv(ann_fname,sep='\t', header=0)

        ann_stage_events = []

        start_time = None

        for i, row in annot.iterrows():
            start = float(row['onset'])
            if start_time is None:
                start_time = start

            duration = float(row['duration'])

            stage_hum = row['stage_hum']
            ann_stage_events.append({'Stage': stage_hum,
                                    'Start': start-start_time,
                                    'Duration': duration})


        return ann_stage_events, start_time, None, None
