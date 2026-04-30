from pymatreader import read_mat

from datasets.base import BaseDataset
from datasets.registry import register_dataset
from datasets.file_handlers import EEGLABHandler

@register_dataset("UCSD")
class UCSD(BaseDataset):
    """UCSD - Forehead Patch Sleep Validation Dataset (ds006695) dataset"""

    def __init__(self):
        super().__init__("UCSD","UCSD - Forehead Patch Sleep Validation Dataset (ds006695)", keep_folder_structure=False)

        self._file_handler = EEGLABHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {1: "W",
                          3: "N1",
                          4: "N2",
                          5: "N3",
                          2: "REM",
                          0: "UNK"
                          }
        
        # 33-channel data is not included in this release, maybe updated in the future, see https://doi.org/10.18112/openneuro.ds006695.v1.0.2
        self.channel_names = ['FP1-AFz', 'FP2-AFz', 'FF']
        
        # see https://doi.org/10.3389/frsle.2024.1349537
        self.inter_dataset_mapping = {
            'FP1-AFz': self.Mapping(self.TTRef.Fp1, self.TTRef.AFz),
            'FP2-AFz': self.Mapping(self.TTRef.Fp2, self.TTRef.AFz),
            'FF': self.Mapping(self.TTRef.Fp1, self.TTRef.Fp2),
        }
        
        self.channel_types = {'analog':['FP1-AFz', 'FP2-AFz', 'FF'], 
                              'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['FP1-AFz', 'FP2-AFz', 'FF'],
                                }
                
        self.file_extensions = {'psg_ext': '**/*.set',
                                'ann_ext': '**/*.set'} # annotations embedded in same file
        

    def dataset_paths(self):
        return ['', '']
    
    def ann_parse(self, ann_fname):
        mat = read_mat(ann_fname, variable_names=['VisualHypnogram'])

        hypnogram = mat['VisualHypnogram']

        ann_stage_events = []
        epoch_duration = 30

        for i, stage in enumerate(hypnogram):
            onset = i * epoch_duration
            duration = epoch_duration
            ann = {
                'Stage': stage,
                'Start': onset,
                'Duration': duration
            }
            ann_stage_events.append(ann)

        return ann_stage_events, None, None, None

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) == len(signals) + 1:
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)
