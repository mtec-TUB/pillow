import os
import pandas as pd
from pathlib import Path
import glob

from datasets.base import BaseDataset
from datasets.registry import register_dataset

from datasets.file_handlers import EEGLABHandler


@register_dataset("EffectsOfAlertness")
class EffectsOfAlertness(BaseDataset):
    """Effects of alertness on perceptual detection and discrimination dataset"""
    
    def __init__(self):
        super().__init__("EffectsOfAlertness","Effects of alertness on perceptual detection and discrimination", keep_folder_structure=False)

        self._file_handler = EEGLABHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {
                        1: "W",   # Wake
                        3: "N1",   # NREM Stage 1
                        4: "N2",   # NREM Stage 2
                        5: "N3",   # NREM Stage 3
                        2: "REM",   # REM sleep
                        7: "UNK",   # Artefact
                        8: "UNK",   # Unknown
                        }
        
        
        self.channel_names = ['AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'E10', 'E100', 'E101', 
                              'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E11', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E12', 'E122', 'E123', 
                              'E124', 'E13', 'E15', 'E16', 'E18', 'E19', 'E2', 'E20', 'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E3', 'E30', 'E31', 'E33', 'E34', 'E35', 
                              'E36', 'E37', 'E39', 'E4', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E5', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E6', 'E60', 
                              'E61', 'E62', 'E65', 'E66', 'E67', 'E7', 'E70', 'E71', 'E72', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E83', 'E84', 'E85', 'E86', 'E87', 'E9', 'E90', 
                              'E91', 'E92', 'E93', 'E96', 'E97', 'E98', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'FT7', 'FT8', 
                              'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 
                              'TP10', 'TP7', 'TP8', 'TP9']
        
        self.inter_dataset_mapping = {
        }
        
        self.channel_types = {'analog': ['AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'E10', 'E100', 'E101', 
                              'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E11', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E12', 'E122', 'E123', 
                              'E124', 'E13', 'E15', 'E16', 'E18', 'E19', 'E2', 'E20', 'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E3', 'E30', 'E31', 'E33', 'E34', 'E35', 
                              'E36', 'E37', 'E39', 'E4', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E5', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E6', 'E60', 
                              'E61', 'E62', 'E65', 'E66', 'E67', 'E7', 'E70', 'E71', 'E72', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E83', 'E84', 'E85', 'E86', 'E87', 'E9', 'E90', 
                              'E91', 'E92', 'E93', 'E96', 'E97', 'E98', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'FT7', 'FT8', 
                              'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 
                              'TP10', 'TP7', 'TP8', 'TP9'], 
                              'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'E10', 'E100', 'E101', 
                              'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E11', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E12', 'E122', 'E123', 
                              'E124', 'E13', 'E15', 'E16', 'E18', 'E19', 'E2', 'E20', 'E22', 'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E3', 'E30', 'E31', 'E33', 'E34', 'E35', 
                              'E36', 'E37', 'E39', 'E4', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E5', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E6', 'E60', 
                              'E61', 'E62', 'E65', 'E66', 'E67', 'E7', 'E70', 'E71', 'E72', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E83', 'E84', 'E85', 'E86', 'E87', 'E9', 'E90', 
                              'E91', 'E92', 'E93', 'E96', 'E97', 'E98', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'FT7', 'FT8', 
                              'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 
                              'TP10', 'TP7', 'TP8', 'TP9'],
                                }
                
        
        self.file_extensions = {'psg_ext': '**/*.set',
                                'ann_ext': '**/*scoring.tsv'} 
        
    def dataset_paths(self):
        return [
            '',
            ''
        ]
    
    
    # def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):

    #     return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs) 
    
    # def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

    #     if len(signals) > len(labels):
    #         return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
    #     elif len(labels) == len(signals) + 1:
    #         return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)
