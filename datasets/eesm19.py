import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from decimal import Decimal
from mne.io import read_raw_eeglab
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psg_processing.core import Dataset_Explorer
from datasets.base import BaseDataset
from datasets.registry import register_dataset

from datasets.file_handlers import EEGLABHandler


@register_dataset("EESM19")
class EESM19(BaseDataset):
    """Ear-EEG Sleep Monitoring 2019 (EESM19) dataset"""
    
    def __init__(self):
        super().__init__("EESM19","Ear-EEG Sleep Monitoring 2019 (EESM19)", keep_folder_structure=False)

        self._file_handler = EEGLABHandler()
    
    def _setup_dataset_config(self):
        self.ann2label = {
                        1: "W",   # Wake
                        3: "N1",  # NREM Stage 1
                        4: "N2",  # NREM Stage 2
                        5: "N3",  # NREM Stage 3
                        2: "REM",   # REM sleep
                        7: "UNK",    # Artefact
                        }
        
        
        self.channel_names = ['ERC', 'ELC', 'EOGl', 'ELA', 'O2', 'EMGr', 'EMGl', 'ELT', 'M1', 'C3', 'EOGr', 'O1', 'C4', 'ERA', 'EMGc', 'ERE', 
                              'ERT', 'F4', 'ELE', 'ELI', 'ELB', 'ERI', 'ERB', 'M2', 'F3']
        
        self.inter_dataset_mapping = {
            "F3": self.Mapping(self.TTRef.F3, None),
            "F4": self.Mapping(self.TTRef.F4, None),
            "O1": self.Mapping(self.TTRef.O1, None),
            "O2": self.Mapping(self.TTRef.O2, None),
            "C3": self.Mapping(self.TTRef.C3, None),
            "C4": self.Mapping(self.TTRef.C4, None),
            "M1": self.Mapping(self.TTRef.LPA, None),
            "M2": self.Mapping(self.TTRef.RPA, None),
            "EOGl": self.Mapping(self.TTRef.EL, None),
            "EOGr": self.Mapping(self.TTRef.ER, None),
            "EMGc": self.Mapping(self.TTRef.EMG_CHIN, None),
            "EMGl": self.Mapping(self.TTRef.EMG_LCHIN, None),
            "EMGr": self.Mapping(self.TTRef.EMG_RCHIN, None),
        }
        
        self.channel_types = {'analog': ['ELA', 'ELT', 'C3', 'ERC', 'EMGc', 'ERB', 'M2', 'ERE', 'M1', 'O1', 'ELB', 'EOGr', 'C4', 'O2', 'ELC', 
                                         'F3', 'ERA', 'EMGl', 'EMGr', 'EOGl', 'ERT', 'ELE', 'ERI', 'ELI', 'F4'], 
                              'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['ERC', 'ELC', 'EOGl', 'ELA', 'O2', 'ELT', 'M1', 'C3', 'EOGr', 'O1', 'C4', 'ERA', 'ERE', 
                              'ERT', 'F4', 'ELE', 'ELI', 'ELB', 'ERI', 'ERB', 'M2', 'F3'],
                                'emg': ['EMGl', 'EMGr', 'EMGc'],
                                }
                
        
        self.file_extensions = {'psg_ext': '**/*task-sleep_acq-*_eeg.set',  # include PSG and earEEG files, but exclude ‘auditory steady state responses’ (ASSR) and electrodeTestLab files
                                'ann_ext': '**/*scoring1_events.tsv'} 
        
    def dataset_paths(self):
        return [
            '',
            ''
        ]
    
    def get_file_identifier(self, psg_fname=None, ann_fname=None):
        psg_id, ann_id = None, None
        if psg_fname:
            psg_id = Path(psg_fname).parent
        if ann_fname:
            ann_id = Path(ann_fname).parent
        return psg_id, ann_id
    
    def ann_parse(self, ann_fname):

        base_fname = ann_fname.replace('scoring1_events.tsv','')

        ann_stage_events_1 = []
        ann_stage_events_2 = []

        ann_fname1 = base_fname + 'scoring1_events.tsv'
        if os.path.exists(ann_fname1):
            annot = pd.read_csv(ann_fname1,sep='\t', header=0)
            start_time_label = None
            for i, row in annot.iterrows():
                start = round(row['onset'])

                if start_time_label == None:
                    start_time_label = start

                duration = row['duration']
                stage = row['Scoring1']
                ann_stage_events_1.append({'Stage': stage,
                                            'Start': start - start_time_label,
                                            'Duration': duration})
            

        ann_fname2 = base_fname + 'scoring2_events.tsv'
        if os.path.exists(ann_fname2):
            start_time_label = None
            annot2 = pd.read_csv(ann_fname2,sep='\t', header=0)
            for i, row in annot2.iterrows():
                start = round(row['onset'])

                if start_time_label == None:
                    start_time_label = start

                duration = row['duration']
                stage = row['Scoring2']
                ann_stage_events_2.append({'Stage': stage,
                                            'Start': start - start_time_label,
                                            'Duration': duration})
                
        events_file = ann_fname.replace("scoring1_events", "diary_events")
        events = pd.read_csv(events_file,sep='\t', header=0)

        lights_off = events.loc[events['trial_type'] == 'Lights Out', 'onset']
        if len(lights_off) == 1:
            lights_off = lights_off.iloc[0]
        else:
            raise Exception(f"Expected exactly one 'Lights Out' event in {events_file}, but found {len(lights_off)}.")
            
        return [ann_stage_events_1, ann_stage_events_2], start_time_label, lights_off, None
    
    def ann_label(self, logger, ann_stage_events: List[List[Dict]], STAGE_DICT, epoch_duration: int):
        """
        Convert multi-scorer sleep stage events to epoch-wise labels for ISRUC dataset.
        Returns 2D array (n_epochs, n_scorers).
        """
        labels = [np.array([]), np.array([])]

        for i, annotation in enumerate(ann_stage_events):  # two scorers
            total_duration = 0
            for event in annotation:
                onset_sec = int(event['Start'])
                duration_sec = int(event['Duration'])
                ann_str = event['Stage']

                # Sanity check
                assert onset_sec == total_duration, f"Onset sec of epoch is {onset_sec} but last epoch ended at {total_duration}"

                # Get label value
                if ann_str in self.ann2label:
                    label = self.ann2label[ann_str]
                else:
                    logger.info(f"Something unexpected: label {ann_str} not found")
                    raise Exception(f"Something unexpected: label {ann_str} not found")
                label = STAGE_DICT[label]   # Map to standardized label

                # Compute # of epoch for this stage
                if duration_sec % epoch_duration != 0:
                    logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                    raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
                duration_epoch = int(duration_sec / epoch_duration)

                # Generate sleep stage labels
                label_epoch = np.ones(duration_epoch, dtype=np.int32) * label
                labels[i] = np.append(labels[i], label_epoch)

                total_duration += duration_sec

                # logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                #     onset_sec, duration_sec, label, ann_str
                # ))

        # Pad shorter annotation to match longer one
        if len(labels[0]) != len(labels[1]):
            max_len = max(len(labels[0]), len(labels[1]))
            labels[0] = np.pad(labels[0], (0, max_len - len(labels[0])), mode='constant', constant_values=6)
            labels[1] = np.pad(labels[1], (0, max_len - len(labels[1])), mode='constant', constant_values=6)

        labels = np.array(labels).T  # Transpose to (n_epochs, n_scorers)
        
        return labels
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):
        logger.info("Alignment of scorer 1")
        start_time_shift, signal1, labels1 = self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels[:,0],fs) 
        logger.info("Alignment of scorer 2")
        start_time_shift, signal2, labels2 = self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels[:,1],fs)

        assert len(signal1) == len(signal2), f"Length mismatch after front alignment: signal1={len(signal1)}, signal2={len(signal2)}"
        assert len(labels1) == len(labels2), f"Length mismatch after front alignment: labels1={len(labels1)}, labels2={len(labels2)}"
        
        return start_time_shift, signal1, np.array([labels1, labels2]).T  # Return (signal, (n_epochs, n_scorers))
    
    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) > len(signals):
            logger.info("Alignment of scorer 1")
            signals1, labels1 = self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels[:,0])
            logger.info("Alignment of scorer 2")
            signals2, labels2 = self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels[:,1])
            assert len(signals1) == len(signals2), f"Length mismatch after end alignment: signals1={len(signals1)}, signals2={len(signals2)}"
            assert len(labels1) == len(labels2), f"Length mismatch after end alignment: labels1={len(labels1)}, labels2={len(labels2)}"
            return signals1, np.array([labels1, labels2]).T  # Return (signals, (n_epochs, n_scorers))

        if len(signals) > len(labels):
            logger.info("Alignment of scorer 1")
            signals1, labels1 = self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels[:,0])
            logger.info("Alignment of scorer 2")
            signals2, labels2 = self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels[:,1])
            assert len(signals1) == len(signals2), f"Length mismatch after end alignment: signals1={len(signals1)}, signals2={len(signals2)}"
            assert len(labels1) == len(labels2), f"Length mismatch after end alignment: labels1={len(labels1)}, labels2={len(labels2)}"
            return signals1, np.array([labels1, labels2]).T  # Return (signals, (n_epochs, n_scorers))
        
        raise Exception(f"Unexpected case during end alignment: {psg_fname}, len(signals)={len(signals)}, len(labels)={len(labels)}")

    def preprocess(self, data_dir, ann_dir, output_dir):
        return EESM_Preprocessor(self).preprocess(data_dir, ann_dir, output_dir)

class EESM_Preprocessor:

    def __init__(self, dataset):
        self.dataset = dataset

    def preprocess(self, data_dir, ann_dir, output_dir):
        print("\n Files originally contain NaN values in signals. \n")
        
        execute_preprocess = input("Do you want to interpolate over these NaN values now (as recommended from the dataset authors)? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":

            self.interpolate_files(data_dir, ann_dir, output_dir)
            
            if str(input("Do you want to continue with processing now? (Y/N) ")).lower() == "n":
                return False
        
        if str(input("Do you want to use the interpolated files for processing (if existing)? (Y/N) ")).lower() == "y":
            self.dataset.file_extensions['psg_ext'] = self.dataset.file_extensions['psg_ext'].replace('*.set','*_interpolated.set')
        return True
    

    def interpolate_files(self, data_dir, ann_dir, output_dir):
        # Get files using dataset-specific extensions
        explorer = Dataset_Explorer(
            logger=None,
            dataset = self.dataset,
            data_dir=data_dir,
            ann_dir=ann_dir
        )
        psg_fnames, _ = explorer.get_files()

        # Process each file
        for psg_idx, psg_fname in enumerate(psg_fnames):
            print(f"\n--- Preprocessing file {psg_idx+1}/{len(psg_fnames)} ---")

            path,ext = os.path.splitext(psg_fname)
            output_path = os.path.join(path + '_interpolated' + ext)
            if os.path.exists(output_path):
                print(f"Interpolated file already exists: {output_path}, skipping interpolation for this file.")
                continue
            
            try:
                raw_data = read_raw_eeglab(psg_fname, verbose=False, preload=True)
            except Exception as e:
                print(f"Error reading {psg_fname} with MNE: {e}. Skipping interpolation for this file.")
                continue

            signals = raw_data.get_data()
            fs = raw_data.info['sfreq']

            signals_interpolated = self.interpolateOverNans(signals, fs)
                
            raw_data._data= signals_interpolated
            raw_data.export(output_path, fmt='eeglab', verbose=False,overwrite=True)
            print(f"Interpolated file saved as {output_path}")


    def findRuns(self,input,noWarning=False):

        runStarts=[]
        runLengths=[]

        if len(input)==0:
            print('Warning: findRuns received empty input')
            return np.array(runStarts), np.array(runLengths)

        sequence=np.asarray(input).reshape(-1)
        if ~(sequence.all() | ((1-sequence).all())):
            sequence=sequence.astype(int) #diff complains if it's boolean
            changes=np.diff([0, *sequence, 0])
            runStarts=(changes>0).nonzero()[0]
            runEnds=(changes<0).nonzero()[0]
            runLengths=runEnds-runStarts
            assert all(runLengths>0)
        elif sequence.all():
            runStarts=np.array([0])
            runLengths=np.array([len(sequence)])
        elif (1-sequence).all():
            if not noWarning:
                print('Warning: findRuns received vector of all zeros')

        return np.array(runStarts), np.array(runLengths)

    def interpolateOverNans(self,allDeriv,fs):
        #we can't have nans at the end:
        allDeriv[np.isnan(allDeriv[:,0]),0]=0
        allDeriv[np.isnan(allDeriv[:,-1]),-1]=0


        for iDeriv in range(allDeriv.shape[0]):

            nanSamples=np.isnan(allDeriv[iDeriv,:]).nonzero()[0]

            if nanSamples.size>0:
                [nanStart, nanDur]=self.findRuns(np.isnan(allDeriv[iDeriv,:]))
                nanDur=nanDur-1
                realSamples=np.unique([nanStart-1, (nanStart+nanDur)+1])

                distanceToReal=nanSamples*0
                counter=0
                for iRun in range(len(nanDur)):
                    distanceToReal[range(counter,counter+nanDur[iRun])]=[*range(int(np.floor(nanDur[iRun]/2))), *range(int(np.ceil(nanDur[iRun]/2)),0,-1) ]
                    counter=counter+nanDur[iRun]

                interpValues=interp1d(realSamples,allDeriv[iDeriv,realSamples])(nanSamples)
                interpValues=interpValues*np.exp(-distanceToReal/(fs*1))

                # plt.plot(allDeriv[iDeriv,:],label='Before Interp')

                allDeriv[iDeriv,nanSamples]=interpValues
                # plt.plot(allDeriv[iDeriv,:],'--',label='Interpolated')
                # plt.show()

        return allDeriv

