import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from decimal import Decimal
from mne.io import read_raw_eeglab

from psg_processing.core import Dataset_Explorer
from datasets.base import BaseDataset
from datasets.registry import register_dataset


@register_dataset("EESM23")
class EESM23(BaseDataset):
    """Ear-EEG Sleep Monitoring 2023 (EESM23) dataset"""
    
    def __init__(self):
        super().__init__("EESM23","Ear-EEG Sleep Monitoring 2023 (EESM23)", keep_folder_structure=False)
    
    def _setup_dataset_config(self):
        self.ann2label = {
                        "Wake": 0,   # Wake
                        "N1": 1,  # NREM Stage 1
                        "N2": 2,  # NREM Stage 2
                        "N3": 3,  # NREM Stage 3
                        "REM": 4,   # REM sleep
                        "Artefact": 6
                        }
        
        
        self.channel_names = ['F3', 'EMGl', 'F4', 'O2', 'O1', 'EMGr', 'EOGr', 'EMGc', 'C4', 'C3', 'M2', 'M1', 'EOGl']
        
        
        self.channel_types = {'analog': ['F3', 'EMGl', 'F4', 'O2', 'O1', 'EMGr', 'EOGr', 'EMGc', 'C4', 'C3', 'M2', 'M1', 'EOGl'], 'digital': []}
        
        
        self.channel_groups = {'eeg_eog': ['F3', 'F4', 'O2', 'O1', 'C4', 'C3', 'EOGr', 'EOGl','M2', 'M1'],
                                'emg': ['EMGl', 'EMGr', 'EMGc'],
                                }
                
        
        self.file_extensions = {'psg_ext': '**/*_task-sleep_acq-PSG_eeg.set',
                                'ann_ext': '**/*_task-sleep_acq-scoring_events.tsv'}
    def dataset_paths(self) -> tuple[str, str]:
        """
        EESM23 dataset paths.
        """
        data_dir = "Ear-EEG Sleep Monitoring 2023 (EESM23)"
        ann_dir = "Ear-EEG Sleep Monitoring 2023 (EESM23)"
        return data_dir, ann_dir
    
    def ann_parse(self, ann_fname):
        annot = pd.read_csv(ann_fname,sep='\t', header=0)

        ann_stage_events = []
        start_time_label = None

        for i, row in annot.iterrows():
            start = row['onset']

            if start_time_label == None:
                start_time_label = Decimal(str(start))

            duration = row['duration']
            stage = row['scoring']
            ann_stage_events.append({'Stage': stage,
                                        'Start': float(Decimal(str(start)) - start_time_label),
                                        'Duration': duration})
            

        return ann_stage_events, float(start_time_label)
    
    def align_front(self, logger, alignment, pad_values, epoch_duration, delay_sec, signal, labels, fs):
        if not (delay_sec*fs).is_integer():
            raise Exception("Annotations start at timestamp outside of sample rate")

        return self.base_align_front(logger, delay_sec, alignment, pad_values, epoch_duration, signal, labels,fs) 

    def align_end(self, logger, alignment, pad_values, psg_fname, ann_fname, signals, labels):

        if len(labels) == len(signals) + 1:
            return self.base_align_end_labels_longer(logger, alignment, pad_values, signals, labels)

        if len(signals) > len(labels):
            return self.base_align_end_signals_longer(logger, alignment, pad_values, signals, labels)
    
    def preprocess(self, data_dir, ann_dir, output_dir):
        print("\n EESM23 files originally contain NaN values in signals. \n")
        
        execute_preprocess = input("Do you want to interpolate over these NaN values now (as recommended from the dataset authors)? (Y/N) ")
        
        if str(execute_preprocess).lower() == "y":

            self.interpolate_files(data_dir, ann_dir, output_dir)

            self.file_extensions['psg_ext'] = '**/*_task-sleep_acq-PSG_eeg_interpolated.set'
            
            if str(input("Do you want to continue with processing now? (Y/N) ")).lower() == "n":
                return False
        
        if str(input("Do you want to use the interpolated files for processing (if existing)? (Y/N) ")).lower() == "y":
            self.file_extensions['psg_ext'] = '**/*_task-sleep_acq-PSG_eeg_interpolated.set'
        return True
    

    def interpolate_files(self, data_dir, ann_dir, output_dir):
        # Get files using dataset-specific extensions
        explorer = Dataset_Explorer(
            logger=None,
            psg_file_handler=None,
            data_dir=data_dir,
            ann_dir=ann_dir,
            **self.file_extensions,
        )
        psg_fnames, _ = explorer.get_files()

        # Process each file
        for psg_idx, psg_fname in enumerate(psg_fnames):
            print(f"\n--- Preprocessing file {psg_idx+1}/{len(psg_fnames)} ---")
            
            raw_data = read_raw_eeglab(psg_fname, verbose=False, preload=True)

            signals = raw_data.get_data()
            fs = raw_data.info['sfreq']

            signals_interpolated = self.interpolateOverNans(signals, fs)
                
            raw_data._data= signals_interpolated
            path,ext = os.path.splitext(psg_fname)
            raw_data.export(os.path.join(path + '_interpolated' + ext), fmt='eeglab', verbose=False,overwrite=True)
            print(f"Interpolated file saved as {os.path.basename(path + '_interpolated' + ext)}")


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

