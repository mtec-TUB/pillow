
# PSG Processing Toolkit

A comprehensive toolkit for processing and storing polysomnography (PSG) datasets in a harmonized way, supporting various file formats and automated signal processing.

---

## **Table of Contents**
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Supported File Formats](#supported-file-formats)
- [Processing steps](#processing-steps)
- [Output](#output)
- [Adding New Datasets](#adding-new-datasets)

---

## **Architecture**

The codebase is organized into three main components:

```
├── datasets/            # Dataset-specific logic
|   ├── file_handlers/             # Support for generic file formats
│   ├── base.py                    # Abstract base class for dataset
│   ├── registry.py                # Dataset registration and management
│   └── <dataset>.py               # One file per dataset
├── psg_processing/                # Core modules for processing
│   ├── core/
│   │   ├── dataset_explorer.py    # File search & channel analysis
│   │   ├── processor.py           # Main processing pipeline
│   │   └── signal_processor.py    # Filtering & resampling 
│   └── utils/                     # Logging, helper functions and classes
│       ├── alignment.py           # Enum for available alignment options
│       ├── config.py              # Helper class for process configuration
│       └── logging_manager.py     # Helper class for logging
├── config.yaml                    # Configuration file
└── process_dataset.py             # Main command-line interface

```

---

## **Quickstart**

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set configuration parameters**
    - Check default configuration in [config.yaml](/config.yaml)
    - modify them according to your needs (espacially the name of the dataset you want to process and where it is located (base_data_dir))

3. **Start the processing pipeline**
   ```bash
   python3 process_dataset.py
   ```

---

## **Processing steps**

The steps that are performed during processing depend heavily on your [config.yaml](/config.yaml). If you keep it as the default, the following will be applied (see [processor.py](/psg_processing/core/processor.py)):

- all files with matching extension are automatically detected and processed iterative (number of found psg and annot files has to match)
- the annotations for one file are loaded either from a seperate corresponding annotation file or from the input file itself 
- each file is checked for defined channel names and each found channel is processed seperately
- the output path is generated (see also [Output](#output)) and checked for invalid characters
- if not defined differently (param: overwrite) the file will be skipped if the output path already exists
- the signal is extracted from the file using the file handler
- the signal is truncated to whole epochs
- if resampling condition is not None (e.g.resample is 100):
    - type of the channel (digital/discrete or analog/continuos) is taken from dataset (can be determined with get_channel_types for new datasets)  
    - digital channels are resampled with a nearest neighbour interpolation and analog continous channels with polyphase filtering (mse.filter), where clipped values will be preserved to keep similarity or original data
    - signal is filtered based on AASM recomendation (e.g. EEG and EOG with 0.3-35Hz bandpass)
- the signal is reshaped to [num_epochs, epoch_dur * fs]
- the extracted annotations are checked for timing inconsistencies and aligned with the signal length
- the signal is cleaned/shortened based on annotations
    - max. 30min of wake epochs are kept at beginning and ending of signal
    - epochs labeled with 'Movement' or 'Artifact' are removed
- the signal, annotations and other information is saved to the output_path (see [Output](#output))


---

## **Output**

You can choose the desired output format inside the [config.yaml](/config.yaml). Possible formats are: 'npz', 'edf' and 'hdf5'.

If you choose 'npz', the pipeline will generate one .npz file per processed channel per file in a channel based folder structure. Thus for example three input .edf files with each five recorded channels will result in five output folders named after the channels with each containing three .npz files. It can happen that the output folders do not all contain the same number of files (number of input files) which means that some input files did not include all channels that were defined to be processed. This structure is espacially helpful if you plan to use the processed data as input for SleePyCo.

The `.npz` files can be extracted afterwars with `numpy.load()` and contain the following entries:

- **x**: Processed signal data array with dim [num_epochs, epoch_dur * fs]
- **y**: Sleep stage annotations ("W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4)
- **fs**: Sampling rate
- **ch_label**: Channel label (like it was originally found in the input file)
- **file_duration**: Total duration of the signal in seconds
- **epoch_duration**: Duration of each epoch (def: 30 seconds)
- **n_epochs**: Number of final processed epochs (without labeled as Movement or Artifact and with shortened wake periods at beginning and ending of file)

If there is a second annotation entry called **y2**, this results from a second scorer (for example in ISRUC dataset)

If the output format 'edf' is chosen, the pipeline will generate one edf file per input psg file, including all chosen channels. The annotations will be stored inside the edf annotation channel.

For the output format hdf5, the pipeline will generate one h5 file per input psg file, including all chosen channels. You will find the metadata as h5 attributes and the annotations as a h5 dataset called 'y'. The signals will be stored inside the h5 group 'signals', with each channel stored as a h5 dataset inside this group, with its corresponding metadata (channel name, sampling rate etc.) as attributes. You can use https://hdfviewer.com/ to check the structure before trying to load it programatically.

---

## **Adding New Datasets**

1. **Create a new dataset descriptive file:**
    - Add a file `datasets/<your_dataset>.py`.
    - Inherit from [`BaseDataset`](datasets/base.py) and implement the following methods:
        - `_setup_dataset_config` (specify file extensions as this is the only known property in the beginning)
        - `ann_parse`
        - `dataset_paths`
        - optionally `preprocess` (see [FDCSR dataset](/datasets/fdcsr.py))
        
    - to register the dataset use the decorator `@register_dataset("YOUR_DS_NAME")`.
    - Check if the polysomnography file extension is already covered by one of the [file_handlers](/datasets/file_handlers/). For standardized formats (e.g. EDF, EEGLAB, WFDB) you can use the existing generic handlers by specifying them inside your __init__ function. For all others you have to implement the following functions yourself inside the dataset script to be able to use the pipeline (Or you create a new generic handler if the format could be reused across other datasets):
        - get_channles()
        - read_signals()
        - get_signal_data()

2. **Explore dataset:**
    - search for existing channel names inside your dataset by setting the parameters `action` inside [config.yaml](/config.yaml) to 'get_channel_names' and run
    ```bash
    python process_dataset.py
    ```
    - if you want to apply resampling and filtering to your data you need to know which channels are analog (e.g. eeg channels) and which digital (e.g. Oxygen saturation). You can find it out by setting the parameters `action` inside [config.yaml](/config.yaml) to 'get_channel_types' and run the main script again (Check the results manually to prevent wrong processing)
3. **Define all pending dataset properties:**
    - Now you are good to go to define all pending properties inside your script to make use of the whole processing pipeline. Inside `_setup_dataset_config`, specify the collected channel names, channel types, channel groups (to apply filtering according to AASM) and optional intra-dataset- and inter-dataset-channel-mappings (can be used tp harmonize channel names, see [BESTAIR dataset](/datasets/bestair.py))
4. **Perform processing:**
    - set the `action` parameter back to 'process', choose all other configurations, start the pipeline and lay back :) 


---
