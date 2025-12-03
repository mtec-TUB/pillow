
# PSG Processing Toolkit

A comprehensive, object-oriented toolkit for processing polysomnography (PSG) datasets, supporting various file formats and automated signal processing.

---

## **Table of Contents**
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Running the Processing Pipeline](#running-the-processing-pipeline)
- [Supported File Formats](#supported-file-formats)
- [Processing steps](#processing-steps)
- [Output](#output)
- [Adding New Datasets](#adding-new-datasets)
- [Helpful Notes](#helpful-notes)

---

## **Architecture**

The codebase is organized into three main components:

```
├── datasets/            # Dataset-specific logic
│   ├── base.py                    # Abstract base class for dataset
│   ├── registry.py                # Dataset registration and management
│   └── <dataset>.py               # One file per dataset
├── psg_processing/                # Core modules for processing
│   ├── core/
│   │   ├── dataset_explorer.py    # File search & channel analysis
│   │   ├── processor.py           # Main processing pipeline
│   │   └── signal_processor.py    # Filtering & resampling
│   ├── file_handlers/             # Support for various file formats
│   └── utils/                     # Logging, helper functions
└── process_dataset.py             # Main command-line interface
```

---

## **Quickstart**

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the processing pipeline**
   ```bash
   python process_dataset.py --dataset <DATASETNAME> --action prepare --base_data_dir <PATH_TO_DATA>
   ```
   Example:
   ```bash
   python process_dataset.py --dataset ABC --action prepare --base_data_dir /media/linda/Elements/sleep_data
   ```

   For more examples and options, see [Running the Processing Pipeline](#running-the-processing-pipeline).

---

## **Running the Processing Pipeline**

The main script is [`process_dataset.py`](process_dataset.py). It provides the following actions:

- **prepare**: Starts processing and harmonization of a dataset.
- **get_channel_names**: Lists all channels of a dataset.
- **get_channel_types**: Lists all channel types of a dataset.

**Key arguments:**
- `--dataset`: Name of the dataset (e.g. ABC, MESA, HOMEPAP, ...)
- `--base_data_dir`: Base directory of all raw data
- `--data_dir`, `--ann_dir`, `--output_dir`: Optional, to manually set paths
- `--resample`: Target sampling rate (e.g. 100) or "None"
- `--channels`: List of channels to process
- `--num-jobs`: Parallelization option 
- `--overwrite`: Overwrite existing output files

**Example calls:**
```bash
python process_dataset.py --dataset ABC --action get_channel_names
python process_dataset.py --dataset MESA --resample None --overwrite
python process_dataset.py --dataset HOMEPAP --channels EEG1 EOG1 --resample 100
```


---

## **Supported File Formats**

The following file types can be handled to extract the signal from (see [file_handlers](/psg_processing/file_handlers/)). New file handlers can be added easily.

- **EDF** (European Data Format) - `.edf`
- **HDF5** (Hierarchical Data Format) - `.h5`
- **WFDB** (WaveForm DataBase) - `.hea`
- **MAT** (Matlab Data Format) - `.mat` (maybe not suitable for all different kind of mat-files and likely dataset-specific)

These formats require dataset-specific handling due to varying structures (Each dataset requires its own CSV handler, e.g., `DreamtCSVHandler` for DREAMT dataset):
- **CSV** (Comma-Separated Values) - `.csv`

For the annotation files there is no common handler because most datasets have a unique annotation saving format. The base parsing strategy can be found in [base.py](/datasets/base.py) ann_parse() function, and can be overwritten individually for each dataset in [datasets](/datasets/). 

---

## **Processing steps**

These steps will be performed to process the polysomnography datasets (see [processor.py](/psg_processing/core/processor.py)):

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

The pipeline will generate one .npz file per processed channel per file in a channel based folder structure. Thus three input .edf files with each five recorded channels will result in five output folders named after the channels with each containing three .npz files. It can happen that the output folders do not all contain the same number of files (number of input files) which means that some input files did not include all channels that were defined to be processed.

If not defined differently the files are saved in the same subfolder structure as they were found in the data directory. The output folder (DS_harmonized) will have two subfolders corresponding to the harmonized files resampled to 100Hz and filtered (100Hz_filt) and in the original sample rate (orig).

Example: the 100Hz resampled signal of channel ECG found in
`/sleep_data/ABC_dataset/polysomnography/follow_up/abc_001.edf` 
will be saved as 
`/sleep_data/ABC_dataset/ABC_harmonized/100Hz_filt/follow_up/ECG/ECG_abc_001.npz`

The `.npz` files can be extracted with `numpy.load()` and contain the following entries:

- **x**: Processed signal data array with dim [num_epochs, epoch_dur * fs]
- **y**: Sleep stage annotations ("W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4)
- **fs**: Sampling rate
- **ch_label**: Channel label (like it was originally found in the input file)
- **start_datetime**: Start date and time of the recording if existed
- **file_duration**: Total duration of the original signal in seconds
- **epoch_duration**: Duration of each epoch (def: 30 seconds)
- **n_all_epochs**: Total number of epochs before cleaning
- **n_epochs**: Number of final processed epochs (without labeled as Movement or Artifact and with shortened wake periods at beginning and ending of file)

If there is a second annotation entry called **y2**, this results from a second scorer (for example in ISRUC dataset)

---

## **Adding New Datasets**

1. **Create a new dataset descriptive file:**
    - Add a file `datasets/<your_dataset>.py`.
    - Inherit from [`BaseDataset`](datasets/base.py) and implement the methods `_setup_dataset_config` (specify file extensions as this is the only known property in the beginning), `ann_parse`, optionally `preprocess` (see [FDCSR dataset](/datasets/fdcsr.py)) and `dataset_paths`.
    - to register the dataset use the decorator `@register_dataset("YOURNAME")`.
    - Check if the polysomnography file extension is already covered by one of the [file_handlers](/psg_processing/file_handlers/):
        - **For standardized formats** (EDF, H5, WFDB): Use existing generic handlers
        - **For CSV files**: Create a dataset-specific handler (e.g., `your_dataset_csv_handler.py`) since CSV structures vary significantly between datasets
        - **For new formats**: Create a new generic handler if the format could be reused across datasets

2. **Explore dataset:**
    - search for existing channel names in dataset with:
     ```bash
      python process_dataset.py --dataset <your_dataset> --action get_channel_names
      ```
    - determine the type of channels with:
    ```bash
    python process_dataset.py --dataset <your_dataset> --action get_channel_types
    ```
    - Channels like 'Oxygen saturation', 'Light' or 'Position' are mostly digital, while EEG, ECG and EMG channels should be analog. Check the results manually to prevent wrong processing.
3. **Define all pending dataset properties:**
    - In `_setup_dataset_config`, specify channel names, channel types, channel groups and optional alias_mappings (can be used if many different channel names appear that seem to belong all to the same channel, see [BESTAIR dataset](/datasets/bestair.py))
4. **Perform processing:**
    - see [Running the Processing Pipeline](#running-the-processing-pipeline)


---
