# PSG Processing Toolkit

A comprehensive, object-oriented toolkit for processing polysomnography (PSG) datasets with support for multiple file formats and automated signal processing. The system uses a modular architecture with dataset-specific processors and a unified processing pipeline.

## **Architecture Overview**

The codebase is organized into three main components:

```
├── dataset_processors/            # Dataset-specific processing logic
│   ├── __init__.py                # Processor registry and imports
│   ├── base.py                    # Abstract base processor class
│   ├── registry.py                # Dynamic processor registration
│   ├── abc_processor.py           # ABC dataset processor
│   ├── mesa_processor.py          # MESA dataset processor
│   ├── sof_processor.py           # SOF dataset processor
│   └── ...                        # Other dataset processors
├── psg_processing/                # Core processing modules
│   ├── __init__.py                # Package interface
│   ├── core/
│   │   ├── dataset_explorer.py    # File discovery & channel analysis
│   │   ├── processor.py           # Main file process pipeline
│   │   └── signal_processor.py    # Signal filtering & resampling
│   ├── file_handlers/             # Multi-format file support
│   │   ├── base.py                # Base handler interface
│   │   ├── edf_handler.py         # EDF/BDF file support
│   │   ├── h5_handler.py          # HDF5 file support
│   │   ├── csv_handler.py         # CSV file support
│   │   ├── wfdb_handler.py        # WFDB file support
│   │   └── factory.py             # Handler factory pattern
│   └── utils/
│       └── logging_manager.py     # Centralized logging
├── dataset_preprocessings/        # One-time dataset reorganization
└── process_dataset.py            # Entry-Point: Unified command-line interface
```

## **Quick Start**

Install requirements.txt


### Command Line Usage

```bash
python procress_dataset.py --dataset ABC --action get_channel_names

python process_dataset.py --dataset MSP --resample None --overwrite True

python process_dataset.py --dataset SHHS
```

## **Supported File Formats**

- **EDF** (European Data Format) - `.edf`
- **HDF5** (Hierarchical Data Format) - `.h5` 
- **CSV** (Comma-Separated Values) - `.csv`
- **WFDB** (WaveForm DataBase) - `.hea`

## **Key Features**

### File Handlers
- **Extensible Design**: Easy to add new file formats
- **Unified Interface**: Consistent API across all formats

### Signal Processing
- **Digital/Analog Detection**: Automatic signal type classification to keep the resulting signal as close to the original one as possible
- **Resampling**: Polyphase filtering with clipping preservation
- **Filtering**: AASM suggested bandpass filtering per channel type
  - EEG and EOG channels: 0.3-35 Hz
  - EMG channels: 10+ Hz (high-pass)
  - ECG channels: 0.3+ Hz (high-pass)
  - Abdominal/Respiratory signals: 0.1-15 Hz
  - Nasal pressure: 0.03+ Hz (high-pass)
  - Snoring: 10+ Hz (high-pass)

### Dataset Processing
- **Batch Processing**: Handle multiple files and channels efficiently
- **Progress Tracking**: Comprehensive logging and progress reporting
- **Error Handling**: Robust error handling with detailed logging
- **Output Management**: Organized output structure with metadata

### Logging & Monitoring
- **Multi-level Logging**: Console and file logging with configurable levels
- **Per-channel Logs**: Individual log files for detailed tracking
- **Progress Reporting**: Real-time processing status updates

## **Development**

### Adding New Dataset


### Adding New File Formats

1. Create a new handler in `file_handlers/`:
```python
from .base import FileHandler

class NewFormatHandler(FileHandler):
    def __init__(self):
        super().__init__()
        self.file_extension = ".new"
    
    def get_channels(self, filepath):
        # Implement channel extraction
        pass
    
    def get_signal_info(self, logger, filepath, epoch_duration, channel):
        # Implement signal extraction
        pass
```

2. Register in `factory.py`:
```python
self.handlers = {
    # ... existing handlers
    '.new': NewFormatHandler()
}
```
