# data/processed

## Overview
This folder holds all processed or intermediate files, which are typically generated after running preprocessing steps on the raw data. This saves you from having to redo heavy transformations or conversions every time you train or evaluate.

## How to Generate Processed Data
1. Place or download your raw data into data/raw/.
2. Run the data preprocessing script, for example:
   python scripts/preprocess_data.py
3. The script will generate cleaned or transformed data, which should appear here in data/processed/.

## Folder Structure
Depending on the project, you can separate data by modality or domain. For example:

data/
  processed/
    eeg/
    fmri/
    images/
    ...
Adjust as needed to fit your workflow.

## Notes
- You can commit small processed files or subsets to allow users to test the code quickly.
- Large processed files can be ignored if they exceed typical size limits or remain outside version control.
- Document any transformations and normalizations to ensure reproducibility.
