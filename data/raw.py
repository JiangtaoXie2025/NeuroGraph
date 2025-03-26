# data/raw

## Overview
This folder is intended to store the raw or original datasets for this project, such as unprocessed EEG or fMRI files, or uncompressed image or video data for human pose estimation.

Usually, these raw files can be very large and might not be fully committed to the repository. You can instead keep a small sample subset here for demonstration, or provide scripts or links so that users can download the entire dataset externally.

## Download or Acquisition
- Refer to the main project README or scripts in the root folder (for example: scripts/download_data.sh) for instructions on how to download or acquire the original data.
- Make sure that large files are not tracked by version control (for instance, by adding them to .gitignore).

## Suggested Folder Structure
You can organize raw data based on your dataset type. For example:

data/
  raw/
    subject1/
    subject2/
    ...
or
data/
  raw/
    images/
    labels/
    ...
Feel free to adapt the layout to your needs.

## Notes
- Only store small samples or minimal examples if needed for quick tests.
- Always keep track of dataset sources, licensing, and citations.
