EEG Platform - cute little project
simple eeg signal processing toolkit - load, clean, analyze, and stream brain data



to install
pip install -e .


some of the commands to run this project:
quick demo (generates fake eeg, processes it, saves plots)
eeg-platform demo

inspect a csv
eeg-platform inspect data/recording.csv --sfreq 256

full pipeline
eeg-platform run data/recording.csv --sfreq 256 --outdir outputs/

to extract
eeg-platform features-spectral data/recording.csv --sfreq 256 --out features.json

stream from device
eeg-platform stream-record --board synthetic --duration 30


run `eeg-platform --help` for all commands

