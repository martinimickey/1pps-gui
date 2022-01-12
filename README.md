## Time Tagger 1PPS Tracking Graphical User Interface

This application allows you to track 1PPS signals with your Swabian Instruments Time Tagger.
The `CustomMeasurement` class `PPStracking` (located in `PPS/measurement.py`) groups the
events on the different channels with respect to a reference channel, visualizes, and stores them.
The data are stored in CSV format, every row represents a group of tags and are labeled by a system UTC timestamp.

## Requirements

Python 3.8 is required and the packages listed in `requirements.txt`.

## How to use it

Clone the repository and create a working directory (outside the repository).
The working directory will be used to store settings, a data folder can be specified via the GUI.
Run the `PPS_App.py` script from your working directory by calling

```
python "<your cloned repository path>/PPS_App.py"
```

in an appropriate command shell. The GUI opens
