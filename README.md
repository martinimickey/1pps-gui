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

in an appropriate Python command shell.
The GUI opens and is primarily controlled via the menu.

Prepare your Time Tagger under `File -> Hardware Settings`. The most important task here is to assign roles (`Reference`, `Signal channel`, `External clock`) to the different input channels you want to use. To configure the Time Tagger's software clock, you can use the `Clock frequency`and`Clock divider` setttings.

The storage behavior is controlled under `File -> Storage settings`.
Choose a folder to store your data and a time of the day when a new file is started (default is midnight).
You can also choose how many data points to show in the display.
If you want to store the Time Taggers's sensor data for debugging, activate the checkbox.

Finally, you can start the measurement by selecting `Measurement -> Start measurement`.
