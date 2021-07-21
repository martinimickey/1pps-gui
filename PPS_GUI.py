import tkinter as tk
from tkinter import BooleanVar, IntVar, StringVar, Variable, ttk, messagebox
from threading import Thread
from datetime import time
from os.path import dirname, realpath, exists
import pickle
from time import sleep
from typing import Optional, Dict, Union, List, Callable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import TimeTagger
from enum import Enum
from PPStracking import PpsTracking

SettingsType = Dict[Union[int, str], Union[Variable, "SettingsType"]]

if exists("PULSE_STREAMER"):
    # If a file PULSE_STREAMER is found in the current directory, DIVIDER is set to 16.
    # Just needed for testing with Pulse Streamer, should otherwise always be 1
    DIVIDER = 16
else:
    DIVIDER = 1


def this_folder():
    return dirname(realpath(__file__))


class ChannelRoles(Enum):
    UNUSED = "Unused"
    REFERENCE = "Use as Reference"
    CHANNEL = "Signal channel"
    CLOCK = "10 MHz clock"


class DisplayUpdater(Thread):
    def __init__(self, measurement: PpsTracking, plot: Figure, channels: List[int], add_message: Callable[[str], None]):
        super().__init__()
        self.measurement = measurement
        self.plot = plot
        self.last_tag = -1
        self.last_message = 0
        self.channels = channels
        self.channel_names = measurement.getChannelNames()
        self.add_message = add_message

    def run(self):
        while self.measurement.isRunning():
            timetag_number, message_number = self.measurement.getMeasurementStatus()
            if timetag_number > self.last_tag:
                try:
                    data = self.measurement.getData()
                    index = self.measurement.getIndex()
                    for ch in range(data.shape[0]):
                        axis = self.plot.get_axes()[ch]
                        axis.clear()
                        axis.set_ylabel(f"{self.channel_names[ch]}\nOffset, ps")
                        axis.scatter(index, data[ch, :])
                    axis.set_xlabel("Time tag index")
                    self.plot.canvas.draw_idle()
                except:
                    print("Display error")
                self.last_tag = timetag_number
            if message_number > self.last_message:
                for msg in self.measurement.getMessages(self.last_message):
                    self.add_message(msg)
                self.last_message = message_number
            sleep(0.1)


class PPS_App:
    """The graphical user interface for tracking PPS signals."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.iconbitmap(this_folder() + "/iconTT.ico")
        self.fig = Figure()
        self.measurement: Optional[PpsTracking] = None
        self.tagger: Optional[TimeTagger.TimeTagger] = None
        taggers = TimeTagger.scanTimeTagger()
        print(taggers)
        self.settings: SettingsType = dict(serial=tk.StringVar(self.root, taggers[0] if taggers else ""),
                                           channels={ch: tk.StringVar(self.root, ChannelRoles.UNUSED.value) for ch in range(1, 9)},
                                           channel_names={ch: tk.StringVar(self.root, "") for ch in range(1, 9)},
                                           storage_folder=StringVar(self.root, ""),
                                           store_debug_info=BooleanVar(self.root, False),
                                           store_unscaled_data=BooleanVar(self.root, False),
                                           storage_time={key: StringVar(self.root, "00", key) for key in ("hour", "minute", "second")},
                                           max_live_tags=IntVar(self.root, 300),
                                           clock_divider=IntVar(self.root, 1))
        self._load_settings()
        for var in self.settings["storage_time"].values():
            var.trace("w", self._adjust_storage_time)
        self.settings["storage_folder"].trace("w", self._adjust_storage_folder)
        self.settings["max_live_tags"].trace("w", self._adjust_max_live_tags)
        panes = tk.PanedWindow(orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=1)
        self.messages = tk.Listbox(panes)
        panes.add(self.messages)
        self.fig_frame = tk.Frame(panes)
        panes.add(self.fig_frame)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.state("zoomed")

    def run(self):
        """Run the tkinter application."""
        self.root.wm_title("PPS tracking")
        self._setup_menu()
        self._setup_figure()
        tk.mainloop()

    def _setup_menu(self):
        """Setup of the tkinter menus."""
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Hardware settings", command=lambda: SettingsWindow(self, self.settings))
        file_menu.add_command(label="Storage settings", command=lambda: StorageConfigWindow(self, self.settings))
        file_menu.add_command(label="Close", command=self._quit)
        meas_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Measurement", menu=meas_menu)
        meas_menu.add_command(label="Start measurement", command=self._start_measurement)
        meas_menu.add_command(label="Stop measurement", command=self._stop_measurement)

    def _setup_figure(self):
        """Prepare graph area for pyplot."""
        canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, self.fig_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def _adjust_storage_time(self, *args):
        if self.measurement:
            self.measurement.setNewFileTime(time(**{key: int(self.settings["storage_time"][key].get()) for key in ["hour", "minute", "second"]}))

    def _adjust_storage_folder(self, *args):
        if self.measurement:
            self.measurement.setFolder(self.settings["storage_folder"].get())

    def _adjust_max_live_tags(self, *args):
        if self.measurement:
            self.measurement.setTimetagsMaximum(self.settings["max_live_tags"].get())

    def _start_measurement(self):
        if not exists(self.settings["storage_folder"].get()):
            self.add_message("Data folder does not exist")
            return
        clock = None
        reference_name = ""
        reference = 0
        for ch, val in self.settings["channels"].items():
            if val.get() == ChannelRoles.REFERENCE.value:
                reference = ch
                reference_name = self.settings["channel_names"][ch].get()
            elif val.get() == ChannelRoles.CLOCK.value:
                clock = ch
        if not reference:
            self.add_message("No reference given")
            return

        try:
            self.tagger = TimeTagger.createTimeTagger(self.settings["serial"].get())
        except RuntimeError:
            serial = self.settings["serial"].get()
            self.add_message(f"Cannot connect to Time Tagger '{serial}'")
            return
        self.tagger.reset()
        channels = list()
        channel_names = list()
        for channel, value, name in zip(self.settings["channels"].keys(), self.settings["channels"].values(), self.settings["channel_names"].values()):
            if value.get() == ChannelRoles.CHANNEL.value:
                channels.append(channel)
                channel_names.append(name.get())
                self.tagger.setEventDivider(channel, DIVIDER)
        self.tagger.setEventDivider(reference, DIVIDER)
        if clock:
            self.tagger.setEventDivider(clock, self.settings["clock_divider"].get())
        backend_rescaling = hasattr(self.tagger, "setRescaling")
        clock_period = 100000*self.settings["clock_divider"].get()
        if backend_rescaling:
            self.tagger.setRescaling(clock, clock_period)
        print(backend_rescaling)
        self.measurement = PpsTracking(self.tagger,
                                       channels=channels,
                                       reference=reference,
                                       clock=None if backend_rescaling else clock,
                                       clock_period=clock_period,
                                       channel_names=channel_names,
                                       debug_to_file=self.settings["store_debug_info"].get(),
                                       unscaled_to_file=self.settings["store_unscaled_data"].get(),
                                       reference_name=reference_name,
                                       folder=self.settings["storage_folder"].get())
        # self.filewriter = TimeTagger.FileWriter(tagger=self.tagger,
        #                                         filename="test5.ttbin",
        #                                         channels=list(range(1, 9)))
        self.measurement.setTimetagsMaximum(self.settings["max_live_tags"].get())
        self._adjust_storage_time()
        self.fig.clear()
        self.fig.subplots(len(channels), 1, sharex="all")
        self.fig.subplots_adjust(hspace=0, right=0.99, top=0.97, bottom=0.08, left=0.1)
        self.fig.canvas.draw_idle()
        updater = DisplayUpdater(self.measurement, self.fig, channels, self.add_message)
        updater.start()

    def _stop_measurement(self):
        try:
            self.measurement.stop()
            TimeTagger.freeTimeTagger(self.tagger)
            self.measurement = None
            self.tagger = None
        except AttributeError:
            pass

    def add_message(self, msg):
        self.messages.insert(0, msg)

    def _quit(self):
        if not messagebox.askyesno("Quit?", "Do you really want to close the program?"):
            return
        self._stop_measurement()
        self._save_settings()
        self.root.quit()
        self.root.destroy()

    def _save_settings(self):
        def to_string_dict(original: dict):
            new = dict()
            for key, value in original.items():
                if isinstance(value, tk.Variable):
                    new[key] = value.get()
                elif isinstance(value, dict):
                    new[key] = to_string_dict(value)
            return new
        with open("settings.dat", "wb") as storage:
            pickle.dump(to_string_dict(self.settings), storage)

    def _load_settings(self):
        def from_string_dict(settings: dict, dump: dict):
            for key in settings:
                if key in dump:
                    if isinstance(dump[key], dict):
                        from_string_dict(settings[key], dump[key])
                    else:
                        settings[key].set(dump[key])
        try:
            with open("settings.dat", "rb") as storage:
                from_string_dict(self.settings, pickle.load(storage))
        except:
            pass


class ModalWindow(tk.Toplevel):
    def __init__(self, parent: PPS_App, title: str):
        super().__init__()
        self.iconbitmap(this_folder() + "/iconTT.ico")
        self.parent = parent.root
        self.title(title)
        self.grab_set()
        self.focus()
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def cancel(self, event=None):
        self.parent.focus_set()
        self.destroy()


class SettingsWindow(ModalWindow):
    def __init__(self, parent: PPS_App, settings: Dict[str, StringVar]):
        super().__init__(parent=parent, title="Settings")
        ttk.Label(self, text="Serial number").grid(row=0, column=0, pady=10, sticky=tk.E)
        taggers = TimeTagger.scanTimeTagger()
        tt_select = ttk.OptionMenu(self, settings["serial"], taggers[0] if taggers else None, *taggers)
        tt_select.grid(row=0, column=1)

        ttk.Label(self, text="Role").grid(row=1, column=1)
        ttk.Label(self, text="Name").grid(row=1, column=2)
        for i in range(1, 9):
            ttk.Label(self, text=f"Input {i}").grid(row=i+1, column=0, sticky=tk.E)
            ttk.OptionMenu(self, settings["channels"][i], settings["channels"][i].get(), *[item.value for item in ChannelRoles]).grid(row=i+1, column=1)
            ttk.Entry(self, textvariable=settings["channel_names"][i]).grid(row=i+1, column=2)

        ttk.Label(self, text="Clock divider").grid(row=10, column=0, sticky=tk.E, pady=10)
        tk.Spinbox(self, textvariable=settings["clock_divider"], from_=1, to=10000, width=5, state="readonly").grid(row=10, column=1)

        # ttk.Label(self, text="Store debug information").grid(row=11, column=0, sticky=tk.E, pady=10)
        # tk.Checkbutton(self, variable=settings["store_debug_info"]).grid(row=11, column=1)


class StorageConfigWindow(ModalWindow):
    def __init__(self, parent: PPS_App, settings: dict):
        super().__init__(parent=parent, title="Storage configuration")
        tk.Grid.columnconfigure(self, 1, weight=1)
        browser = tk.Frame(self)
        browser.grid(row=0, column=1, sticky=tk.E+tk.W)
        self.filename = settings["storage_folder"]

        tk.Label(self, text="Data folder").grid(row=0, column=0, sticky=tk.E)
        tk.Entry(browser, textvariable=self.filename, state="disabled").pack(side=tk.LEFT, fill=tk.X, expand=1)
        tk.Button(browser, text="Browse", command=self.browse_folder).pack(side=tk.RIGHT)

        tk.Label(self, text="New file time").grid(row=1, column=0, sticky=tk.E)
        self.time_display = tk.Frame(self)
        self.time_display.grid(row=1, column=1, sticky=tk.W)
        self._time_digit(settings["storage_time"], "hour")
        tk.Label(self.time_display, text=":").pack(side=tk.LEFT)
        self._time_digit(settings["storage_time"], "minute")
        tk.Label(self.time_display, text=":").pack(side=tk.LEFT)
        self._time_digit(settings["storage_time"], "second")

        # Maximum number of live tags
        tk.Label(self, text="Max. tags in live view").grid(row=2, column=0, sticky=tk.E)
        tk.Spinbox(self,
                   textvariable=settings["max_live_tags"],
                   width=10,
                   from_=0,
                   to=9999999999).grid(row=2, column=1, sticky=tk.W)

        tk.Label(self, text="Store debug info").grid(row=3, column=0, sticky=tk.E)
        tk.Checkbutton(self, variable=settings["store_debug_info"]).grid(row=3, column=1, sticky=tk.W)

        tk.Label(self, text="Store unscaled tags").grid(row=4, column=0, sticky=tk.E)
        tk.Checkbutton(self, variable=settings["store_unscaled_data"]).grid(row=4, column=1, sticky=tk.W)

    def _time_digit(self, settings: dict, key: str):
        max_value = dict(hour=23, minute=59, second=59)
        tk.Spinbox(self.time_display,
                   textvariable=settings[key],
                   state="readonly",
                   width=2,
                   from_=0,
                   to=max_value[key],
                   format_="%02.0f").pack(side=tk.LEFT)

    def browse_folder(self):
        self.filename.set(tk.filedialog.askdirectory())

    def run(self):
        self.withdraw()


if __name__ == "__main__":
    window = PPS_App()
    window.run()
