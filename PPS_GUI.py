import tkinter as tk
from tkinter import BooleanVar, IntVar, StringVar, Variable, ttk, messagebox, Widget
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
    DIVIDER = 10
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


class Input:
    def __init__(self, root: tk.Tk, number: int):
        self.number = number
        self.rising_role = StringVar(root, ChannelRoles.UNUSED.value)
        self.falling_role = StringVar(root, ChannelRoles.UNUSED.value)
        self.name = StringVar(root, "")
        self.resolution = StringVar(root, "")
        self.enabled = False
        self.__elements = set()

    def add_element(self, element: Widget):
        self.__elements.add(element)

    def on_tagger_change(self, tagger: Optional[TimeTagger.TimeTagger]):
        enabled = False
        if tagger is not None:
            if self.number in tagger.getChannelList(type=TimeTagger.ChannelEdge.StandardRising):
                self.resolution.set("Standard")
            elif self.number in tagger.getChannelList(type=TimeTagger.ChannelEdge.HighResRising):
                self.resolution.set("HighRes")
            else:
                self.resolution.set("")
            enabled = bool(self.resolution.get())
        self.enabled = enabled
        remove = set()
        for element in self.__elements:
            try:
                element.config(state="normal" if enabled else "disabled")
            except:
                remove.add(element)
        self.__elements -= remove


class PPS_App:
    """The graphical user interface for tracking PPS signals."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.iconbitmap(this_folder() + "/iconTT.ico")
        self.fig = Figure()
        self.measurement: Optional[PpsTracking] = None
        self.tagger: Optional[TimeTagger.TimeTagger] = None
        taggers = TimeTagger.scanTimeTagger()
        self.settings: SettingsType = dict(serial=tk.StringVar(self.root, taggers[0] if taggers else ""),
                                           channels={ch: Input(self.root, ch) for ch in range(1, 19)},
                                           resolution=tk.StringVar(self.root, next(mode for mode in TimeTagger.Resolution).name),
                                           connect=tk.BooleanVar(self.root, False),
                                           #    channel_names={ch: tk.StringVar(self.root, "") for ch in range(1, 9)},
                                           storage_folder=StringVar(self.root, ""),
                                           store_debug_info=BooleanVar(self.root, False),
                                           store_unscaled_data=BooleanVar(self.root, False),
                                           storage_time={key: StringVar(self.root, "00", key) for key in ("hour", "minute", "second")},
                                           max_live_tags=IntVar(self.root, 300),
                                           clock_divider=IntVar(self.root, 1))
        self.__last_connect_setting = None
        self.settings["connect"].trace_add("write", lambda *args: self._connect_tagger(False))
        self.settings["resolution"].trace_add("write", lambda *args: self._connect_tagger(True))
        self.settings["serial"].trace_add("write", lambda *args: self._connect_tagger(True))
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

    def _connect_tagger(self, force_reconnect: bool = False):
        settings = dict(serial=self.settings["serial"].get(),
                        resolution=self.settings["resolution"].get(),
                        connect=self.settings["connect"].get())
        if self.__last_connect_setting != settings:
            self.__last_connect_setting = settings
            if force_reconnect and self.tagger is not None:
                TimeTagger.freeTimeTagger(self.tagger)
                self.tagger = None
            if settings["connect"]:
                if self.tagger is None or self.tagger.getSerial() != settings["serial"]:
                    try:
                        self.tagger = TimeTagger.createTimeTagger(settings["serial"], TimeTagger.Resolution[self.settings["resolution"].get()])
                    except RuntimeError:
                        if self.settings["resolution"].get() != "Standard":
                            self.add_message(f"Cannot connect in HighRes mode, reset to Standard")
                            self.settings["resolution"].set("Standard")
                            return
                        serial = self.settings["serial"].get()
                        self.add_message(f"Cannot connect to Time Tagger '{serial}'")
                        self.tagger = None
                        self.settings["connect"].set(False)
            elif self.tagger is not None:
                TimeTagger.freeTimeTagger(self.tagger)
                self.tagger = None
            self._update_inputs()

    def _update_inputs(self):
        for phys_input in self.settings["channels"].values():
            phys_input.on_tagger_change(self.tagger)

    def _start_measurement(self):
        if not exists(self.settings["storage_folder"].get()):
            self.add_message("Data folder does not exist")
            return
        clock = None
        reference_name = ""
        reference = 0
        channels = list()
        channel_names = list()
        for ch, phys_input in self.settings["channels"].items():
            if phys_input.enabled:
                for attribute_name, factor in (("rising_role", 1), ("falling_role", -1)):
                    role = getattr(phys_input, attribute_name).get()
                    if role == ChannelRoles.REFERENCE.value:
                        reference = factor * ch
                        reference_name = phys_input.name.get()
                    elif role == ChannelRoles.CLOCK.value:
                        clock = factor * ch
                    elif role == ChannelRoles.CHANNEL.value:
                        channel_names.append(phys_input.name.get())
                        channels.append(factor * ch)
        if not reference:
            self.add_message("No reference given")
            return

        self.settings["connect"].set(True)
        if self.tagger is None:
            return
        self.tagger.reset()
        for channel in channels + [reference]:
            self.tagger.setEventDivider(channel, DIVIDER)
        backend_rescaling = False
        clock_period = 100000
        if clock:
            backend_rescaling = hasattr(self.tagger, "setSoftwareClock")
            self.tagger.setEventDivider(clock, self.settings["clock_divider"].get())
            clock_period = 100000*self.settings["clock_divider"].get()
            if backend_rescaling:
                self.tagger.setSoftwareClock(clock, clock_period)
            else:
                self.tagger.disableSoftwareClock()
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
            self.measurement = None
        except AttributeError:
            pass

    def add_message(self, msg):
        self.messages.insert(0, msg)

    def _quit(self):
        if not messagebox.askyesno("Quit?", "Do you really want to close the program?"):
            return
        if self.tagger:
            TimeTagger.freeTimeTagger(self.tagger)
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
                elif hasattr(value, "__dict__"):
                    new[key] = to_string_dict(value.__dict__)
            return new
        with open("settings.dat", "wb") as storage:
            pickle.dump(to_string_dict(self.settings), storage)

    def _load_settings(self):
        def from_string_dict(settings: dict, dump: dict):
            for key in settings:
                if key in dump:
                    if isinstance(dump[key], dict):
                        if isinstance(settings[key], dict):
                            from_string_dict(settings[key], dump[key])
                        elif hasattr(settings[key], "__dict__"):
                            from_string_dict(settings[key].__dict__, dump[key])
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
        ttk.OptionMenu(self, settings["resolution"], None, *(mode.name for mode in TimeTagger.Resolution)).grid(row=0, column=2)
        tt_connect = ttk.Checkbutton(self, text="connect", variable=settings["connect"])
        tt_connect.grid(row=0, column=3, sticky=tk.E)

        ttk.Label(self, text="Rising edge").grid(row=1, column=1)
        ttk.Label(self, text="Falling edge").grid(row=1, column=2)
        ttk.Label(self, text="Name").grid(row=1, column=3)
        for i, phys_input in settings["channels"].items():
            label_entry = ttk.Label(self, text=f"Input {i}")
            label_entry.grid(row=i+1, column=0, sticky=tk.E)
            phys_input.add_element(label_entry)
            phys_input.add_element(self.role(phys_input.rising_role, i, 1))
            phys_input.add_element(self.role(phys_input.falling_role, i, 2))
            name_entry = ttk.Entry(self, textvariable=phys_input.name)
            name_entry.grid(row=i+1, column=3)
            phys_input.add_element(name_entry)
            phys_input.on_tagger_change(parent.tagger)
            resolution_label = ttk.Label(self, textvariable=phys_input.resolution)
            resolution_label.grid(row=i+1, column=4, sticky=tk.E)
            phys_input.add_element(resolution_label)

        divider_row = 5+len(settings["channels"])
        ttk.Label(self, text="Clock divider").grid(row=divider_row, column=0, sticky=tk.E, pady=10)
        tk.Spinbox(self, textvariable=settings["clock_divider"], from_=1, to=10000, width=5, state="readonly").grid(row=divider_row, column=1)

    def role(self, var, row, col) -> ttk.OptionMenu:
        menu = ttk.OptionMenu(self, var, var.get(), *[item.value for item in ChannelRoles])
        menu.grid(row=row+1, column=col)
        return menu


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
