import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from datetime import time
from os.path import dirname, realpath, exists
from time import sleep
import ctypes
from typing import Iterable, Optional, List, Callable, Type
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import TimeTagger
from .measurement import PpsTracking
from .utilities import AbstractTimeTaggerProxy, NetworkTimeTaggerProxy, USBTimeTaggerProxy
from .settings import Input, Settings, ChannelRoles


def this_folder():
    return dirname(realpath(__file__))


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


class PPS_GUI:
    """The graphical user interface for tracking PPS signals."""

    def __init__(self):
        app_id = 'com.swabianinstruments.pps.app'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

        self.root = tk.Tk()
        self.root.iconbitmap(this_folder() + "/iconTT.ico")
        panes = tk.PanedWindow(orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=1)
        self.messages = tk.Listbox(panes)
        panes.add(self.messages)
        self.fig_frame = tk.Frame(panes)
        panes.add(self.fig_frame)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.state("zoomed")

        self.__taggers: dict[str, AbstractTimeTaggerProxy] = dict()
        self.fig = Figure()
        self.measurement: Optional[PpsTracking] = None
        self.__current_tagger: Optional[AbstractTimeTaggerProxy] = None
        self.scan_time_taggers()
        self.settings = Settings(self.root)
        self.__last_connect_setting = None
        self.settings.connect.trace_add("write", lambda *args: self._connect_tagger(False))
        self.settings.resolution.trace_add("write", lambda *args: self._connect_tagger(True))
        self.settings.id_string.trace_add("write", lambda *args: self._connect_tagger(True))
        self.settings.load()
        for var in self.settings.storage_time.values():
            var.trace("w", self._adjust_storage_time)
        self.settings.storage_folder.trace("w", self._adjust_storage_folder)
        self.settings.max_live_tags.trace("w", self._adjust_max_live_tags)

    @property
    def current_tagger(self):
        return self.__current_tagger

    def scan_time_taggers(self):
        self.__taggers = dict()
        if self.__current_tagger:
            self.__taggers[self.__current_tagger.get_id()] = self.__current_tagger
        self.__scan_time_taggers(TimeTagger.scanTimeTagger, USBTimeTaggerProxy)
        self.__scan_time_taggers(TimeTagger.scanTimeTaggerServers, NetworkTimeTaggerProxy)

    def __scan_time_taggers(self, method: Callable[[], Iterable[str]], proxy_type: Type[AbstractTimeTaggerProxy]):
        for connection_string in method():
            proxy = proxy_type(connection_string)
            id_string = proxy.get_id()
            if id_string not in self.__taggers:
                self.__taggers[id_string] = proxy

    def get_tagger_ids(self):
        return list(self.__taggers.keys())

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
        file_menu.add_command(label="Hardware settings", command=lambda: SettingsWindow(self))
        file_menu.add_command(label="Storage settings", command=lambda: StorageConfigWindow(self))
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
            self.measurement.setNewFileTime(time(**{key: int(self.settings.storage_time[key].get()) for key in ["hour", "minute", "second"]}))

    def _adjust_storage_folder(self, *args):
        if self.measurement:
            self.measurement.setFolder(self.settings.storage_folder.get())

    def _adjust_max_live_tags(self, *args):
        if self.measurement:
            self.measurement.setTimetagsMaximum(self.settings.max_live_tags.get())

    def _connect_tagger(self, force_reconnect: bool = False):
        settings = dict(serial=self.settings.id_string.get(),
                        resolution=self.settings.resolution.get(),
                        connect=self.settings.connect.get())
        if self.__last_connect_setting != settings:
            self.__last_connect_setting = settings
            if force_reconnect and self.__current_tagger is not None:
                self.__current_tagger.disconnect()
                self.__current_tagger = None
            if settings["connect"] and settings["serial"] in self.__taggers:
                new_active = self.__taggers[settings["serial"]]
                if self.__current_tagger is not new_active:
                    self.__current_tagger = new_active
                    try:
                        self.__current_tagger.connect(resolution=self.settings.resolution.get())
                    except RuntimeError:
                        if self.settings.resolution.get() != "Standard":
                            self.add_message(f"Cannot connect in HighRes mode, reset to Standard")
                            self.settings.resolution.set("Standard")
                            return
                        serial = self.settings.id_string.get()
                        self.add_message(f"Cannot connect to Time Tagger '{serial}'")
                        self.__current_tagger = None
                        self.settings.connect.set(False)
            elif self.__current_tagger is not None:
                self.__current_tagger.disconnect()
                self.__current_tagger = None
            self._update_inputs()

    def _update_inputs(self):
        for phys_input in self.settings.channels.values():
            phys_input.on_tagger_change(self.__current_tagger)

    def _start_measurement(self):
        if not exists(self.settings.storage_folder.get()):
            self.add_message("Data folder does not exist")
            return
        clock = None
        reference_name = ""
        reference = 0
        channels = list()
        channel_names = list()
        for ch, phys_input in self.settings.channels.items():
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

        self.settings.connect.set(True)
        if self.__current_tagger is None:
            return
        tagger = self.__current_tagger.get_tagger()
        try:
            if clock:
                tagger.setEventDivider(clock, self.settings.clock_divider.get())
                tagger.setSoftwareClock(input_channel=clock,
                                        input_frequency=self.settings.clock_frequency.get()/self.settings.clock_divider.get())
            else:
                tagger.disableSoftwareClock()
        except RuntimeError:
            pass
        self.measurement = PpsTracking(tagger,
                                       channels=channels,
                                       reference=reference,
                                       channel_names=channel_names,
                                       debug_to_file=self.settings.store_debug_info.get(),
                                       reference_name=reference_name,
                                       folder=self.settings.storage_folder.get())
        self.measurement.setTimetagsMaximum(self.settings.max_live_tags.get())
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
        if self.__current_tagger:
            self.__current_tagger.disconnect()
        self._stop_measurement()
        self.settings.save()
        self.root.quit()
        self.root.destroy()


class ModalWindow(tk.Toplevel):
    def __init__(self, parent: PPS_GUI, title: str):
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
    def __init__(self, parent: PPS_GUI):
        # setters = parent.current_tagger.allows_setters()
        self.__parent = parent
        super().__init__(parent=parent, title="Settings")
        ttk.Label(self, text="Serial number").grid(row=0, column=0, pady=10, sticky=tk.E)
        taggers = parent.get_tagger_ids()
        self.__tt_select = ttk.OptionMenu(self, parent.settings.id_string)
        self.scan_time_taggers()
        self.__tt_select.grid(row=0, column=1)
        ttk.OptionMenu(self, parent.settings.resolution, None, *(mode.name for mode in TimeTagger.Resolution)).grid(row=0, column=2)
        tt_connect = ttk.Checkbutton(self, text="connect", variable=parent.settings.connect)
        tt_connect.grid(row=0, column=3, sticky=tk.E)
        tt_scan = ttk.Button(self, text="scan", command=self.scan_time_taggers)
        tt_scan.grid(row=0, column=4, sticky=tk.E)

        ttk.Label(self, text="Rising edge").grid(row=1, column=1)
        ttk.Label(self, text="Falling edge").grid(row=1, column=2)
        ttk.Label(self, text="Name").grid(row=1, column=3)
        for i, phys_input in parent.settings.channels.items():
            label_entry = ttk.Label(self, text=f"Input {i}")
            label_entry.grid(row=i+1, column=0, sticky=tk.E)
            phys_input.add_element(label_entry)
            phys_input.add_element(self.role(phys_input.rising_role, i, 1), Input.Edge.RISING)
            phys_input.add_element(self.role(phys_input.falling_role, i, 2), Input.Edge.FALLING)
            name_entry = ttk.Entry(self, textvariable=phys_input.name)
            name_entry.grid(row=i+1, column=3)
            phys_input.add_element(name_entry)
            phys_input.on_tagger_change(parent.current_tagger)
            resolution_label = ttk.Label(self, textvariable=phys_input.resolution)
            resolution_label.grid(row=i+1, column=4, sticky=tk.E)
            phys_input.add_element(resolution_label)

        row = 5+len(parent.settings.channels)
        padding = 10
        ttk.Label(self, text="Average events").grid(row=row, column=0, sticky=tk.E, pady=(padding, 0))
        tk.Spinbox(self, textvariable=parent.settings.signal_average, from_=1, to=100000000, width=12).grid(row=row, column=1, pady=(padding, 0))
        row += 1
        ttk.Label(self, text="Signal divider").grid(row=row, column=0, sticky=tk.E)
        tk.Spinbox(self, textvariable=parent.settings.signal_divider, from_=1, to=10000, width=12).grid(row=row, column=1)
        row += 1
        ttk.Label(self, text="Clock frequency").grid(row=row, column=0, sticky=tk.E, pady=(padding, 0))
        tk.Spinbox(self, textvariable=parent.settings.clock_frequency, from_=1E3, to=475E6, width=12).grid(row=row, column=1, pady=(padding, 0))
        row += 1
        ttk.Label(self, text="Clock divider").grid(row=row, column=0, sticky=tk.E)
        tk.Spinbox(self, textvariable=parent.settings.clock_divider, from_=1, to=10000, width=12).grid(row=row, column=1)

    def role(self, var, row, col) -> ttk.OptionMenu:
        menu = ttk.OptionMenu(self, var, var.get(), *[item.value for item in ChannelRoles])
        menu.grid(row=row+1, column=col)
        return menu

    def scan_time_taggers(self):
        self.__parent.scan_time_taggers()
        self.__tt_select["menu"].delete(0, "end")
        for id_string in self.__parent.get_tagger_ids():
            self.__tt_select["menu"].add_command(label=id_string, command=lambda: self.__parent.settings.id_string.set(id_string))


class StorageConfigWindow(ModalWindow):
    def __init__(self, parent: PPS_GUI):
        super().__init__(parent=parent, title="Storage configuration")
        tk.Grid.columnconfigure(self, 1, weight=1)
        browser = tk.Frame(self)
        browser.grid(row=0, column=1, sticky=tk.E+tk.W)
        self.filename = parent.settings.storage_folder

        tk.Label(self, text="Data folder").grid(row=0, column=0, sticky=tk.E)
        tk.Entry(browser, textvariable=self.filename, state="disabled").pack(side=tk.LEFT, fill=tk.X, expand=1)
        tk.Button(browser, text="Browse", command=self.browse_folder).pack(side=tk.RIGHT)

        tk.Label(self, text="New file time").grid(row=1, column=0, sticky=tk.E)
        self.time_display = tk.Frame(self)
        self.time_display.grid(row=1, column=1, sticky=tk.W)
        self._time_digit(parent.settings.storage_time, "hour")
        tk.Label(self.time_display, text=":").pack(side=tk.LEFT)
        self._time_digit(parent.settings.storage_time, "minute")
        tk.Label(self.time_display, text=":").pack(side=tk.LEFT)
        self._time_digit(parent.settings.storage_time, "second")

        # Maximum number of live tags
        tk.Label(self, text="Max. tags in live view").grid(row=2, column=0, sticky=tk.E)
        tk.Spinbox(self,
                   textvariable=parent.settings.max_live_tags,
                   width=10,
                   from_=0,
                   to=9999999999).grid(row=2, column=1, sticky=tk.W)

        tk.Label(self, text="Store debug info").grid(row=3, column=0, sticky=tk.E)
        tk.Checkbutton(self, variable=parent.settings.store_debug_info).grid(row=3, column=1, sticky=tk.W)

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
    window = PPS_GUI()
    window.run()
