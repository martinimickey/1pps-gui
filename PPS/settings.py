from __future__ import annotations
from typing import Optional
from enum import Enum
import pickle
import tkinter as tk
from tkinter import BooleanVar, IntVar, StringVar, Variable, ttk, messagebox, Widget, DoubleVar
from .utilities import AbstractTimeTaggerProxy
import TimeTagger


class ChannelRoles(Enum):
    UNUSED = "Unused"
    REFERENCE = "Use as Reference"
    CHANNEL = "Signal channel"
    CLOCK = "External clock"


class Input:
    class Edge:
        RISING = 1
        FALLING = 2
        BOTH = 3

    def __init__(self, root: tk.Tk, number: int):
        self.number = number
        self.rising_role = StringVar(root, ChannelRoles.UNUSED.value)
        self.falling_role = StringVar(root, ChannelRoles.UNUSED.value)
        self.name = StringVar(root, "")
        self.resolution = StringVar(root, "")
        self.enabled = 0
        self.__elements = dict()

    def add_element(self, element: Widget, edges: Edge = Edge.BOTH):
        self.__elements[element] = edges

    def on_tagger_change(self, tagger: Optional[AbstractTimeTaggerProxy]):
        self.resolution.set(tagger.get_resolution(self.number) if tagger else "")
        self.enabled = ((int(tagger.is_channel_allowed(-self.number)) << 1) + int(tagger.is_channel_allowed(self.number))) if tagger else 0
        for element in list(self.__elements.keys()):
            try:
                element.config(state="normal" if self.enabled & self.__elements[element] else "disabled")
            except:
                del self.__elements[element]


class Settings:
    def __init__(self, root, id_string=""):
        self.id_string = tk.StringVar(root, id_string)
        self.channels = {ch: Input(root, ch) for ch in range(1, 19)}
        self.resolution = tk.StringVar(root, next(iter(TimeTagger.Resolution)).name)
        self.connect = tk.BooleanVar(root, False)
        self.storage_folder = StringVar(root, "")
        self.store_debug_info = BooleanVar(root, False)
        self.storage_time = {key: StringVar(root, "00", key) for key in ("hour", "minute", "second")}
        self.max_live_tags = IntVar(root, 300)
        self.clock_divider = IntVar(root, 1)
        self.clock_frequency = DoubleVar(root, 1E7)
        self.signal_divider = IntVar(root, 1)
        self.signal_average = IntVar(root, 1)

    def save(self):
        with open("settings.dat", "wb") as storage:
            pickle.dump(Settings.to_string_dict(self.__dict__), storage)

    def load(self):
        try:
            with open("settings.dat", "rb") as storage:
                Settings.from_string_dict(self.__dict__, pickle.load(storage))
        except:
            pass

    @ staticmethod
    def to_string_dict(original: dict):
        new = dict()
        for key, value in original.items():
            if isinstance(value, tk.Variable):
                new[key] = value.get()
            elif isinstance(value, dict):
                new[key] = Settings.to_string_dict(value)
            elif hasattr(value, "__dict__"):
                new[key] = Settings.to_string_dict(value.__dict__)
        return new

    @ staticmethod
    def from_string_dict(settings: dict, dump: dict):
        for key in settings:
            if key in dump:
                if isinstance(dump[key], dict):
                    if isinstance(settings[key], dict):
                        Settings.from_string_dict(settings[key], dump[key])
                    elif hasattr(settings[key], "__dict__"):
                        Settings.from_string_dict(settings[key].__dict__, dump[key])
                else:
                    settings[key].set(dump[key])
