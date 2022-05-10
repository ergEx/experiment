""" Helper functions for the experiment. Doing many different things
like getting frame timings, waiting functions etc. """
from psychopy import core, gui, visual
from ...file_handler import extract_from_fname, make_filename
from ...utils import wealth_change
from .logger import ExperimentLogger
import os
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict
import time
import re
import json


def get_frame_timings(win:visual.Window) -> Tuple[float, float]:
    """Gets frame timings from the current window.

    Args:
        win (visual.Window): Window opbject.

    Returns:
        Tuple[float, float]: frame rate in Hz and frame duration in 1/Hz
    """
    # store frame rate of monitor if we can measure it from psychopy builder.
    frameRate = win.getActualFrameRate(nIdentical=20, nMaxFrames=200,
                                       nWarmUpFrames=15, threshold=1)
    if frameRate != None:
        frameDur = 1.0 / frameRate
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess

    return frameRate, frameDur


class WaitTime:
    def __init__(self, win:visual.Window, logger:ExperimentLogger = None,
                 frameDuration:float = 1/60):
        """Class to wait for a given time. Logs keypresses in between.

        Args:
            win (visual.Window): Psychopy window.
            logger (ExperimentLogger, optional): Logging class, to get MR pulses.
                Defaults to None.
            frameDuration (float, optional): Duration in 1/HZ. Defaults to 1/60.
        """

        self.logger = logger
        self.frameDuration = frameDuration
        self.win = win

    def wait(self, time:float, start:float = None):
        """Wait for a given time.

        Args:
            time (float): Time to wait.
            start (float, optional): Specify time or use logging clock. Defaults to None.
        """

        if start is None:
            start = self.logger.getTime()

        t_wait = start + time - self.frameDuration

        # Trying to avoid unecessary checks
        if self.logger is not None:
            while t_wait > self.logger.getTime():
                self.logger.keyStrokes(self.win)

        else:
            while t_wait > self.logger.getTime():
                pass

        self.win.flip()


def gui_update_dict(g_dict:Dict, gui_name:str = None) -> Dict:
    """Opens a gui to update dictionary values.

    Args:
        g_dict (Dict): Dictionary from which to spawn psychopy gui.
        gui_name (str, optional): Name of the gui. Defaults to None.

    Returns:
        Dict: Return updated dictionary.
    """

    dlg = gui.DlgFromDict(dictionary=g_dict, sortKeys=False,
                                    title=gui_name)

    if dlg.OK == False:
        core.quit()  # user pressed cancel

    return g_dict


def continue_from_previous(filename:str, wealth:float,
                           overwrite:bool) -> Tuple[float, int, str]:
    """Function to check if file is existent and wether to continue from
    previous run, spawns a new GUI.

    Args:
        filename (str): Filename to check.
        wealth (float): Starting wealth.
        overwrite (bool): If to overwrite or not.

    Returns:
        Tuple[float, int, str]: Return new wealth, previous trial,
            and mode either append or overwrite.
    """
    mode = 'w'
    trial = 0

    # Extract run no from filename:
    fpath, sub, eta, task, run = extract_from_fname(filename)

    if int(run) > 1:
        fname = make_filename(fpath, sub, eta, task, run - 1, add_dir=False)

        if os.path.exists(fname):
            try:
                data = pd.read_csv(fname, sep='\t')
                data2 = data.query('part != 1')
                if len(data2.query('event_type=="TrialEnd"').trial.values) > 1:
                    trial = data2.query('event_type=="TrialEnd"').trial.values[-1] + 1
                    wealth = data.query('event_type=="TrialEnd"').wealth.values[-1]

                    print(f"Reading settings for sub-{sub},  run-{run-1}")
                else:
                    raise ValueError(f'Check {fname}!')

            except:
                raise ValueError(f'Check {fname}!')

        else:
            raise ValueError(f'{fname} does not exist, start at previous run.')

    if os.path.exists(filename) and not overwrite:
        try:
            data = pd.read_csv(filename, sep='\t')

            if len(data.query('event_type=="TrialEnd"').trial.values) > 1:
                trial = data.query('event_type=="TrialEnd"').trial.values[-1] + 1
                wealth = data.query('event_type=="TrialEnd"').wealth.values[-1]
                print('Continuing for Filename.')
                mode = 'a'

        except:
            pass

    return wealth, trial, mode


def load_calibration(path:str, participant:str, session:str) -> Tuple[float, float]:
    """Load calibration file for given participant.

    Args:
        path (str): Filepath to calibration folder.
        participant (str): Participant ID
        session (str): Session ID.

    Returns:
        Tuple[float, float]: Tuple of x and y coordinates. Offset in calibration units..
    """
    filename = make_filename(path, participant, session, 'calibration', 1)

    if os.path.exists(filename):
        coordinates = pd.read_csv(filename, sep='\t')
        x_offset = coordinates.x_offset.values
        y_offset = coordinates.y_offset.values

        if x_offset and y_offset:
            return (x_offset[0].item(), y_offset[0].item())

    # Returns 0 if no calibration exists.
    return (0, 0)


def calculate_number_of_images(variable_timings:pd.DataFrame,
                               fixed_timings:List[float], TR:float,
                               wriggle_room:float = 0.1) -> int:
    """Calculate number of TR images the experiment will need.

    Args:
        variable_timings (pd.DataFrame): pd.Dataframe with timings in s
        fixed_timings (List[float]): List of fixed timings
            (occuring every trial) in s.
        TR (float): Time of repetition in s.
        wriggle_room (float, optional): Wriggle room for timing in s.
            Defaults to 0.2.

    Returns:
        int: Number of images.
    """

    time = 0

    for _, trial in variable_timings.iterrows():
        time += trial.sum() + np.sum(fixed_timings) + wriggle_room

    no_trs = np.ceil(time / TR).astype(int)

    print(f"Total time {time/60:4.2f} min with {no_trs} images. "
          + f"Avg. duration per trial ca. {time/ len(variable_timings)} s")

    return no_trs


class DebugTimer:

    def __init__(self, active=False):
        self.t0 = 0
        self.name = 'none'
        self.expected = None
        if active:
            self.set = self._seta
            self.end = self._enda
        else:
            self.set = self._setb
            self.end = self._endb

    def _seta(self, name='none', expected=None):
        self.t0 = time.time()
        self.name = name
        self.expected = expected

    def _enda(self):
        time_gone = time.time() - self.t0
        if self.expected is None:
            print(f'Timer {self.name}: {time_gone * 1000:4.2f} ms')
        else:
            print(f'Timer {self.name}: {time_gone * 1000:4.2f} ms || '
                 +f'expected: {self.expected * 1000 :4.2f} ms || '
                 +f'error {(time_gone - self.expected) * 1000 :4.2f} ms')

    def _setb(self, name='none', expected=None):
        pass

    def _endb(self):
        pass


def assign_fractals(participant_id:str, eta:Union[str, float, int],
                    n_fractals:int=9, path='data/stimuli/') -> Dict:
    """Function to create fractals for a given participant ID, assigns a unique
    set of fractals to each participant, which can be looked up for the given
    dynamic.

    Args:
        participant_id (str): [description]
        eta (List[float], optional): [description]. Defaults to [-1.0, 0.0, 0.5, 1.0].
        n_fractals (int, optional): [description]. Defaults to 9.

    Returns:
        List[str]: Fractal list.
    """

    with open(f"{path}fractal_set_nf{n_fractals}.json", "r") as infile:
        fractal_dict = json.load(infile)

    id_entry = re.search('\d{1,3}$|[a-z]{1}$', participant_id)
    id_entry = id_entry.group(0)

    # Make eta a string, if numeric
    if isinstance(eta, (int, float)):
        eta = f'{eta:2.1f}'

    # Check if eta is valid:
    if eta not in ['-1.0', '-0.5', '0.0', '0.5', '1.0']:
        raise ValueError("Eta has to be in '-1.0', '-0.5', '0.0', '0.5', '1.0'")

    return fractal_dict[id_entry][eta]


def format_wealth(wealth:float, fstring:str = "012,.0f") -> str:
    """Helper function to format wealth.

    Args:
        wealth (float): Current wealth
        fstring (str, optional): Formatting definition. Defaults to "07,.0f".

    Returns:
        string: Wealth formatted according to fstring
    """
    return format(wealth, fstring)