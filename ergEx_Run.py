from codebase.experiment import passive_gui, passive_run, run_with_dict
from codebase.experiment.active import active_gui, active_run
from codebase.experiment import calibration_run
from codebase import constants as con
from codebase.experiment import gui_update_dict, assign_fractals
from codebase.experiment.configs import active_configs as acfg
from codebase.experiment.configs import check_configs
import os
from psychopy import visual, core, event
import numpy as np
from codebase.experiment.exp.helper import get_frame_timings
from codebase.file_handler import make_bids_dir

N_REPEATS_PASSIVE = 6
""" How often fractals are shown in the Passive Phase (defines trials as N_REPEATS_PASSIVE * N_FRACTALS"""
N_TRIALS_ACTIVE = 90
""" Number of trials in the active phaes"""
N_TRIALS_NOBRAINER = 10
""" Number of nobrainer trials after the passive phase ends."""

PASSIVE_MODE = 1
""" Mode for the passive phase 1 = trajectory, 2 = resets. """
ACTIVE_MODE = 2
""" Mode for the active phase 1 = DRCMR, 2 = LML """
TR = 2.0
""" TR of the MR scanner (also for simulations) """
SIMULATE_MR = 'Simulate'
""" Mode of the MR: Simulate = simulates scanner, MRIDebug = shows a counter for received triggers,
fMRI = fMRI scanning mode, None = No TR logging / simulation
"""

MAX_RUN_PASSIVE = 2
""" Number of runs of the passive phase"""
MAX_RUN_ACTIVE = 1
""" Number of runs in the active phase"""
MAX_TRIALS_PASSIVE = 27
""" Number of trials per run in the passive phase. """
MAX_TRIALS_ACTIVE = 54 # np.inf
""" Number of trials per run in he active phase. """


if __name__ == '__main__':

    thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisDir)
    filePath = os.path.join(thisDir, 'data', 'outputs') + os.sep

    expInfo = {'participant': '0', # Participant ID
               'eta': 1.0, # Dynamic of the experiment
               'test_mode': True, # Whether an agent automatically presses buttons
               'fullScreen': False, # Whether to use a full screen
               'calibration': False, # Whether to run calibrations before.
               'startPassive': 1, # Which run of the passive phase to start (starts at 1), if larger than MAX_RUN_PASSIVE, skips passive phase.
               'startActive': 1} # Which run of the active phase to start from (starts at 1)

    expInfo = gui_update_dict(expInfo, 'Set Up')

    expInfo.update({
        'n_repeats_passive': N_REPEATS_PASSIVE,
        'n_trials_active': N_TRIALS_ACTIVE,
        'passive_mode': PASSIVE_MODE,
        'active_mode': ACTIVE_MODE,
        'agentActive': expInfo['test_mode'],
        'TR': TR})

    fractalList = assign_fractals(expInfo['participant'], expInfo['eta'])

    os.makedirs(filePath + make_bids_dir(expInfo['participant'], expInfo['eta']),
                exist_ok=True)

    run_with_dict(expInfo=expInfo)

    win = visual.Window(size=[3072 / 2, 1920 / 2], fullscr=expInfo['fullScreen'],
                        screen=0, winType='pyglet', allowGUI=True, monitor=None,
                        color=[-1,-1,-1], colorSpace='rgb', units='pix',
                        waitBlanking=False)

    refreshRate, frameDur = get_frame_timings(win)
    print(f"Frame duration = {frameDur}, refreshRate = {refreshRate}")
    frameDur = None

    Between = visual.TextStim(win=win, name='between',
                                text=f'Experiment set up, starting with passive task.',
                                pos=acfg.textPos, height=acfg.textHeight,
                                color='white')
    Between.draw()
    win.flip()
    core.wait(2)
    win.flip()

    if CALIBRATION:
        calib_conf = check_configs(expInfo.copy(), task='calibration')
        calibration_run(filePath, calib_conf, win=win)

    passive_conf = expInfo.copy()

    passive_conf.update({'simulateMR': SIMULATE_MR,
                          'run' : expInfo['startPassive'],
                          'agentMode': 'random',
                          'feedback': False,
                          'nTrial_noBrainer': N_TRIALS_NOBRAINER,
                          'maxTrial': MAX_TRIALS_PASSIVE})

    passive_conf = check_configs(passive_conf, task='passive')


    for run in range(passive_conf['run'],  MAX_RUN_PASSIVE + 1):

        passive_conf['run'] = run

        passive_conf = passive_gui(filePath, passive_conf, False)
        event.clearEvents()
        wealh = passive_run(passive_conf, filePath, win, fractalList, frameDur)

    expInfo.update({'wealth' : con.X0})

    # Reset run
    active_conf = expInfo.copy()

    active_conf.update(
        {'run': expInfo['startActive'],
        'maxTrial': MAX_TRIALS_ACTIVE,
        'agentMode': 'eta_0.0'})

    active_conf = check_configs(active_conf, task='active')

    Between.setText("Beginning active task soon.")
    Between.draw()
    win.flip()
    core.wait(2)
    win.flip()

    for run in range(active_conf['run'],  MAX_RUN_ACTIVE + 1):

        active_conf['run'] = run
        active_conf = active_gui(filePath, active_conf, False)
        event.clearEvents()
        wealh = active_run(active_conf, filePath, win, fractalList, frameDur)

    Between.setText("You are done, thank you.")
    Between.draw()
    win.flip()
    core.wait(2)
    win.flip()

    core.quit()