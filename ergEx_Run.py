VERSION='v0.4.0a' # 0.4.0a - fMRI pilot
from codebase.experiment import passive_gui, passive_run, run_with_dict, run_slideshow
from codebase.experiment.active import active_gui, active_run
from codebase.experiment import run_questionnaire
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
import gc

ACTIVE_MODE = 3
""" Mode for the active phase 1 = DRCMR, 2 = LML, 3 = TrainTracks, 4 = Hidden wealth """

N_TRIALS_PASSIVE = 3 * 45 # Default 4 * 45
""" How often fractals are shown in the Passive Phase (defines trials as N_REPEATS_PASSIVE * N_FRACTALS"""
N_TRIALS_ACTIVE = 160 # Default 90
""" Number of trials in the active phaes"""
N_TRIALS_NOBRAINER = 15 # Default 15 (total number of permutations)
""" Number of nobrainer trials after the passive phase ends."""

TR = 1.61
""" TR of the MR scanner (also for simulations) """
SIMULATE_MR = 'None'
""" Mode of the MR: Simulate = simulates scanner, MRIDebug = shows a counter for received triggers,
fMRI = fMRI scanning mode, None = No TR logging / simulation
"""

MAX_RUN_PASSIVE = 3 # Defaults to 3
""" Number of runs of the passive phase"""
START_NOBRAINER = 45
"""Starts N_TRIALS_NOBRAINER after np.mod(passive, START_NOBRAINER) == 0 Trials """
MAX_RUN_ACTIVE = 1 # Defaults to 1
""" Number of runs in the active phase"""
MAX_TRIALS_PASSIVE = 45 # By default should be N_TRIALS_PASSIVE / 4
""" Number of trials per run in the passive phase. """
MAX_TRIALS_ACTIVE =  np.inf # Default is np.inf
""" Number of trials per run in he active phase. """
SESSIONS = [1, 2]

SLIDESET = {1: [30, 44, 45, 56], 2: [1, 16, 17, 28], 3: [30, 44, 45, 56], 4: [58, 72, 73, 81]}
""" The 2 sets of start and stop slides for the instructions depending on mode. """

GAMBLE_FILTER = False
""" Whether gambles are filtered in a given range. """
BREAKLENGTH = 10
""" Break duration between Sessions."""


def set_up_win(fscreen, gui=True):
    win = visual.Window(size=[3072 // 2, 1920 // 2], fullscr=fscreen,
                    screen=0, winType='pyglet', allowGUI=gui, monitor=None,
                    color=[-1,-1,-1], colorSpace='rgb', units='pix',
                    waitBlanking=False, useFBO=False)

    if not fscreen and gui:
        win_size = [1920, 1080] #[3072 // 2, 1920 // 2]
    else:
        win_size = [1920, 1080] #win.size

    print(win_size)
    refreshRate, frameDur = get_frame_timings(win)
    print(f"Frame duration = {frameDur}, refreshRate = {refreshRate}")

    Between = visual.TextStim(win=win, name='between',
                                text=f'Experiment set up, starting with passive task.',
                                pos=acfg.textPos, height=acfg.textHeight,
                                color='white')
    if not gui:
        win.mouseVisible = False

    return win, frameDur, Between, win_size


if __name__ == '__main__':

    thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisDir)
    filePath = os.path.join(thisDir, 'data', 'outputs') + os.sep

    expInfo = {'participant': '000', # Participant ID
               'test_mode': False, # Whether an agent automatically presses buttons
               'fullScreen': True, # Whether to use a full screen
               'calibration': False, # Whether to run calibrations before.
               'startPassive': 1, # Which run of the passive phase to start (starts at 1), if larger than MAX_RUN_PASSIVE, skips passive phase.
               'startActive': 1,
               'startSession': 1,
               'showQuestionnaires': False,
               'showInstructions': True} # Which run of the active phase to start from (starts at 1)

    if SIMULATE_MR in ['MRI', 'Simulate', 'MRIDebug']:
        expInfo.update({
            'responseLeft': '7',
            'responseButton': '8',
            'responseRight': '9'})

    expInfo = gui_update_dict(expInfo, f'Running Version: {VERSION}')

    expInfo['gambleFilter'] = GAMBLE_FILTER
    instruction_shown = not expInfo['showInstructions']

    SESSIONS = SESSIONS[expInfo['startSession'] - 1 : ]

    for nsess, sess in enumerate(SESSIONS):

        lambd, fractalList =  assign_fractals(expInfo['participant'], sess)
        lambd = float(lambd)

        expInfo.update({
            'n_resets_passive': 3, # MAX_RUN_PASSIVE,
            'n_trials_passive_before_reset': START_NOBRAINER, #MAX_TRIALS_PASSIVE,
            'n_trials_active': N_TRIALS_ACTIVE,
            'start_nobrainer': START_NOBRAINER,
            'mode': ACTIVE_MODE,
            'agentActive': expInfo['test_mode'],
            'TR': TR,
            'session': sess,
            'eta': lambd,
            'simulateMR': SIMULATE_MR})

        os.makedirs(filePath + make_bids_dir(expInfo['participant'], expInfo['session']),
                    exist_ok=True)

        run_with_dict(expInfo=expInfo)

        if expInfo['calibration']:
            win, frameDur, _, win_size = set_up_win(expInfo['fullScreen'], False)
            calib_conf = check_configs(expInfo.copy(), task='calibration')
            calibration_run(filePath, calib_conf, win=win)
            win.close()

        passive_conf = expInfo.copy()

        passive_conf.update({'run' : expInfo['startPassive'],
                            'agentMode': 'random',
                            'feedback': False,
                            'nTrial_noBrainer': N_TRIALS_NOBRAINER,
                            'maxTrial': MAX_TRIALS_PASSIVE})

        passive_conf = check_configs(passive_conf, task='passive')

        if not instruction_shown:
            win, frameDur, _, win_size = set_up_win(expInfo['fullScreen'], False)
            run_slideshow(win, passive_conf, win_size=win_size, start_slide=SLIDESET[ACTIVE_MODE][0], stop_slide=SLIDESET[ACTIVE_MODE][1])
            win.close()

        for run in range(passive_conf['run'],  MAX_RUN_PASSIVE + 1):

            win, frameDur, _, _ = set_up_win(expInfo['fullScreen'], False)
            passive_conf['run'] = run

            passive_conf = passive_gui(filePath, passive_conf, False)
            passive_conf['wealth'] = con.x_0
            event.clearEvents()
            wealh = passive_run(passive_conf, filePath, win, fractalList, frameDur)
            gc.collect()
            win.close()

        expInfo.update({'wealth' : con.x_0})

        # Reset run
        active_conf = expInfo.copy()

        active_conf.update(
            {'run': expInfo['startActive'],
            'maxTrial': MAX_TRIALS_ACTIVE,
            'agentMode': 'random'})

        active_conf = check_configs(active_conf, task='active')

        if not instruction_shown:
            win, frameDur, Between, _ = set_up_win(expInfo['fullScreen'], False)
            run_slideshow(win, passive_conf, win_size=win_size, start_slide=SLIDESET[ACTIVE_MODE][2], stop_slide=SLIDESET[ACTIVE_MODE][3])
            win.close()

        instruction_shown = True

        for run in range(active_conf['run'],  MAX_RUN_ACTIVE + 1):

            win, frameDur, _, _ = set_up_win(expInfo['fullScreen'], False)
            active_conf['run'] = run
            active_conf = active_gui(filePath, active_conf, False)
            event.clearEvents()
            wealh = active_run(active_conf, filePath, win, fractalList, frameDur)
            win.close()
            gc.collect()

        if nsess < len(SESSIONS) - 1:
            win, frameDur, Break, _ = set_up_win(expInfo['fullScreen'], False)

            breakText = f"Thank you for completing session {nsess + 1} of {len(SESSIONS)}.\nPlease take a break and contact the experimenter.\nBreak:\n"
            Break.setText(breakText)
            Break.draw()
            timer = core.CountdownTimer(BREAKLENGTH * 60)
            while timer.getTime() > 0: #until the timer is negative, after which time has elapsed
                time_left = timer.getTime()
                minutes = int(time_left / 60)
                seconds = int(time_left - minutes * 60)
                Break.setText(breakText + f'{minutes}:{seconds:02d}')
                Break.draw()
                win.flip()

            breakText = f"The break is over,\nplease press SPACE to continue."
            Break.setText(breakText)
            Break.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            win.close()
            gc.collect()


    if expInfo['showQuestionnaires']:

        folder = os.path.join(filePath, make_bids_dir(expInfo['participant'], sess))

        win, frameDur, _, _ = set_up_win(expInfo['fullScreen'], False)

        save_names = ['risk-propensity', 'dospert-risk-taking',
                      'dospert-perceived-risk', 'dospert-risk-benefits']

        quests = ['data/questionnaires/risk_propensity_scale.tsv',
                  'data/questionnaires/dospert_risk_taking.tsv',
                  'data/questionnaires/dospert_perceived_risk.tsv',
                  'data/questionnaires/dospert_risk_benefits.tsv']

        for nquest, quest in zip(save_names, quests):

            run_questionnaire(win, quest, expInfo['participant'],
                            passive_conf['responseLeft'],
                            passive_conf['responseRight'],
                            passive_conf['responseButton'],
                            os.path.join(folder, f'sub-{expInfo["participant"]}_{nquest}.tsv'))

        win.close()

        expInfo['showQuestionnaires'] = False

    win, frameDur, screenText, _ = set_up_win(expInfo['fullScreen'], False)
    screenText.setText("Thank you!\nYou completed the experiment.\nPlease contact the experimenters.")
    screenText.draw()
    win.flip()
    event.waitKeys(keyList=['q'])
    win.close()

    core.quit()