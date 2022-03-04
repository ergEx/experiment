from codebase.experiment import passive_gui, passive_run, run_with_dict
from codebase.experiment.active import active_gui, active_run
from codebase.experiment import calibration_run
from codebase import constants as con
from codebase.experiment import gui_update_dict, assign_fractals
from codebase.experiment.configs import active_configs as acfg
import os
from psychopy import visual, core, event
import numpy as np
from codebase.file_handler import make_bids_dir

if __name__ == '__main__':

    thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisDir)
    filePath = os.path.join(thisDir, 'data', 'outputs') + os.sep

    expInfo = {'participant': '0',
               'eta': 0.0,
               'n_repeats_passive': 6,
               'n_trials_active': 45,
               'test_mode': True,
               'fullScreen': False,
               'calibration': False,
               'passive_mode': 1,
               'active_mode': 2,
               'is_simulation' : False,
               'TR': 2.0}

    expInfo = gui_update_dict(expInfo, 'SetUp')

    expInfo.update({'responseUp': 'up',
                    'responseDown': 'down',
                    'responseLeft': 'left',
                    'responseRight': 'right',
                    'responseSave': 'space',
                    'responseButton': 'space'})

    fractalList = assign_fractals(expInfo['participant'], expInfo['eta'])

    os.makedirs(filePath + make_bids_dir(expInfo['participant'], expInfo['eta']), exist_ok=True)

    run_with_dict(expInfo=expInfo)

    win = visual.Window(size=[3072 / 2, 1920 / 2], fullscr=expInfo['fullScreen'],
                        screen=0, winType='pyglet', allowGUI=True, monitor=None,
                        color=[-1,-1,-1], colorSpace='rgb', units='pix')

    Between = visual.TextStim(win=win, name='between',
                                text=f'Experiment set up, starting with passive task.',
                                pos=acfg.textPos, height=acfg.textHeight,
                                color='white')
    Between.draw()
    win.flip()
    core.wait(2)
    win.flip()

    if expInfo['calibration']:
        calibration_run(filePath, expInfo, win=win)

    expInfo.update({'agentActive': expInfo['test_mode'],
                    'wealth': con.X0,
                    'simulateMR': 'Simulate',
                    'overwrite': True,
                    'maxDuration': 60,
                    'maxTrial': 27,
                    'maxRun': 1,
                    'run' : 1,
                    'agentMode': 'random',
                    'feedback': False,
                    'nTrial_noBrainer': 10})

    for run in range(expInfo['run'],  expInfo['maxRun'] + 1):

        expInfo['run'] = run

        expInfo = passive_gui(filePath, expInfo, False)
        event.clearEvents()

        wealh = passive_run(expInfo, filePath, win, fractalList)

    expInfo.update({'wealth' : con.X0})

    # Reset run
    expInfo['run'] = 1
    expInfo['maxRun'] = 1
    expInfo['maxTrial'] = np.inf

    expInfo = active_gui(filePath, expInfo, spawnGui=False)

    Between.setText("Beginning active task soon.")
    Between.draw()
    win.flip()
    core.wait(2)
    win.flip()

    for run in range(expInfo['run'],  expInfo['maxRun'] + 1):

        expInfo['run'] = run
        expInfo = active_gui(filePath, expInfo, False)
        event.clearEvents()
        wealh = active_run(expInfo, filePath, win, fractalList)

    Between.setText("You are done, thank you.")
    Between.draw()
    win.flip()
    core.wait(2)
    win.flip()

    core.quit()