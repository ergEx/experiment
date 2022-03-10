"""Active part of the experiment. Contains two parts:
- active_gui: does a bit of input checking and allows to load data of previous
runs or to continue from previous state of experiment.
- active_run: starts the experiment.
"""
from psychopy import visual, core
from psychopy.hardware.emulator import SyncGenerator
import numpy as np
import pandas as pd
import os
import gc
from typing import Optional, Type, List
from .. import constants as con
from .exp import ExperimentLogger, ActiveAutoPilot, WaitTime, wealth_calculator
from .exp import active_report, get_frame_timings, DebugLogger
from ..utils import wealth_change
from .exp import continue_from_previous, load_calibration, calculate_number_of_images
from .configs import active_configs as acfg
from typing import Optional, Dict
from .exp.helper import format_wealth, gui_update_dict, DebugTimer, make_filename


def active_gui(filePath:str, expInfo:Optional[Dict] = None, spawnGui:bool=True):
    if expInfo is None:
        expInfo = {'participant': '0',
                'run': 1,
                'eta': 0.0,
                'agentMode': ['time_optimal', 'random', 'eta_-1.0', 'eta_1.0',
                                'eta_0.0', 'soft_-1.0', 'soft_0.0', 'soft_1.0'],
                'agentActive': True,
                'simulateMR': ['None' , 'MRIDebug', 'Simulate', 'MRI'],
                'responseLeft': 'left',
                'responseRight': 'right',
                'fullScreen': False,
                'overwrite': True,
                'wealth' : con.X0,
                'TR': 0.592,
                'maxDuration': 60,
                'maxTrial': np.inf,
                'maxRun': 4}

    if spawnGui:
        expInfo = gui_update_dict(expInfo, 'ergEx_Active')

    offset = load_calibration(filePath, expInfo['participant'], expInfo['eta'])

    # Gui again, to check name and previous data.
    if spawnGui:
        expInfo = gui_update_dict(expInfo, 'ergEx_Active')

    # Buttons to str:
    expInfo['responseLeft'] = str(expInfo['responseLeft'])
    expInfo['responseRight'] = str(expInfo['responseRight'])

    responseKeyList = [expInfo['responseLeft'], expInfo['responseRight']]
    responseMapping = {expInfo['responseLeft'] : 'left',
                    expInfo['responseRight']: 'right'}

    fileName = make_filename(filePath, expInfo['participant'], expInfo['eta'],
                             'active', expInfo['run'])

    wealth, nTrial, writeMode = continue_from_previous(fileName, expInfo['wealth'],
                                                    expInfo['overwrite'])

    # Load trial file:
    trialInfoPath = make_filename('data/inputs/', expInfo['participant'], expInfo['eta'],
                                  'active', extension='input.tsv')


    trialFile = pd.read_csv(trialInfoPath, sep='\t')

    noTR = calculate_number_of_images(trialFile[['iti', 'onset_gamble_pair_left']],
                                    fixed_timings=[acfg.timeResponse,
                                                    acfg.timeSideHighlight, acfg.timeCoinToss,
                                                    acfg.timeFractalSelection,
                                                    acfg.timeWealthUpdate,
                                                    acfg.timeFinalDisplay], TR=expInfo['TR'],
                                                    wriggle_room=0)

    noTR += acfg.waitTR

    expInfo['noTR'] = noTR
    expInfo['wealth'] = wealth
    expInfo['nTrial'] = nTrial
    expInfo['writeMode'] = writeMode
    expInfo['offset'] = offset
    expInfo['responseKeyList'] = responseKeyList
    expInfo['responseMapping'] = responseMapping

    return expInfo


def active_run(expInfo:Dict, filePath:str, win:visual.Window = None,
               fractalList:List[str] = None):

    if win is None:
        win = visual.Window(size=[3072 / 2, 1920 / 2], fullscr=expInfo['fullScreen'],
                            screen=0, winType='pyglet', allowGUI=True, monitor=None,
                            color=[-1,-1,-1], colorSpace='rgb', units='pix')

    wealth = expInfo['wealth']
    nTrial = expInfo['nTrial']
    offset = expInfo['offset']
    noTR = expInfo['noTR']
    responseKeyList = expInfo['responseKeyList']
    responseMapping = expInfo['responseMapping']

    # Rebuild paths
    trialInfoPath = make_filename('data/inputs/', expInfo['participant'], expInfo['eta'],
                                  'active', extension='input.tsv')

    fileName = make_filename(filePath, expInfo['participant'], expInfo['eta'],
                             'active', expInfo['run'])

    # Currently testing if the supposed ones are better.
    frameRate, frameDur = get_frame_timings(win) #- hard coded

    if fractalList is None:
        fractalList = acfg.fractalList
        print("Using default fractals")
    else:
        if len(fractalList) != 9:
            raise ValueError("Fractal List needs to be of length 9")
        if np.array(fractalList).dtype.type is not np.str_:
            raise ValueError("Fractal List needs to contain string!")

    # Create logger
    if expInfo['simulateMR'] == 'MRIDebug':
        Counter = visual.TextStim(win=win, name='Counter',
                                text='0',
                                pos=(-600, -300), height=50, color='white')
        Counter.pos += offset
        Counter.setAutoDraw(True)
        Logger = DebugLogger(fileName=fileName, globalClock=core.Clock(),
                                wealth=wealth, participant_id=expInfo['participant'],
                                eta=expInfo['eta'], mode='active', seq_tr=expInfo['TR'], run=expInfo['run'],
                                text=Counter)
        Logger.create(expInfo['writeMode'])

    else:
        Logger = ExperimentLogger(fileName=fileName, globalClock=core.Clock(),
                                wealth=wealth, participant_id=expInfo['participant'],
                                eta=expInfo['eta'], mode='active', seq_tr=expInfo['TR'], run=expInfo['run'])
        Logger.create(expInfo['writeMode'])

    # Autopilot
    Agent = ActiveAutoPilot(0.4, 0.1, active=expInfo['agentActive'],
                            mode=expInfo['agentMode'],
                            buttonLeft=expInfo['responseLeft'],
                            buttonRight=expInfo['responseRight'])
    # Waiting Tool
    Wait = WaitTime(win, Logger, frameDuration=frameDur)

    if expInfo['simulateMR'] in ['Simulate']:
        SyncGen = SyncGenerator(TR=expInfo['TR'], TA=expInfo['TR'] / 10, volumes = noTR)
    ###################### Preloading Images #######################################
    Initialization = visual.TextStim(win=win, name='initialization',
                                    text='Initializing!!', pos=acfg.textPos,
                                    height=acfg.textHeight, ori=0.0, color='white')
    Initialization.setAutoDraw(True)
    win.flip()

    fractals = {location: {} for location in acfg.imgLocation}
    coins = {location: {} for location in acfg.imgLocation}

    for imL in acfg.imgLocation:
        for nFl, fl in enumerate(fractalList):

            if "right" in imL and expInfo['active_mode'] == 2:
                pos = list(acfg.imgLocPos[imL])
                pos[1] = 0
            else:
                pos = acfg.imgLocPos[imL]

            fractals[imL][nFl] = visual.ImageStim(win=win, pos=pos,
                                                size=acfg.imgSize, opacity=0,
                                                image=acfg.imagePath + os.sep + 'fractals' + os.sep + fl + '.png')
            fractals[imL][nFl].pos += offset
            fractals[imL][nFl].setAutoDraw(True)

        coins[imL] = visual.ImageStim(win=win, pos=acfg.imgLocPos[imL], size=acfg.coinSize, opacity=0,
                                    image=acfg.imagePath + os.sep + acfg.coinPos[imL] + '.png')
        coins[imL].pos += offset
        coins[imL].setAutoDraw(True)

    MoneyBox = visual.TextStim(win=win, name='MoneyBox', text=format_wealth(wealth),
                            pos=acfg.textPos, height=acfg.textHeight, color='white')
    MoneyBox.pos += offset
    MoneyBox.setAutoDraw(False)

    Reminder = visual.TextStim(win=win, name='Reminder',
                            text='press\n\n\nearlier',
                            pos=acfg.textPos, height=acfg.textHeight, color='white')
    Reminder.pos += offset
    Reminder.setAutoDraw(False)

    trials = pd.read_csv(trialInfoPath, sep='\t')

    Initialization.setAutoDraw(False)
    win.flip()
    ########################### Instruction Screen #################################
    Instructions = visual.TextStim(win=win, name='instruction',
                                text=f'Press {responseKeyList[0]} or {responseKeyList[1]} to begin!',
                                pos=acfg.textPos, height=acfg.textHeight, color='white')
    Instructions.pos += offset
    Instructions.setAutoDraw(True)

    if expInfo['simulateMR'] in ['MRI', 'Simulate', 'MRIDebug']:
        Instructions.setText('Starting soon')
        win.flip()
        # Start simulator
        if expInfo['simulateMR'] in ['Simulate']:
            SyncGen.start()
        # Wait for triggers here:
        while Logger.tr < acfg.waitTR:
            Logger.keyStrokes(win)

    elif expInfo['simulateMR'] == 'None':
        win.flip()
        startResp = True

        if Agent.active:
            Agent.start_timer(Logger.getTime(), 10000, [1, 1, 2, 2], 1)
        # Wait for response
        while startResp:
            # Loop until response is received
            if Agent.active and Logger.getTime() > Agent.press_time:
                Agent.press()

            response = Logger.keyStrokes(win, keyList=responseKeyList)

            if response:
                startResp = False

    # Reset trial clock
    Logger.globalClock.reset()
    Instructions.setAutoDraw(False)

    ############################ Setup Elements ####################################
    MoneyBox.setAutoDraw(True)
    win.flip()

    ###################### This is were the experiment begins ######################
    noTrials = trials.shape[0] - nTrial
    terminateNormally = True

    for curTrial in range(noTrials):

        thisTrial = trials.iloc[nTrial].to_dict()

        if thisTrial != None:
            fractal1, fractal2 = int(thisTrial['fractal_left_up']), int(thisTrial['fractal_left_down'])
            fractal3, fractal4 = int(thisTrial['fractal_right_up']), int(thisTrial['fractal_right_down'])
            gamma1, gamma2 = thisTrial['gamma_left_up'], thisTrial['gamma_left_down']
            gamma3, gamma4 = thisTrial['gamma_right_up'], thisTrial['gamma_right_down']
            iti, eta = thisTrial['iti'], thisTrial['eta']
            onset_gamble1 = thisTrial['onset_gamble_pair_left']
            onset_gamble2 = thisTrial['onset_gamble_pair_right']
            coin_toss = np.int(thisTrial['gamble_up'])

        Logger.setTrialTime()
        Logger.trial = nTrial

        if thisTrial['no_brainer']:
            Logger.trial_type = 'NOB'
        elif thisTrial['stochasticly_dominated']:
            Logger.trial_type = '1SD'
        else:
            Logger.trial_type = 'NSD'

        currentFractals = [fractal1, fractal2, fractal3, fractal4]
        currentGammas = [gamma1, gamma2, gamma3, gamma4]
        # Logging dict - to include continuously updated info:
        logDict = {}
        ############################### ITI ########################################
        itiOnset = Logger.getTime()

        Wait.wait(iti)

        Logger.logEvent({"event_type": "ITI", "expected_duration": iti},
                        wealth=Logger.wealth, onset=itiOnset)
        ########################### Gamble Left ####################################
        fractals['leftUp'][fractal1].setOpacity(1)
        fractals['leftDown'][fractal2].setOpacity(1)

        # Timing of keystrokes
        Logger.keyStrokes(win)
        gambOnset = Logger.getTime()

        logDict.update({'fractal_left_up': fractalList[fractal1],
                        'fractal_left_down': fractalList[fractal2],
                        'gamma_left_up': gamma1,
                        'gamma_left_down': gamma2})

        if expInfo['active_mode'] != 2:
            win.flip()
            Logger.keyStrokes(win)

            Wait.wait(onset_gamble1, gambOnset)

            Logger.logEvent({"event_type": "GambleLeft", "expected_duration": onset_gamble1,
                            **logDict}, onset=gambOnset)

        ########################### Gamble Right ###################################
        fractals['rightUp'][fractal3].setOpacity(1)
        fractals['rightDown'][fractal4].setOpacity(1.0 * (expInfo['active_mode'] != 2))
        win.flip()
        Logger.keyStrokes(win)

        gambOnset = Logger.getTime()

        logDict.update({'fractal_right_up': fractalList[fractal3],
                        'fractal_right_down': fractalList[fractal4],
                        'gamma_right_up': gamma3,
                        'gamma_right_down': gamma4})

        Logger.logEvent({"event_type": "GambleRight", "expected_duration": onset_gamble2,
                        **logDict}, onset=gambOnset)
        ########################### Response Cue ###################################
        Logger.keyStrokes(win)
        ######################## Response Window ###################################
        respOnset = Logger.getTime()

        if Agent.active:
            Agent.start_timer(respOnset, Logger.wealth, currentGammas, eta)

        response = False
        responseTo = 'n/a'

        while (acfg.timeResponse + respOnset) > Logger.getTime() and not response:

            if Agent.active and Logger.getTime() > Agent.press_time:
                Agent.press()

            presses = Logger.keyStrokes(win, keyList=responseKeyList)

            if presses is not None:
                if 'left' in responseMapping[presses[0]]:
                    response = 'left'
                    if np.mean(currentGammas[:2]) >= np.mean(currentGammas[2:]):
                        responseTo = True
                    else:
                        responseTo = False

                if 'right' in responseMapping[presses[0]]:
                    response = 'right'
                    if np.mean(currentGammas[2:]) >= np.mean(currentGammas[:2]):
                        responseTo = True
                    else:
                        responseTo = False

                Logger.logEvent({"event_type": "Response",
                                'response_button': presses[0],
                                'response_time': presses[1] - respOnset,
                                'response_time_optimal': responseTo,
                                'response_is_last': False,
                                **logDict})

                win.flip()
                Logger.keyStrokes(win)

        ########################### Control Flow ###################################
        # Control flow - given response (or not)
        if response:
            ######################### Side Selection ###############################
            logDict.update({'no_response': False,
                            'response_time_optimal': responseTo,
                            'selected_side': response})

            if response == 'left':
                logDict.update({'chosen_expected_gamma': np.mean(currentGammas[:2])})
                fractals['rightUp'][fractal3].setOpacity(0)
                fractals['rightDown'][fractal4].setOpacity(0)

            elif response == 'right':
                logDict.update({'chosen_expected_gamma': np.mean(currentGammas[2:])})
                fractals['leftUp'][fractal1].setOpacity(0)
                fractals['leftDown'][fractal2].setOpacity(0)

            win.flip()
            Logger.keyStrokes(win)
            selectionOnset = Logger.getTime()

            Wait.wait(acfg.timeSideHighlight, start=selectionOnset)

            Logger.logEvent({"event_type": "SideSelection",
                            "expected_duration": acfg.timeSideHighlight, **logDict},
                            onset=selectionOnset)
            ############################ Coin ######################################
            if coin_toss:
                if (response=='right') and expInfo['active_mode'] == 2:
                    pass
                else:
                    coins[response + 'Up'].setOpacity(1)
                logDict.update({'gamble_up': 'up'})
            else:
                if (response == 'right') and expInfo['active_mode'] == 2:
                    pass
                else:
                    coins[response + 'Down'].setOpacity(1)

                logDict.update({'gamble_up': 'down'})

            win.flip()
            Logger.keyStrokes(win)

            coinOnset = Logger.getTime()
            Wait.wait(acfg.timeCoinToss, start=coinOnset)
            Logger.logEvent({"event_type": "Coin",
                            "expected_duration": acfg.timeCoinToss,
                            **logDict}, onset=coinOnset)
            ############################ Fractal Selection #########################
            if coin_toss:
                fractals['leftDown'][fractal2].setOpacity(0)
                fractals['rightDown'][fractal4].setOpacity(0)
            else:
                fractals['leftUp'][fractal1].setOpacity(0)
                fractals['rightUp'][fractal3].setOpacity(0)

            if response == 'left':
                ch_gamma = currentGammas[:2][np.abs(coin_toss -1)]
                logDict.update({'realized_gamma': ch_gamma})
            elif response == 'right':
                ch_gamma = currentGammas[2:][np.abs(coin_toss -1)]
                logDict.update({'realized_gamma': ch_gamma})

            win.flip()
            Logger.keyStrokes(win)
            fractalOnset = Logger.getTime()

            Wait.wait(acfg.timeFractalSelection, start=fractalOnset)
            Logger.logEvent({"event_type": "FractalSelection",
                            "expected_duration": acfg.timeFractalSelection,
                            **logDict},
                            onset=fractalOnset)

        else:
            ############################# No Response ##############################
            Reminder.setAutoDraw(True)

            logDict.update({'no_response': True})

            worst_fractal = np.argmin(currentGammas)

            for kk, imL in enumerate(acfg.imgLocation):
                if kk != worst_fractal:
                    fractals[imL][currentFractals[kk]].setOpacity(0)

            win.flip()
            Logger.keyStrokes(win)

            fractalOnset = Logger.getTime()
            Wait.wait(acfg.timeNoResponse, fractalOnset)

            Logger.logEvent({"event_type": "FractalSelection",
                            "expected_duration": fractalOnset, **logDict},
                            onset=fractalOnset)

            ch_gamma = currentGammas[worst_fractal]

            logDict.update({'realized_gamma': ch_gamma})

            Reminder.setAutoDraw(False)
            win.flip()
        ################################# Wealth Update ############################
        new_wealth = wealth_change(wealth, ch_gamma, eta)

        up_steps = int(np.rint(acfg.timeWealthUpdate / frameDur))

        wealth_steps = np.linspace(wealth, new_wealth, up_steps)
        wealth = new_wealth

        wealthOnset = Logger.getTime()

        for ws in wealth_steps:
            MoneyBox.setText(format_wealth(ws))
            Logger.keyStrokes(win)
            win.flip()

        Logger.logEvent({"event_type": "WealthUpdate",
                        "expected_duration": acfg.timeWealthUpdate, **logDict},
                        onset=wealthOnset, wealth=wealth)

        Wait.wait(acfg.timeFinalDisplay)

        MoneyBox.setText(format_wealth(wealth))
        ################################# Trial Completion ##########################
        # Resetting fractals and coins
        for n, imL in enumerate(acfg.imgLocation):
            fractals[imL][currentFractals[n]].setOpacity(0)
            Logger.keyStrokes(win)

        for imL in acfg.imgLocation:
                coins[imL].setOpacity(0)
                Logger.keyStrokes(win)

        win.flip()
        Logger.keyStrokes(win)

        Logger.logEvent({"event_type": "TrialEnd", **logDict})

        # Break condition time
        if Logger.getTime() > (expInfo['maxDuration'] - 10) or curTrial >= expInfo['maxTrial']:
            break

        if new_wealth >= 1e9 or new_wealth <= 0:
            Logger.logEvent({"event_type": "WealthCeilingReached", **logDict},
                            wealth=wealth)
            terminateNormally = False
            break

        nTrial += 1

    ################################ Post Experiment clean up ######################
    MoneyBox.setAutoDraw(False)
    Logger.keyStrokes(win)

    Outro = visual.TextStim(win=win, name='Outro', text='Thanks!', pos=acfg.textPos,
                            height=acfg.textHeight, color='white')
    Outro.pos += offset
    Outro.setAutoDraw(True)

    Wait.wait(1)

    if expInfo['simulateMR'] in ['Simulate']:
        SyncGen.stop()
        del SyncGen

    Logger.close()
    gc.collect()

    try:
        active_report(fileName)
    except:
        print("Report did not run.")

    Outro.setAutoDraw(False)
    if expInfo['simulateMR'] == 'MRIDebug':
        Counter.setAutoDraw(False)

    win.flip()

    return terminateNormally

