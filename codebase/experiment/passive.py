from copy import deepcopy
from psychopy import visual, core, event
import itertools
from psychopy.hardware.emulator import SyncGenerator
import numpy as np
import pandas as pd
import os
import gc
import typing
from typing import Type, List
from .exp import ExperimentLogger, PassiveAutoPilot, DebugLogger, ActiveAutoPilot
from .. import wealth_change
from .. import constants as con
from .exp import WaitTime, get_frame_timings, passive_report
from .exp import continue_from_previous, load_calibration, calculate_number_of_images
from .configs import passive_configs as pcfg
from .configs import active_configs as acfg
from typing import Optional, Dict
from .exp.dashboard import nobrainer_report
from .exp.helper import assign_fractals, gui_update_dict, DebugTimer, make_filename, format_wealth

####################### Setup GUI ##############################################
def passive_gui(filePath:str, expInfo:Optional[Dict] = None, spawnGui=True):

    if expInfo is None:
        expInfo = {'participant': '0',
                'run': '01',
                'eta': 0.0,
                'agentActive': True,
                'wealth': con.X0,
                'simulateMR': ['Simulate', 'MRI', 'None', 'MRIDebug'],
                'responseButton': 'space',
                'responseLeft': 'left',
                'responseRight': 'right',
                'fullScreen': False,
                'overwrite': True,
                'TR': 0.592,
                'maxDuration': 60,
                'maxTrial': np.inf,
                'maxRun': 4,
                'nTrial_NoBrainer': 10}

    if spawnGui:
        expInfo = gui_update_dict(expInfo, 'ergEx_Passive')

    # Parse GUI inputs into variables.
    expInfo['responseButton'] = str(expInfo['responseButton'])
    expInfo['responseLeft'] = str(expInfo['responseLeft'])
    expInfo['responseRight'] = str(expInfo['responseRight'])


    keyList = [expInfo['responseButton']]
    responseKeyList = [expInfo['responseLeft'], expInfo['responseRight']]

    responseMapping = {expInfo['responseLeft'] : 'left',
                       expInfo['responseRight']: 'right'}

    fileName = make_filename(filePath, expInfo['participant'], expInfo['eta'],
                             'passive', expInfo['run'])

    offset = load_calibration(filePath, expInfo['participant'], expInfo['eta'])

    wealth, nTrial, writeMode = continue_from_previous(fileName, expInfo['wealth'],
                                                    expInfo['overwrite'])

    # Pre-Calculate Number of MR-Images:
    trialInfoPath = make_filename('data/inputs/', expInfo['participant'], expInfo['eta'],
                                  'passive', extension='input.tsv')

    trialFile = pd.read_csv(trialInfoPath, sep='\t')
    noTR = calculate_number_of_images(trialFile[['iti', 'fractal_duration']],
                                    fixed_timings=[pcfg.timeToReminder, pcfg.timeWealthUpdate,
                                    pcfg.timeFinalDisplay, pcfg.wheelSpinTime], TR=expInfo['TR'])
    # Increment the number of images, by images recorded before experimental start.
    noTR += pcfg.waitTR

    expInfo['noTR'] = noTR
    expInfo['wealth'] = wealth
    expInfo['nTrial'] = nTrial
    expInfo['writeMode'] = writeMode
    expInfo['offset'] = offset
    expInfo['keyList'] = keyList
    expInfo['responseMapping'] = responseMapping
    expInfo['responseKeyList'] = responseKeyList

    return expInfo

################# Initialize Window and Classes ################################
def passive_run(expInfo:Dict, filePath:str, win:visual.Window = None,
               fractalList:List[str] = None):

    if win is None:
        win = visual.Window(size=[3072 / 2, 1920 / 2], fullscr=expInfo['fullScreen'],
                            screen=0, winType='pyglet', allowGUI=True, monitor=None,
                            color=[-1,-1,-1], colorSpace='rgb', units='pix')

    # Currently testing if the supposed ones are better.
    frameRate, frameDur =  get_frame_timings(win)

    wealth = expInfo['wealth']
    nTrial = expInfo['nTrial']
    offset = expInfo['offset']
    noTR = expInfo['noTR']
    keyList = expInfo['keyList']
    responseKeyList = expInfo['responseKeyList']
    responseMapping = expInfo['responseMapping']
    nTrial_noBrainer = expInfo['nTrial_noBrainer']

    if fractalList is None:
        fractalList = pcfg.fractalList
        print("Using default fractals")
    else:
        if len(fractalList) != con.N_FRACTALS:
            raise ValueError(f"Fractal List needs to be of length {con.N_FRACTALS}")
        if np.array(fractalList).dtype.type is not np.str_:
            raise ValueError("Fractal List needs to contain string!")

    # copy fractal, so that outside list not affected.
    fractalList = fractalList[:]
    fractalList.append('grey100')

    fileName = make_filename(filePath, expInfo['participant'], expInfo['eta'],
                             'passive', expInfo['run'])

    trialInfoPath = make_filename('data/inputs/', expInfo['participant'], expInfo['eta'],
                                  'passive', extension='input.tsv')

    # Create logger
    if expInfo['simulateMR'] == 'MRIDebug':
        Counter = visual.TextStim(win=win, name='Counter', text='0',
                                pos=(-300, -300), height=30, color='white')
        Counter.pos += offset
        Counter.setAutoDraw(True)
        Logger = DebugLogger(fileName=fileName, globalClock=core.Clock(),
                                wealth=wealth, participant_id=expInfo['participant'],
                                eta=expInfo['eta'], mode='passive', seq_tr=expInfo['TR'], run=expInfo['run'],
                                text=Counter)
        Logger.create(expInfo['writeMode'])

    else:
        Logger = ExperimentLogger(fileName=fileName, globalClock=core.Clock(),
                                wealth=wealth, participant_id=expInfo['participant'],
                                eta=expInfo['eta'], mode='passive', seq_tr=expInfo['TR'], run=expInfo['run'])
        Logger.create(expInfo['writeMode'])
    # Autopilot:
    Agent = PassiveAutoPilot(0.5, 0.3, expInfo['agentActive'],
                            responseButton=expInfo['responseButton'])
    # Waiting Tool
    Wait = WaitTime(win, Logger, frameDuration=frameDur)

    if expInfo['simulateMR'] in ['Simulate']:
        SyncGen = SyncGenerator(TR=expInfo['TR'], TA=expInfo['TR'] / 10, volumes=noTR)
    ###################### Preloading Images #######################################
    initialization = visual.TextStim(win=win, name='initialization',
                                    text='Initializing!!', pos=pcfg.textPos,
                                    height=pcfg.textHeight, ori=0.0, color='white')
    initialization.setAutoDraw(True)
    win.flip()

    fractals = {}
    for nFl, fl in enumerate(fractalList):
        fractals[nFl] = visual.ImageStim(win=win, pos=pcfg.imgPos, size=pcfg.imgSize,
                                        opacity=0,
                                        image=pcfg.imagePath + os.sep + 'fractals' + os.sep + fl + '.png')
        fractals[nFl].pos += offset
        fractals[nFl].setAutoDraw(True)

    Wheel = visual.ImageStim(win=win, name='wheel', image=pcfg.imagePath + os.sep + 'wheel.png',
                            mask='circle', ori=0.0, pos=pcfg.wheelPos,
                            size=pcfg.wheelSize, color=[1,1,1])
    Wheel.pos += offset
    Wheel.setAutoDraw(False)

    Stopper = visual.ShapeStim(win=win, name='stopper', size=pcfg.stopperSize,
                            vertices='triangle', ori=180.0,
                            pos=pcfg.stopperPos,
                            lineWidth=1.0, lineColor='blue', fillColor='blue')
    Stopper.pos += offset
    Stopper.setAutoDraw(False)

    MoneyFrame = visual.Rect(win=win, name='MoneyFrame', width=pcfg.boxSize[0],
                            height=pcfg.boxSize[1], ori=0.0, pos=pcfg.boxPos,
                            lineWidth=pcfg.boxWidth, lineColor=None, fillColor='grey',
                            opacity=0.8)
    MoneyFrame.pos += offset
    MoneyFrame.setAutoDraw(False)

    MoneyBox = visual.TextStim(win=win, name='MoneyBox',
                            text=format_wealth(wealth), pos=pcfg.textPos,
                            height=pcfg.textHeight,  color='white')
    MoneyBox.pos += offset
    MoneyBox.setAutoDraw(False)

    Reminder = visual.TextStim(win=win, name='Reminder',
                            text='press\n\n\nearlier',
                            pos=pcfg.textPos, height=pcfg.textHeight, color='white')
    Reminder.pos += offset
    Reminder.setAutoDraw(False)

    # set up handler to look after randomisation of conditions etc
    trials = pd.read_csv(trialInfoPath, sep='\t')

    initialization.setAutoDraw(False)
    win.flip()
    ########################### Instruction Screen #################################
    Instructions = visual.TextStim(win=win, name='instruction',
                                text=f'Press {keyList[0]} to continue!',
                                pos=pcfg.textPos, height=pcfg.textHeight, color='white')
    Instructions.pos += offset
    Instructions.setAutoDraw(True)

    if expInfo['simulateMR'] in ['MRI', 'Simulate', 'MRIDebug']:
        inst_text = 'Please Wait'
        Instructions.setText(inst_text + f'\nStart in {pcfg.waitTR}')
        win.flip()
            # Start simulator
        if expInfo['simulateMR'] in ['Simulate']:
            SyncGen.start()

        previousTR = Logger.tr
        # Wait for triggers here:
        while Logger.tr < pcfg.waitTR:
            Logger.keyStrokes(win)
            if previousTR < Logger.tr:
                Instructions.setText(inst_text + f'\nStart in {pcfg.waitTR - Logger.tr}')
                win.flip()
                previousTR = Logger.tr

    elif expInfo['simulateMR'] == 'None':
        win.flip()
        startResp = True

        if Agent.active:
            Agent.start_timer(0)
        # Wait for response
        while startResp:
            # Loop until response is received
            if Agent.active and Logger.getTime() > Agent.press_time:
                Agent.press()

            response = Logger.keyStrokes(win, keyList=keyList)

            if response:
                startResp = False

    Instructions.setAutoDraw(False)
    win.flip()
    ############################ Setup Elements ####################################
    Wheel.setAutoDraw(True)
    Stopper.setAutoDraw(True)
    MoneyFrame.setAutoDraw(True)
    MoneyBox.setAutoDraw(True)

    win.flip()
    ############################# Start Trial Handler ##############################
    # Reset trial clock
    Logger.globalClock.reset()
    ###################### This is were the experiment begins ######################
    noTrials = trials.shape[0] - nTrial

    for curTrial in range(noTrials):
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        thisTrial = trials.iloc[nTrial].to_dict()
        if thisTrial != None:
            gamma, fractal = thisTrial['gamma'], int(thisTrial['fractal'])
            eta, iti = thisTrial['eta'], thisTrial['iti']
            fractal_duration = thisTrial['fractal_duration']
            exp_wealth = thisTrial['p_seq_wealth']

        Logger.setTrialTime()
        Logger.trial = nTrial
        Logger.part = 0

        if gamma < 0:
            Logger.trial_type = 'negative'
        elif gamma > 0:
            Logger.trial_type = 'positive'
        elif gamma == 0:
            Logger.trial_type = 'neutral'

        # Logging dict - to include continuously updated info:
        logDict = {}
        ############################### ITI ########################################
        itiOnset = Logger.getTime()
        Wait.wait(iti)

        Logger.logEvent({"event_type": "ITI", "expected_duration": iti},
                        onset=itiOnset)
        ############################## Cue Onset ###################################
        MoneyFrame.setBorderColor("white")
        win.flip()
        Logger.keyStrokes(win)
        Logger.logEvent({"event_type": "ResponseCue"})
        ############################# Wait for response ############################
        startResp = True
        reminderPresent = False
        respOnset = Logger.getTime()
        remindTime = respOnset + pcfg.timeToReminder
        responseWindow = respOnset + pcfg.timeResponseWindow

        if Agent.active:
            Agent.start_timer(respOnset)

        # Loop until response is received
        while startResp:
            if Agent.active and Logger.getTime() > Agent.press_time:
                Agent.press()

            response = Logger.keyStrokes(win, keyList=keyList)

            if response:
                RT = response[1] - respOnset
                Logger.logEvent({'event_type': 'Response',
                                    'response_button': response[0],
                                    'response_time': RT,
                                    'response_late': RT > pcfg.timeToReminder},
                                    onset=respOnset)
                startResp = False

            # Duration for response window = 1 s.
            if not reminderPresent and remindTime < Logger.getTime():
                if not reminderPresent:
                    Logger.logEvent({"event_type": "ReminderOnset"})
                    reminderPresent = True
                    Reminder.setAutoDraw(True)
                    win.flip()

            if responseWindow <= Logger.getTime():
                Logger.logEvent({'event_type': 'ResponseTimeOut'})
                startResp = False
                RT = None
                logDict.update({'no_response': True})

        Reminder.setAutoDraw(False)
        MoneyFrame.setBorderColor("none")
        win.flip()
        Logger.keyStrokes(win)
        ############################## Wheel Spin ##################################
        if RT is not None:
            if reminderPresent is False:
                # Increase spin duration due to fast response
                spinDuration = pcfg.wheelSpinTime + max(pcfg.timeToReminder - RT, 0)
            elif reminderPresent:
                # Reduce spin duraion by response lateness.
                spinDuration = pcfg.wheelSpinTime - (RT - pcfg.timeToReminder) # This should be between 0 and 1
        else:
            # Reduce spin duration by (right now) 1 s.
            spinDuration = pcfg.wheelSpinTime - (pcfg.timeResponseWindow - pcfg.timeToReminder)

        steps = int(np.rint(spinDuration / frameDur)) - 1
        wheelOnset = Logger.getTime()
        for ii in range(steps):

            if ii < steps - 4:
                Wheel.setOri(np.mod(Wheel.ori + pcfg.revolution, 360))

            Logger.keyStrokes(win)
            win.flip()
            Logger.keyStrokes(win)

        Logger.logEvent({"event_type": "WheelSpin",
                        "expected_duration": spinDuration,
                        **logDict},
                        onset=wheelOnset)
        ############################ Fractal Onset #################################
        fractals[fractal].setOpacity(1)
        logDict.update({'gamma': gamma, 'fractal': fractalList[fractal]})
        win.flip()
        Logger.keyStrokes(win)

        fractalOnset = Logger.getTime()
        Wait.wait(fractal_duration)

        Logger.logEvent({"event_type": "FractalOnset",
                        "expected_duration": fractal_duration,
                        **logDict},
                        onset=fractalOnset)
        ########################### Wealth Update ##################################
        new_wealth = wealth_change(wealth, gamma, eta).item()

        new_wealth = exp_wealth
        up_steps = int(np.rint(pcfg.timeWealthUpdate / frameDur)) - 1
        wealth_steps = np.linspace(wealth, new_wealth, up_steps)
        wealth = new_wealth

        moneyOnset = Logger.getTime()
        for ws in wealth_steps:
            MoneyBox.setText(format_wealth(ws))
            Logger.keyStrokes(win)
            win.flip()
            Logger.keyStrokes(win)

        Logger.logEvent({"event_type": "WealthUpdate",
                        "expected_duration": pcfg.timeWealthUpdate,
                        **logDict}, wealth=wealth, onset=moneyOnset)

        MoneyBox.setText(format_wealth(wealth))
        Wait.wait(pcfg.timeFinalDisplay)

        fractals[fractal].setOpacity(0)
        win.flip()
        ########################## Fractal offset ##################################

        Logger.keyStrokes(win)

        Logger.wealth = wealth
        Logger.logEvent({"event_type": "TrialEnd", **logDict})
        nTrial += 1

        if Logger.getTime() > expInfo['maxDuration'] - 10 or curTrial >= expInfo['maxTrial']:
            break

    ################################ Post Experiment clean up ######################
    Wheel.setAutoDraw(False)
    Stopper.setAutoDraw(False)
    MoneyFrame.setAutoDraw(False)
    MoneyBox.setAutoDraw(False)
    Logger.keyStrokes(win)

    ############################### Nobrainer about here ###########################
    Agent = ActiveAutoPilot(0.4, 0.1, active=expInfo['agentActive'],
                            mode='random',
                            buttonLeft=expInfo['responseLeft'],
                            buttonRight=expInfo['responseRight'])

    imgLocation = ['leftUp', 'rightUp']
    fractals = {location: {} for location in imgLocation}

    for imL in imgLocation:
        for nFl, fl in enumerate(fractalList):
            fractals[imL][nFl] = visual.ImageStim(win=win, pos=[acfg.imgLocPos[imL][0], 0],
                                                size=acfg.imgSize, opacity=0,
                                                image=acfg.imagePath + os.sep + 'fractals' + os.sep + fl + '.png')

            fractals[imL][nFl].pos += offset
            fractals[imL][nFl].setAutoDraw(True)

    Reminder.setText("press earlier")
    fractalData = pd.read_csv(trialInfoPath, sep='\t')
    # Create dataset:
    fractalData = fractalData[['fractal', 'gamma']]
    unqFractals = np.unique(fractalData.fractal)
    unqFractals = unqFractals[unqFractals < con.N_FRACTALS]

    fractalCombination = list(itertools.combinations(unqFractals, 2))
    fractalCombination = np.array(fractalCombination)

    fractalGammaDict = {}
    for kk in unqFractals:
            fractalGammaDict[kk] = fractalData.query('fractal == @kk')['gamma'].values[0].item()

    randomIdx = np.random.choice(len(fractalCombination), len(fractalCombination), replace=False)
    trials  = pd.DataFrame(columns=['fractal1', 'fractal2', 'gamma1', 'gamma2'],
                          index=np.arange(nTrial_noBrainer))

    for nn, ri in enumerate(randomIdx[:nTrial_noBrainer]):
        pair = fractalCombination[ri, :].copy()

        if np.random.rand() < 0.5:
            trials.iloc[nn, :] = [pair[0], pair[1],
                                  fractalGammaDict[pair[0]],
                                  fractalGammaDict[pair[1]]]
        else:
            trials.iloc[nn, :] = [pair[1], pair[0],
                                  fractalGammaDict[pair[1]],
                                  fractalGammaDict[pair[0]]]

    ############################ Setup Elements ####################################
    if expInfo['feedback']:
        MoneyBox.setAutoDraw(True)

    win.flip()
    ###################### This is were Nobrainers begins ######################
    noTrials = trials.shape[0]
    iti = 1.0
    eta = expInfo['eta']

    logDict = {}
    Logger.logEvent({"event_type": "BeginningNobrainers"})

    for nbTrial in range(nTrial_noBrainer):
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        thisTrial = trials.iloc[nbTrial].to_dict()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            fractal1, fractal2 = thisTrial['fractal1'], thisTrial['fractal2']
            gamma1, gamma2 = thisTrial['gamma1'], thisTrial['gamma2']

        Logger.setTrialTime()
        Logger.trial = nTrial
        Logger.part = 1

        if thisTrial['gamma1'] > thisTrial['gamma2']:
            Logger.trial_type = 'left'
        if thisTrial['gamma2'] > thisTrial['gamma1']:
            Logger.trial_type = 'right'

        currentFractals = [fractal1, fractal2]
        currentGammas = [gamma1, gamma2]
        # Logging dict - to include continuously updated info:
        logDict = {}
        ############################### ITI ########################################
        itiOnset = Logger.getTime()

        Wait.wait(iti)

        Logger.logEvent({"event_type": "ITI", "expected_duration": iti},
                        wealth=Logger.wealth, onset=itiOnset)
        ########################### Gamble Left ####################################
        fractals['leftUp'][fractal1].setOpacity(1)
        fractals['rightUp'][fractal2].setOpacity(1)
        win.flip()
        ########################### Gamble Right ###################################
        Logger.keyStrokes(win)

        logDict.update({'fractal_right': fractal2, 'fractal_left': fractal1,
                        'gamma_right': gamma2, 'gamma_left': gamma1})

        Logger.logEvent({"event_type": "Decision", **logDict})

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
                    if currentGammas[0] >= currentGammas[1]:
                        responseTo = True
                    else:
                        responseTo = False

                if 'right' in responseMapping[presses[0]]:
                    response = 'right'
                    if currentGammas[1] >= currentGammas[0]:
                        responseTo = True
                    else:
                        responseTo = False

                # Response is last, currently not used...
                Logger.logEvent({"event_type": "Response",
                                'response_button': presses[0],
                                'response_time': presses[1] - respOnset,
                                'response_correct': responseTo,
                                **logDict})

        ########################### Control Flow ###################################
        # Control flow - given response (or not)
        if response:
            logDict.update({'no_response': False,
                            'response_correct': responseTo,
                            'selected_side': response})

            ######################### Side Selection ###############################
            if response == 'left':
                fractals['rightUp'][fractal2].setOpacity(0)
                logDict.update({'chosen_gamma': currentGammas[0]})
                ch_gamma = gamma1
            elif response == 'right':
                fractals['leftUp'][fractal1].setOpacity(0)
                logDict.update({'chosen_gamma': currentGammas[1]})
                ch_gamma = gamma2

            win.flip()

            Logger.keyStrokes(win)
            selectionOnset = Logger.getTime()

            Logger.logEvent({"event_type": "SideSelection",
                            "expected_duration": acfg.timeSideHighlight, **logDict},
                            onset=selectionOnset)

        else:
            ############################# No Response ##############################
            logDict.update({'no_response': True})
            worst_fractal = np.argmin(currentGammas)

            for kk, imL in enumerate(imgLocation):
                fractals[imL][currentFractals[kk]].setOpacity(0)

            Reminder.setAutoDraw(True)
            win.flip()
            Logger.keyStrokes(win)

            fractalOnset = Logger.getTime()

            ch_gamma = currentGammas[worst_fractal]
            logDict.update({'chosen_gamma': ch_gamma})

            Logger.logEvent({"event_type": "FractalSelection",
                            "expected_duration": acfg.timeNoResponse, **logDict},
                            onset=fractalOnset)

        ################################# Wealth Update ############################

        new_wealth = wealth_change(wealth, ch_gamma, eta).item()

        up_steps = int(np.rint(acfg.timeWealthUpdate / frameDur))

        wealth_steps = np.linspace(wealth, ch_gamma, up_steps)

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
        for n, imL in enumerate(imgLocation):
            fractals[imL][currentFractals[n]].setOpacity(0)
            Logger.keyStrokes(win)

        Reminder.setAutoDraw(False)
        win.flip()
        Logger.keyStrokes(win)

        Logger.logEvent({"event_type": "TrialEnd", **logDict})
        nTrial += 1

    ############################### Nobrainer over #################################
    Outro = visual.TextStim(win=win, name='outro',
                                text='Thanks!',
                                pos=pcfg.textPos, height=pcfg.textHeight, color='white')
    Outro.pos += offset
    Outro.setAutoDraw(True)

    Wait.wait(1)

    if expInfo['simulateMR'] in ['Simulate']:
        SyncGen.stop()
        del SyncGen

    Logger.close()

    gc.collect()

    try:
        # assert False
        passive_report(fileName)
    except:
        print("Passive Report did not run.")

    try:
        nobrainer_report(fileName)
    except:
        print("Nobrainer report did not run.")

    # Final clean up
    Outro.setAutoDraw(False)

    if expInfo['simulateMR'] == 'MRIDebug':
        Counter.setAutoDraw(False)

    win.flip()

    return wealth
