from .exp import ExperimentLogger, get_frame_timings
from psychopy import core,  gui, visual, data
from .configs import active_configs as acfg
from .configs import STIMULUSPATH, DEFAULT_FRACTALS
import os
from typing import Optional, Dict
from .exp.helper import gui_update_dict, make_filename
####################### Setup GUI ##############################################

def calibration_run(filepath:str, expInfo:Optional[Dict]=None,
                win:Optional[visual.Window]=None):

    responseKeys = ['responseUp', 'responseDown', 'responseButton',
                    'responseLeft', 'responseRight']

    for rK in responseKeys:
        expInfo[rK] = str(expInfo[rK])

    filename = make_filename(filepath, expInfo['participant'], expInfo['session'],
                            expInfo['eta'], 'calibration', 1)

    responseKeyList = [expInfo[rK] for rK in responseKeys]

    responseMapping = {expInfo['responseUp'] : 'up',
                    expInfo['responseDown']: 'down',
                    expInfo['responseRight']: 'right',
                    expInfo['responseLeft']: 'left',
                    expInfo['responseButton']: 'space'}

    # Create clocks:
    GlobalClock = core.Clock()
    Logger = ExperimentLogger(fileName=filename, globalClock=GlobalClock,
                            wealth=None, participant_id=expInfo['participant'],
                            eta=0, mode='calibration')
    Logger.create('w')
    ###################### Preloading Images #######################################
    MoneyFrame = visual.Rect(win=win, name='MoneyFrame', width=acfg.boxSize[0],
                             height=acfg.boxSize[1], pos=acfg.boxPos,
                             lineWidth=1.0, fillColor='grey', opacity=0.5)
    MoneyFrame.setAutoDraw(True)

    MoneyBox = visual.TextStim(win=win, name='MoneyBox', text=format(1000, "07.2f"),
                               pos=acfg.textPos, height=acfg.textHeight, color='white')
    MoneyBox.setAutoDraw(True)

    fractals = []
    for imL in acfg.imgLocation:
        fractals.append(visual.ImageStim(win=win, pos=acfg.imgLocPos[imL],
                                        size=acfg.imgSize, opacity=1,
                                        image=os.path.join(STIMULUSPATH, DEFAULT_FRACTALS[0] + '.png')))
        fractals[-1].setAutoDraw(True)

    onScreen = [MoneyBox, MoneyFrame, *fractals]
    origPos = [onSobj.pos for onSobj in onScreen]

    win.flip()

    calibratingWindow = True

    x_offset = 0
    y_offset = 0

    while calibratingWindow:

        presses = Logger.keyStrokes(win, keyList=responseKeyList)
        if presses is not None:

            if 'left' in responseMapping[presses[0]]:
                x_offset = x_offset - 5
            if 'right' in responseMapping[presses[0]]:
                x_offset = x_offset + 5
            if 'up' in responseMapping[presses[0]]:
                y_offset = y_offset + 5
            if 'down' in responseMapping[presses[0]]:
                y_offset = y_offset - 5

            for onSobj, onSobjPos in zip(onScreen, origPos):
                onSobj.pos = onSobjPos + (x_offset, y_offset)

            win.flip()

            if 'space' in responseMapping[presses[0]]:

                Logger.logEvent({'x_offset': x_offset, 'y_offset': y_offset})
                calibratingWindow = False

                for f in onScreen:
                    f.setAutoDraw(False)

                win.flip()

    Logger.close()
