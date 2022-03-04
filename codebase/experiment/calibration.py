from .exp import ExperimentLogger, get_frame_timings
from psychopy import core,  gui, visual, data
from .configs import active_configs as acfg
import os
from typing import Optional, Dict
from .exp.helper import gui_update_dict, make_filename
####################### Setup GUI ##############################################

def calibration_run(filepath:str, expInfo:Optional[Dict]=None,
                win:Optional[visual.Window]=None):

    if expInfo is None:

        expInfo = {'participant': '0',
                'session': '001',
                'responseUp': 'up',
                'responseDown': 'down',
                'responseLeft': 'left',
                'responseRight': 'right',
                'responseSave': 'space',
                'fullScreen': False}

    if win is None:
        win = visual.Window(size=[3072 / 2, 1920 / 2], fullscr=expInfo['fullScreen'],
                            screen=0, winType='pyglet', allowGUI=True, monitor=None,
                            color=[-1,-1,-1], colorSpace='rgb', units='pix')

    responseKeys = ['responseUp', 'responseDown', 'responseSave',
                    'responseLeft', 'responseRight']

    for rK in responseKeys:
        expInfo[rK] = str(expInfo[rK])

    filename = make_filename(filepath, expInfo['participant'], expInfo['eta'],
                             'calibration', 1)

    responseKeyList = [expInfo[rK] for rK in responseKeys]

    responseMapping = {expInfo['responseUp'] : 'up',
                    expInfo['responseDown']: 'down',
                    expInfo['responseRight']: 'right',
                    expInfo['responseLeft']: 'left',
                    expInfo['responseSave']: 'space'}

    frameRate, frameDur = get_frame_timings(win)

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

    FrameLeft = visual.Rect(win=win, name='FrameLeft', width=acfg.frameSize[0],
                            height=acfg.frameSize[1], pos=acfg.frameLeftPos,
                            lineWidth=10.0, lineColor='white', fillColor=None,
                            opacity=0.25)
    FrameLeft.setAutoDraw(True)

    FrameRight = visual.Rect(win=win, name='FrameRight', width=acfg.frameSize[0],
                             height=acfg.frameSize[1], pos=acfg.frameRightPos,
                             lineWidth=10.0, lineColor='white', fillColor=None,
                             opacity=0.25)
    FrameRight.setAutoDraw(True)

    fractals = []
    for imL in acfg.imgLocation:
        fractals.append(visual.ImageStim(win=win, pos=acfg.imgLocPos[imL],
                                        size=acfg.imgSize, opacity=1,
                                        image=(os.path.join(acfg.imagePath, acfg.fractalList[0]) + '.png')))
        fractals[-1].setAutoDraw(True)

    onScreen = [MoneyBox, MoneyFrame, FrameLeft, FrameRight, *fractals]
    origPos = [onSobj.pos for onSobj in onScreen]

    win.flip()

    calibratingWindow = True

    x_offset = 0
    y_offset = 0

    while calibratingWindow:

        presses = Logger.keyStrokes(win, keyList=responseKeyList)
        if presses is not None:

            if 'left' in responseMapping[presses[0]]:
                x_offset = x_offset - 1
            if 'right' in responseMapping[presses[0]]:
                x_offset = x_offset + 1
            if 'up' in responseMapping[presses[0]]:
                y_offset = y_offset + 1
            if 'down' in responseMapping[presses[0]]:
                y_offset = y_offset - 1

            for onSobj, onSobjPos in zip(onScreen, origPos):
                onSobj.pos = onSobjPos + (x_offset, y_offset)

            win.flip()

            if 'space' in responseMapping[presses[0]]:
                # Response is last, currently not used...
                Logger.logEvent({'x_offset': x_offset, 'y_offset': y_offset})
                calibratingWindow = False

                for f in onScreen:
                    f.setAutoDraw(False)

                win.flip()
    # win.close()
    Logger.close()
