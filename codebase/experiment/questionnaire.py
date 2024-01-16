from psychopy import visual, event
import pandas as pd
import numpy as np


def run_questionnaire(win, fname, participant, leftKey, rightKey, acceptKey, outpath):

    questionnaire = pd.read_csv(fname, sep='\t', index_col='index', encoding='utf-8')
    responsefile = questionnaire.copy()
    responsefile['response'] = 'NaN'
    responsefile['participant'] = participant
    responsefile['starting_response'] = 'NaN'
    responsefile['display_order'] = 'NaN'

    intro = ('In the following we will present you with a questionnaire.\n' +
            'Please pay close attention to the instructions at the top of the screen and below the rating scale.\n\n\n' +
            f'Press MIDDLE to continue.')

    textIntro = visual.TextStim(win, intro, pos=(-0.7, 0), units='norm', wrapWidth=1.4, alignText='center', height=0.075,  anchorHoriz='left')
    textIntro.draw()
    win.flip()

    event.waitKeys(keyList=[acceptKey])

    text1 = visual.TextStim(win, '', pos=(-0.7, 0.8), units='norm', wrapWidth=1.4, alignText='left', height=0.05,  anchorHoriz='left')
    text2 = visual.TextStim(win, '', pos=(-0.7, 0.4), units='norm', wrapWidth=1.4, alignText='center', height=0.075,  anchorHoriz='left')
    text3 = visual.TextStim(win, f'Press LEFT to move left and RIGHT to move right.\nPress MIDDLE to select your response.',
            pos=(-0.7, -0.5), units='norm', wrapWidth=1.4, alignText='center', height=0.04,  anchorHoriz='left')
    text3.setAutoDraw(True)

    for trial, item in enumerate(list(questionnaire.index)):

        labs = questionnaire.loc[item, 'options'].split(',')
        labs = [i[::-1].replace(' ', '\n', 1)[::-1] for i in labs]

        text1.setText(questionnaire.loc[item, 'questionText'])
        text2.setText(questionnaire.loc[item, 'itemText'])

        if len(labs) < questionnaire.loc[item, 'elements']:
            ratingScale = visual.RatingScale(
                win,
                labels=labs,
                low=1,
                high=questionnaire.loc[item, 'elements'],
                singleClick=False,
                scale=None,
                markerStart=1,
                leftKeys=leftKey,
                rightKeys=rightKey,
                acceptKeys=acceptKey,
                showValue=True,
                marker='glow',
                markerExpansion=0,
                skipKeys=None,
                noMouse=True,
                size=1.2,
                stretch=1.5,
                pos=(0, 0),
                showAccept=False,
                textSize=0.5,
                tickHeight=1.0,
                disappear=True)

        else:
            ratingScale = visual.RatingScale(
                win,
                labels=labs,
                low=1,
                high=questionnaire.loc[item, 'elements'],
                singleClick=False,
                scale=None,
                markerStart=1,
                leftKeys=leftKey,
                rightKeys=rightKey,
                acceptKeys=acceptKey,
                showValue=True,
                marker='glow',
                markerExpansion=0,
                skipKeys=None,
                noMouse=True,
                size=1.2,
                stretch=1.5,
                pos=(0, 0),
                showAccept=False,
                textSize=0.5,
                tickHeight=1.0,
                tickMarks=np.arange(1, questionnaire.loc[item, 'elements'] + 1).tolist(),
                disappear=True)

        while ratingScale.noResponse:
            ratingScale.draw()
            text1.draw()
            text2.draw()
            win.flip()

        responsefile.loc[item, 'response'] = ratingScale.getRating()
        responsefile.loc[item, 'starting_response'] = 1
        responsefile.loc[item, 'display_order'] = trial

    responsefile.to_csv(outpath, sep='\t')
    text3.setAutoDraw(False)
