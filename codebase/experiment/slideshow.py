# %%
import re
import os
from glob import glob
from psychopy import core, visual, event

SLIDE_PATH = 'instructions/ergEx_instructions_slideshow/'

# From https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
# helps with missing trailing zeros:
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def nat_sorted(ls):
    natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]
    return sorted(ls.copy(), key=natsort)


def run_slideshow(win, expInfo, path=SLIDE_PATH, start_slide=0, stop_slide=None, win_size=[800, 600]):

    responseKeys = ['responseButton', 'responseLeft', 'responseRight']

    if expInfo['simulateMR'] in ['MRI', 'Simulate', 'MRIDebug']:
        path = 'instructions/ergEx_instructions_fmri/'

    responseKeyList = [expInfo[rK] for rK in responseKeys]
    responseKeyList.append('q')

    responseMapping = {
                    expInfo['responseRight']: 'right',
                    expInfo['responseLeft']: 'left',
                    expInfo['responseButton']: 'space'}

    slides = glob(os.path.join(path, 'Slide*'))

    slides = nat_sorted(slides)[start_slide : stop_slide + 1]

    initialization = visual.TextStim(win=win, name='initialization',
                                    text='Initializing!',
                                    height=12, ori=0.0, color='white')
    initialization.setAutoDraw(True)
    win.flip()
    images = []

    width_ratio = win.size[0] / win.size[1]

    for im in slides:
        images.append(visual.ImageStim(win=win, opacity=1,
                                        image=im, units='height', size=[width_ratio * 1, 1]))
        images[-1].setAutoDraw(False)


    initialization.setAutoDraw(False)
    slideShow = True

    min_slides = 0
    max_slides = len(images) - 1

    images[min_slides].setAutoDraw(True)
    win.flip()

    sl_counter = min_slides
    allow_skip = False

    while slideShow:

        presses = event.getKeys(keyList=responseKeyList)

        if presses:
            # Assuming that last press is most important! (could have issues - need more checks)
            for resp in presses:

                if 'q' == resp:
                    win.close()
                    core.quit()

                elif 'left' in responseMapping[resp]:
                    images[sl_counter].setAutoDraw(False)
                    sl_counter = max([min_slides, sl_counter - 1])
                    images[sl_counter].setAutoDraw(True)
                    win.flip()
                elif 'right' in responseMapping[resp]:
                    images[sl_counter].setAutoDraw(False)
                    sl_counter = min([max_slides, sl_counter + 1])
                    images[sl_counter].setAutoDraw(True)
                    win.flip()

                elif ('space' in responseMapping[resp]) and allow_skip:
                    slideShow = False

        if sl_counter == max_slides:
            allow_skip = True