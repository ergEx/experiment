# %%
from mimetypes import init
import re
import os
from glob import glob
from psychopy import core,  gui, visual, data, event

SLIDE_PATH = 'instructions/First_draft_of_visual_instructions'

# From https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
# helps with missing trailing zeros:
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]

def nat_sorted(ls):
    natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]
    return sorted(ls.copy(), key=natsort)



def run_slideshow(win, expInfo, path=SLIDE_PATH):

    responseKeys = ['responseButton', 'responseLeft', 'responseRight']

    responseKeyList = [expInfo[rK] for rK in responseKeys]
    responseKeyList.append('q')

    responseMapping = {
                    expInfo['responseRight']: 'right',
                    expInfo['responseLeft']: 'left',
                    expInfo['responseButton']: 'space'}

    slides = glob(os.path.join(path, 'Slide*'))

    slides = nat_sorted(slides)

    initialization = visual.TextStim(win=win, name='initialization',
                                    text='Initializing!',
                                    height=12, ori=0.0, color='white')
    initialization.setAutoDraw(True)
    win.flip()

    images = []
    for im in slides:
        images.append(visual.ImageStim(win=win, opacity=1,
                                        image=im, size=(500, 500)))
        images[-1].setAutoDraw(False)


    initialization.setAutoDraw(False)
    slideShow = True

    images[0].setAutoDraw(True)
    win.flip()

    min_slides = 0
    max_slides = len(images) - 1
    sl_counter = 0

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

                elif 'space' in responseMapping[resp]:
                    slideShow = False

