# %%
from mimetypes import init
from multiprocessing.resource_sharer import stop
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



def run_slideshow(win, expInfo, path=SLIDE_PATH, start_slide=0, stop_slide=None, win_size=[800, 600]):

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
                                        image=im, units='pix', size=win_size))
        images[-1].setAutoDraw(False)


    initialization.setAutoDraw(False)
    slideShow = True

    images[0].setAutoDraw(True)
    win.flip()

    min_slides = start_slide
    if stop_slide is None:
        max_slides = len(images) - 1
    else:
        max_slides = stop_slide

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