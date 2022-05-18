# %%
import pandas as pd
import numpy as np
from PIL import Image
import os

if __name__ == '__main__':

    frac_norms = pd.read_csv('/Users/simons/Desktop/experiment/data/stimuli/fractals_norms.csv')

    frac_norms = frac_norms.loc[frac_norms['d_prime'].isna() == False, :]

    frac_norms = frac_norms.sort_values(by=['d_prime', 'memorability'])
    # %% Highest counts:
    # There are elements with d_prime, we want to reduce to 50
    frac_norms['modal_name_english'].value_counts()
    # there are 5 "flower" (remove 3)
    # there are 4 "Snail"  (remove 3, cumulative = 7)
    # there are 3 "spiral" (remove 2, cumulative = 9)
    # there are 3 "waves" (remove 2, cumulative)

    # Multiple 2 counts - but so far only removing multiple counts of the first set:
    remove_flower = frac_norms.eval("modal_name_english == 'flower'")
    remove_snail = frac_norms.eval("modal_name_english == 'snail'")
    remove_spiral = frac_norms.eval("modal_name_english == 'spiral'")
    remove_waves = frac_norms.eval("modal_name_english == 'waves'")

    # Removing entries, mostly at the lower end of the data, thus less d-prime:
    remove_entries = remove_flower | remove_snail | remove_waves | remove_spiral
    cum_entries = np.cumsum(remove_entries) # There are 15 entries
    remove_entries = remove_entries & (cum_entries > 5) # Removing over 5 reduces 50
    frac_reduced = frac_norms.loc[remove_entries == False, :]



    PATH_IN = '/Users/simons/Desktop/experiment/data/stimuli/Fractals_400_380x380_grey_SpecHist_SSIM/'
    PATH_OUT ='/Users/simons/Desktop/experiment/data/stimuli/fractals/'
    IMAGESIZE = (425, 319)


    for n, img in enumerate(frac_reduced.fractal.values):

        img = Image.open(PATH_IN + img + '.bmp')
        print(img)

        img_name = f'F{n:03d}'
        img = img.resize(IMAGESIZE, resample=Image.BICUBIC)
        print(PATH_OUT + os.sep + img_name + '.png')
        img.save(PATH_OUT + os.sep + img_name + '.png')
