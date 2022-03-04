from PIL import Image
import os
from glob import glob

PATH_IN = '/Users/simons/Desktop/materials_for_P0010.5/stimuli/exp2/images/'
PATH_OUT ='/Users/simons/Desktop/ergExPy/stimuli/fractals'
IMAGESIZE = (425, 319)


if __name__ == '__main__':
    images = glob(PATH_IN + '*.png')

    for img_path in images:

        img = Image.open(img_path)
        img_name = img_path.split(os.sep)[-1]
        img = img.resize(IMAGESIZE, resample=Image.BICUBIC)
        img.save(PATH_OUT + os.sep + img_name)
