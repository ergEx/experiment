from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
import os

if __name__ == '__main__':

    slides = glob('ergEx_instructions_fmri/*.jpg')

    for n, img_path in enumerate(slides):

        print(img_path)
        img = Image.open(img_path)

        img = img.resize((1920, 1080), resample=Image.BICUBIC)
        img.save(img_path, subsampling=0, quality=100)
