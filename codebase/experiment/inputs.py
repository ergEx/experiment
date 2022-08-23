import os
import sys

import numpy as np
import pandas as pd

from ..file_handler import make_filename
from ..sequences import generate_dataframes


def run(lambd:float, n_trials_passive:int, n_trials_active:int,
        save_path:str, mode:int = 1, speed_up:int = 1):

    p_df, a_df, meta = generate_dataframes(lambd=lambd,
                                           n_trials_active=n_trials_active,
                                           n_trials_passive=n_trials_passive,
                                           mode=mode,
                                           speed_up=speed_up
                                           )

    p_df.to_csv(save_path.replace('meta', 'passive').replace('txt', 'tsv'), index=False, sep='\t')
    a_df.to_csv(save_path.replace('meta', 'active').replace('txt', 'tsv'), index=False, sep='\t')

    with open(save_path,"w+") as f:
        f.writelines(meta)


def run_with_dict(expInfo):

    try:
        speed_up = expInfo['speed_up']
    except KeyError:
        speed_up = 1
        print("No speed up")

    save_path = make_filename('data/inputs/', expInfo['participant'], expInfo['eta'], 'meta', None, 'input.txt')

    reply = True

    if os.path.isfile(save_path):
        from psychopy import gui

        dlg = gui.Dlg(title="File exists!")
        dlg.addText(f'{save_path} already exitsts!')
        dlg.addField('Overwrite', choices=[True, False])
        reply = dlg.show()

        reply = (reply[0] and dlg.OK)

    if reply:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)

        run(lambd=expInfo['eta'],
            n_trials_passive=expInfo['n_trials_passive'],
            n_trials_active=expInfo['n_trials_active'],
            save_path=save_path,
            mode=expInfo['mode'],
            speed_up=speed_up)

    else:
        print(f"Not creating new inputs for participant {expInfo['participant']}")
        pass
