import sys
import numpy as np
import pandas as pd
import os
from .. import constants as con
from .. import calculate_c, DotDict
from ..sequences import generate_dataframes
from ..file_handler import make_filename


def run(eta:float, c:float, n_repeats_passive:int, n_trials_active:int,
        save_path:str, passive_mode:int = 1, active_mode:int = 1,
        speed_up:int = 1):

    p_df, a_df, meta = generate_dataframes(eta=eta, c=c,
                                           n_trials_active=n_trials_active,
                                           n_repeats_passive=n_repeats_passive,
                                           passive_mode = passive_mode,
                                           active_mode = active_mode,
                                           speed_up=speed_up)

    p_df.to_csv(save_path.replace('meta', 'passive').replace('txt', 'tsv'), index=False, sep='\t')
    a_df.to_csv(save_path.replace('meta', 'active').replace('txt', 'tsv'), index=False, sep='\t')

    with open(save_path,"w+") as f:
        f.writelines(meta)


def run_with_dict(expInfo):

    c = calculate_c(expInfo['eta'])

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

        run(eta=expInfo['eta'], c=c, n_repeats_passive=expInfo['n_repeats_passive'],
            n_trials_active=expInfo['n_trials_active'],
            passive_mode=expInfo['passive_mode'],
            active_mode=expInfo['active_mode'],
            speed_up=speed_up, save_path=save_path)
    else:
        print(f"Not creating new inputs for participant {expInfo['participant']}")
        pass
