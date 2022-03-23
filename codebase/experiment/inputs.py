import sys
import numpy as np
import pandas as pd
import os
from .. import constants as con
from ..sequences import generate_dataframes
from ..file_handler import make_filename


def run(lambd:float, x_0:int, n_repeats_passive:int, n_trials_active:int,
        save_path:str, passive_mode:int = 1, speed_up:int = 1):

    p_df, a_df, meta = generate_dataframes(lambd=lambd,
                                           x_0=x_0,
                                           n_trials_active=n_trials_active,
                                           n_repeats_passive=n_repeats_passive,
                                           passive_mode=passive_mode,
                                           speed_up=speed_up,
                                           indifference_etas = con.INDIFFERENCE_ETAS,
                                           indiffrence_x_0 = con.INDIFFERENCE_X_0,
                                           indifference_dx2 = con.INDIFFERENCE_DX2)

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
            x_0=con.X0,
            n_repeats_passive=expInfo['n_repeats_passive'],
            n_trials_active=expInfo['n_trials_active'],
            save_path=save_path,
            passive_mode=expInfo['passive_mode'],
            speed_up=speed_up)

    else:
        print(f"Not creating new inputs for participant {expInfo['participant']}")
        pass
