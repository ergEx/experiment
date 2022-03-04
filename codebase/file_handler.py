import os
import re
import numpy as np
from typing import List, Union


def lambd_to_bids(lambd):
    bids = np.abs(lambd)
    bids = str(bids).replace('.', 'd')

    if lambd < 0:
        bids = 'm' + bids

    check_bids_lambd(bids)

    return bids


def bids_to_lambd(bids):

    if bids[0] == 'm':
        sign = -1
        lambd = bids[1:]
    else:
        sign = 1
        lambd = bids

    lambd = lambd.replace('d', '.')

    lambd = float(lambd)
    lambd = lambd * sign

    check_lambd_bids(lambd)

    return lambd


def check_bids_lambd(bids):

    if len(bids) > 4:
        raise ValueError(f"{bids} is not in correct format.")

    if bids[0] not in ['m', '0', '1']:
        raise ValueError("{bids} needs to start with m or be 0 or 1")


def check_lambd_bids(lambd):

    if not isinstance(lambd, float):
        raise ValueError(f"{lambd} should be float")

    if lambd not in [1.0, 0.0, -1.0, 0.5, -0.5]:
        raise ValueError(f"{lambd} not in correct range!")


def check_file_parts(fparts:List):

    for obj in fparts:
        if re.match('^[^-_.]*$', str(obj)) is None:
            raise ValueError( f'{obj} cannot contain "-" or "_"')


def make_bids_base(sub:str, lambd:str, task:str, run:int = None):

    if isinstance(lambd, float):
        lambd = lambd_to_bids(lambd)

    if run is not None:
        try:
            run = int(run)
            if run <= 0 or run >= 10:
                raise ValueError('Check run parameter!')
        except:
            raise ValueError('Run has to be integer compatible and less than 10!')

    check_file_parts([sub, lambd, task, run])

    if run is not None:
        return f'sub-{sub}_ses-lambd{lambd}_task-{task}_run-{int(run)}'
    else:
        return f'sub-{sub}_ses-lambd{lambd}_task-{task}'


def make_bids_dir(sub:str, lambd:Union[str, float]):


    if isinstance(lambd, float):
        lambd = lambd_to_bids(lambd)

    check_file_parts([sub, lambd])

    return os.path.join(f'sub-{sub}', f'ses-lambd{lambd}')


def make_filename(file_path:str, sub:str, lambd:float, task:str, run:int = None,
                  extension:str = 'events.tsv', add_dir:bool = True):

    lambd = lambd_to_bids(lambd)

    check_file_parts([sub, lambd, task, run])

    if add_dir:
        dirs = make_bids_dir(sub, lambd)
    else:
        dirs = ''

    fname = make_bids_base(sub, lambd, task, run) + '_' + extension

    fullname = os.path.join(file_path, dirs, fname)

    return fullname


def extract_from_fname(filename):

    run = int(re.search('run-\d', filename)[0][-1])
    lambd = bids_to_lambd(re.search('ses-lambd[^[_\/\\]]*', filename)[0][9:])
    sub = re.search('sub-[^_]{1,}', filename)[0].split('-')[-1]
    task = re.search('task-[^_]{1,}', filename)[0].split('-')[-1]
    filepath, _ = os.path.split(filename)

    return filepath, sub, lambd, task, run