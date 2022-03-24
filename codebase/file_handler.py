"""File handling utilities, creates BIDS compatible path names and checks if
lambda is in correct name. Can also extract important info from pathname.
"""
import os
import re
import numpy as np
from typing import List, Union, Tuple


def lambd_to_bids(lambd:float) -> str:
    """Transforms lambda/eta into bids format (replacing "-" with "m" and
    '.' with 'd').

    Args:
        lambd (float): Lambda of the dynamic.

    Returns:
        str: Lambda in BIDS compatible form.
    """

    bids = np.abs(lambd)
    bids = str(bids).replace('.', 'd')

    if lambd < 0:
        bids = 'm' + bids

    check_bids_lambd(bids)

    return bids


def bids_to_lambd(bids:str) -> float:
    """Transforms BIDS lambda back to float.

    Args:
        bids (str): Lambda in BIDS format.

    Returns:
        float: Lambda as float.
    """

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


def check_bids_lambd(bids:str):
    """Checks if BIDS str is in correct format.

    Args:
        bids (str): Lambda in BIDS format.

    Raises:
        ValueError: Raises error if too long.
        ValueError: Raises error if begins with wrong letter.
    """

    if len(bids) > 4:
        raise ValueError(f"{bids} is not in correct format.")

    if bids[0] not in ['m', '0', '1']:
        raise ValueError("{bids} needs to start with m or be 0 or 1")


def check_lambd_bids(lambd:float):
    """Checks if lambda is correct.

    Args:
        lambd (float): Lambda of the dynamic.

    Raises:
        ValueError: Raises error if not float.
        ValueError: Raises error if lambda not in supported dynamics.
    """
    if not isinstance(lambd, float):
        raise ValueError(f"{lambd} should be float")

    if lambd not in [1.0, 0.0, -1.0, 0.5, -0.5]:
        raise ValueError(f"{lambd} not in correct range!")


def check_file_parts(fparts:List):
    """Checks if entry in list contains invalid characters.

    Args:
        fparts (List): List of file parts.

    Raises:
        ValueError: Raises error if file part contains invalid character.
    """
    for obj in fparts:
        if re.match('^[^-_.]*$', str(obj)) is None:
            raise ValueError( f'{obj} cannot contain ".", "-" or "_"')


def make_bids_base(sub:str, lambd:str, task:str, run:int = None) -> str:
    """Creates BIDS filename.

    Args:
        sub (str): Participant ID
        lambd (str): Dynamic
        task (str): The task
        run (int, optional): Which run, if included. Defaults to None.

    Raises:
        ValueError: If run is out of range.
        ValueError: If run is not an integer (or compatible with integer).

    Returns:
        str: String in form: sub-XXX_ses-lambdXXX_task-XXX(_run-X), where run is
            optional.
    """

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


def make_bids_dir(sub:str, lambd:Union[str, float]) -> str:
    """Returns dictionary structure in BIDS format.

    Args:
        sub (str): Participant ID.
        lambd (Union[str, float]): Dynamic.

    Returns:
        str: Directory name in form /sub-XXX/ses-lambdXXX
    """

    if isinstance(lambd, float):
        lambd = lambd_to_bids(lambd)

    check_file_parts([sub, lambd])

    return os.path.join(f'sub-{sub}', f'ses-lambd{lambd}')


def make_filename(file_path:str, sub:str, lambd:float, task:str, run:int = None,
                  extension:str = 'events.tsv', add_dir:bool = True) -> str:
    """Creates whole filename, including directory.

    Args:
        file_path (str): Path to BIDS directory.
        sub (str): Participant ID
        lambd (float): Dynamic
        task (str): Task
        run (int, optional): Which run. Defaults to None.
        extension (str, optional): File ending. Defaults to 'events.tsv'.
        add_dir (bool, optional): If to create directory. Defaults to True.

    Returns:
        str: Directory and filename.
    """
    lambd = lambd_to_bids(lambd)

    check_file_parts([sub, lambd, task, run])

    if add_dir:
        dirs = make_bids_dir(sub, lambd)
    else:
        dirs = ''

    fname = make_bids_base(sub, lambd, task, run) + '_' + extension

    fullname = os.path.join(file_path, dirs, fname)

    return fullname


def extract_from_fname(filename:str) -> Tuple[str, str, float, str, int]:
    """Extract parts from BIDS filename.

    Args:
        filename (str): BIDS filename.

    Returns:
        Tuple[str, str, float, str, int]: Returns: base path, subject ID, dynamic,
        task, and run.
    """

    run = int(re.search('run-\d', filename)[0][-1])
    lambda_regex = re.compile(r'ses-lambd[^_\/\\]{3,4}')
    lambd = bids_to_lambd(re.search(lambda_regex, filename)[0][9:])
    sub = re.search(r'sub-[^_]{1,}', filename)[0].split('-')[-1]
    task = re.search(r'task-[^_]{1,}', filename)[0].split('-')[-1]
    filepath, _ = os.path.split(filename)

    return filepath, sub, lambd, task, run