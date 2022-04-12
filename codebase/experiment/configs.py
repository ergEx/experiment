"""
Configs for the experiment, change of global properties (sizes, positions, timing.)
All timings are in seconds.
All sizes are in pixels.
"""
from typing import Dict
from ..utils import DotDict
from .. import constants as con
import numpy as np

STIMULUSPATH = 'data/stimuli/'
"""Path to stimulus folder"""
IMGSIZE = (425, 319)
""" Image Size """
DEFAULT_FRACTALS = ['F000', 'F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008']
""" Default fractals  """
CENTER_POS = (0, 0)
""" Center location on screen """
TEXT_HEIGHT = 25
""" Height of text """

passive_configs = DotDict()

passive_configs['centerPos'] = CENTER_POS
""" Position of the center """
passive_configs['imgSize'] = (IMGSIZE[0] * 2, IMGSIZE[1] * 2) # Size of the center image
""" Size of the image in the passive run"""
passive_configs['textHeight'] = TEXT_HEIGHT # Height of text
""" Heights of the text."""

passive_configs['boxSize'] = (180, 30)
""" Size of the box in the center (i.e. the frame of the money)"""
passive_configs['boxWidth'] = 3
""" Linewidth of the money box."""

passive_configs['wheelSize'] = (300, 300) # Size of the wheel image
""" Size of the wheel image. """
passive_configs['revolution'] = 5 # Rotation angle of wheel
""" Rotation angle of the wheel. """

passive_configs['stopperSize'] = (30, 30) # Positon of blue triangle
""" Size of the triangle, acting as a stopper of the wheel."""
passive_configs['stopperPos'] = (0, 160)
""" Position of above triangle. """

# Timings
passive_configs['waitTR'] = 1 # How many TRs to wait
""" How many TRs to wait before the experiment begins"""

passive_configs['timeToReminder'] = 1.0 # How much time until reminder is shown
""" Time after which the press-earlier reminder is shown. """
passive_configs['timeResponseWindow'] = 2.0 # time of the response window
""" Maximal response time """
passive_configs['wheelSpinTime'] = 2.0 # How long the wheel spins at base rate
""" How long the wheel should spin. """
passive_configs['timeWealthUpdate'] = 1.0 # Rolling of the wealth display
""" How long wealth does take to roll up or down. """
passive_configs['timeFinalDisplay'] = 1.0 # How long wealth image is staying on
""" How long the final display is shown (i.e. wealth and fractal before reset) """

############################### Active settings

active_configs = DotDict()
# Image sizes
imgX = 312 # Img, coin, and frame position.
imgY = 200

IMG_POS = [(-imgX, imgY), (-imgX, -imgY), (imgX, imgY), (imgX, -imgY)] # Pos list

active_configs['imgLocation'] = ['leftUp', 'leftDown', 'rightUp', 'rightDown']
""" Location of the images """
active_configs['imgLocPos'] = {imL: imP for imL, imP in zip(active_configs.imgLocation, IMG_POS)} # Dict for pos
""" Dictionary containing image location on screen and the corresponding position """
active_configs['imgSize'] = IMGSIZE
""" The size of the four / three fractals. """

active_configs['coinSize'] = (100, 100)
""" Size of the coin used."""
active_configs['coinPos'] = {imL: cS for imL, cS in zip(active_configs.imgLocation, ['heads', 'tails'] * 2)} # Coin is
""" Position on screen of coin. """

active_configs['textPos'] = CENTER_POS
"""  Position of text. """
active_configs['textHeight'] = TEXT_HEIGHT
""" Height of text. """

active_configs['timerPos'] = (CENTER_POS[0], -25)
""" Size of the timer"""

# Timings
active_configs['waitTR'] = 1
""" How many TRs to wait before the experiment begins"""

active_configs['timeResponse'] = 2.5 # Response Window
""" Maximal response time """
active_configs['timeSideHighlight'] = 1.0 # Time after fractals are removed
""" Time where theres only the selected fractal on screen. """
active_configs['timeCoinToss'] = 0.75 # Time after Coin appears
""" Time where the coin is on the screen. """
active_configs['timeFractalSelection'] = 0.5 # Time after last fractal disappeared.
""" Time where only the fractal is seen on the screen (i.e. after coin toss) """
active_configs['timeNoResponse'] = 1.25  # Time where only the worst fractal is present.
""" Time for a non-response trial"""
active_configs['timeWealthUpdate'] = 1.0 # Time the wealth takes to roll up.
""" Time the wealth takes to roll up or down. """
active_configs['timeFinalDisplay'] = 1.0 # Time after wealth update
""" How long the final display is shown (i.e. wealth and fractal before reset) """

def check_attribute_present(config_dict:Dict, key_val:str,
                    default_val=None) -> Dict:
    """Checks attribute in dictionary, sets to a default value, if specified,
    otherwise raises error.

    Args:
        config_dict (Dict): Dictionary (often expInfo) containing configuration.
        key_val (str): The key to test.
        default_val (_type_, optional): Default value. Defaults to None.

    Raises:
        KeyError: Error if field is not specified.

    Returns:
        Dict: Returns modified modified config_dict.
    """

    try:
        config_dict[key_val]
    except KeyError:
        if default_val is not None:
            print(f'Did not find a value for {key_val},'
                f' setting to default: {default_val}')
            config_dict[key_val] = default_val
        else:
            raise KeyError(f'{key_val} is not optional!')

    return config_dict


def check_attribute_type(config_dict:Dict, key_val:str, test_type=None) -> None:
    """Test if attribute has the correct type.

    Args:
        config_dict (Dict): Configuration dict.
        key_val (str): The key to check.
        test_type (_type_, optional): The type the attribute should have.
            Defaults to None.

    Raises:
        ValueError: Raises error, if attribute has incorrect type.
    """

    if test_type is None:
        pass

    if not isinstance(config_dict[key_val], test_type):
        raise ValueError(f"{key_val} has to be of type: {test_type}")


def check_configs(config_dict:Dict, task='passive') -> Dict:
    """Checks configs, adds default values.

    Args:
        config_dict (Dict): Configs for the experiment.
        task (str, optional): The task to check for. Defaults to 'passive'.

    Returns:
        Dict: Changed config.
    """

    default_dict = {'participant': None, 'eta': None, 'run': None,
                    'responseLeft': 'left', 'responseRight': 'right',
                    'wealth': con.X0,  'overwrite': True, 'agentActive': False,
                    'simulateMR': 'Simulate', 'TR': 2.0, 'maxTrial': np.inf,
                    'maxDuration': np.inf}

    if task =='passive':
        default_dict.update({'nTrial_noBrainer': 10, 'responseButton': 'space'})
    elif task == 'config':
        default_dict.update({'responseUp': 'up', 'responseDown': 'down'})

    type_dict_options = {
        'participant': str, 'eta': float, 'run': int,
        'responseLeft': str, 'responseRight': str, 'responseButtion': str,
        'wealth': float, 'overwrite': bool, 'agentActive': bool, 'simulateMR': str,
        'TR': float, 'maxTrial': [int, float], 'maxDuration': [int, float]}

    for dc in default_dict.keys():
        config_dict = check_attribute_present(config_dict, dc, default_dict[dc])

    for cd in config_dict.keys():
        if cd in [type_dict_options.keys()]:
            check_attribute_type(config_dict, cd, type_dict_options[cd])

    return config_dict


def get_labels(config_dict:Dict) -> Dict:
    """NOT IMPLEMENTED

    Args:
        config_dict (Dict): _description_

    Returns:
        Dict: _description_
    """

    label_dict_options = {
        'participant': 'The participant ID.',
        'eta': 'The dynamic of the experiment.'}

    labels = {}
    return labels