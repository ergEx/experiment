from cgi import test
from distutils.command.config import config
from tkinter import CENTER
from typing import Dict
from ..utils import DotDict

STIMULUSPATH = 'data/stimuli/'
IMGSIZE = (425, 319)
DEFAULT_FRACTALS = ['F000', 'F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008']
CENTER_POS = (0, 0)
TEXT_HEIGHT = 20

passive_configs = DotDict()

passive_configs['centerPos'] = CENTER_POS
passive_configs['imgSize'] = (IMGSIZE[0] * 2, IMGSIZE[1] * 2) # Size of the center image
passive_configs['textHeight'] = TEXT_HEIGHT # Height of text

passive_configs['boxSize'] = (180, 30)
passive_configs['boxWidth'] = 3

passive_configs['wheelSize'] = (300, 300) # Size of the wheel image
passive_configs['revolution'] = 5 # Rotation angle of wheel

passive_configs['stopperSize'] = (30, 30) # Positon of blue triangle
passive_configs['stopperPos'] = (0, 160)

# TODO: Possible redefine or make variable as well.
passive_configs['imagePath'] = STIMULUSPATH
passive_configs['fractalList'] = DEFAULT_FRACTALS

# Timings
passive_configs['waitTR'] = 1 # How many TRs to wait
passive_configs['timeToReminder'] = 1.0 # How much time until reminder is shown
passive_configs['timeResponseWindow'] = 2.0 # time of the response window
passive_configs['wheelSpinTime'] = 2 # How long the wheel spins at base rate
passive_configs['timeWealthUpdate'] = 1.0 # Rolling of the wealth display
passive_configs['timeFinalDisplay'] = 1.0 # How long wealth image is staying on


############################### Active settings

active_configs = DotDict()
# Image sizes
imgX = 312 # Img, coin, and frame position.
imgY = 200

active_configs['imgPos'] = [(-imgX, imgY), (-imgX, -imgY), (imgX, imgY), (imgX, -imgY)] # Pos list
active_configs['imgLocation'] = ['leftUp', 'leftDown', 'rightUp', 'rightDown']
active_configs['imgLocPos'] = {imL: imP for imL, imP in zip(active_configs.imgLocation, active_configs.imgPos)} # Dict for pos

active_configs['imgSize'] = IMGSIZE

active_configs['coinSize'] = (100, 100)
active_configs['coinPos'] = {imL: cS for imL, cS in zip(active_configs.imgLocation, ['heads', 'tails'] * 2)} # Coin is

active_configs['textPos'] = CENTER_POS
active_configs['textHeight'] = TEXT_HEIGHT

# Timings
active_configs['waitTR'] = 1
active_configs['timeResponse'] = 2.5 # Response Window
active_configs['timeCursorEvent'] = 0 # time of highlight
active_configs['timeSideHighlight'] = 1.0 # Time after fractals are removed
active_configs['timeCoinToss'] = 0.75 # Time after Coin appears
active_configs['timeFractalSelection'] = 0.5 # Time after last fractal disappeared.
active_configs['timeWealthUpdate'] = 1.0 # Time the wealth takes to roll up.
active_configs['timeNoResponse'] = 1.25  # Time where only the worst fractal is present.
active_configs['timeFinalDisplay'] = 1.0 # Time after wealth update

active_configs['imagePath'] = STIMULUSPATH
active_configs['fractalList'] = DEFAULT_FRACTALS


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


