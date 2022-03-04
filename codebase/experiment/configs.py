from ..utils import DotDict

STIMULUSPATH = 'data/stimuli/'

passive_configs = DotDict()
passive_configs['imgPos'] = (0.0, 0.0) # Position of the center image
passive_configs['imgSize'] = (425 * 2, 319 * 2) # Size of the center image
passive_configs['textPos'] = (0, 0) # Position of all texts
passive_configs['textHeight'] = 20 # Height of text

passive_configs['boxPos'] = (0,0) # Position of box around money
passive_configs['boxSize'] = (180, 30)
passive_configs['boxWidth'] = 3

passive_configs['wheelPos'] = (0, 0) # Position of the wheel
passive_configs['wheelSize'] = (300, 300) # Size of the wheel image
passive_configs['revolution'] = 5 # Rotation angle of wheel

passive_configs['stopperSize'] = (30, 30) # Positon of blue triangle
passive_configs['stopperPos'] = (0, 160)

# TODO: Possible redefine or make variable as well.
passive_configs['imagePath'] = STIMULUSPATH
passive_configs['fractalList'] = ['F000', 'F001', 'F002', 'F003', 'F004',
               'F005', 'F006', 'F007', 'F008']

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

active_configs['imgSize'] = (425, 319) # Image size

active_configs['coinSize'] = (100, 100)
active_configs['coinPos'] = {imL: cS for imL, cS in zip(active_configs.imgLocation, ['heads', 'tails'] * 2)} # Coin is

active_configs['textPos'] = (0, 0)
active_configs['textHeight'] = 20

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
active_configs['fractalList'] = ['F000', 'F001', 'F002', 'F003', 'F004',
            'F005', 'F006', 'F007', 'F008']