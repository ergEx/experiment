#Both modes
x_0 = 1000
n_trials_passive = 45
n_trials_active=120
n_resets_passive = 4

#Mode 1 (one gamble)
indifference_etas = {0.0: [-1, 0.5, 2],
                     1.0: [-1, 0.5, 2]}
indifference_x_0 = [1000, 5000]
indifference_dx2 = -250

#Mode 2 (two gambles)
c_dict = {0.0: 428,
          1.0: 0.806}
assymetry_dict = {0.0: [43, 32, -16, -32, -49, -4, -15, -28, -10],
                   1.0: [-0.044, 0.065, -0.030, 0.028, 0.006, -0.033, -0.034, -0.051, -0.034]}

N_FRACTALS = 9

OUTPUT_EXTENSION = 'beh.csv'
