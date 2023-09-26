import numpy as np

# Trial numbers
n_trials_passive = 45 * 3
""" How often fractals are shown in the Passive Phase (defines trials as N_REPEATS_PASSIVE * N_FRACTALS"""
n_trials_active = 160
""" Number of trials in the active phase"""
n_trials_nobrainer = 15
""" Number of nobrainer trials after the passive phase ends."""

# Resets (restarts of experiment)
max_run_passive = 3
""" Number of runs of the passive phase"""
max_run_active = 1
""" Number of runs in the active phase"""

# After how many trials to start no-brainers
start_nobrainer = 45
"""Starts N_TRIALS_NOBRAINER after np.mod(passive, START_NOBRAINER) == 0 Trials """

# How many trials in each run
max_trials_passive = 45
""" Number of trials per run in the passive phase. """
max_trials_active = np.inf
""" Number of trials per run in he active phase. """

# What slides to show
SLIDESET = [0, 14, 15, 26]
""" The 2 sets of start and stop slides for the instructions depending on mode. """

# For MR test: the TR
TR = 2.303
""" TR of the MR scanner (also for simulations) """

# Gamble creation
x_0 = 1000
c_dict = {0.0: 450,
          1.0: 0.806}
assymetry_dict = {0.0: [43, 32, -16, -32, -49, -4, -15, -28, -10],
                   1.0: [-0.044, 0.065, -0.030, 0.028, 0.006, -0.033, -0.034, -0.051, -0.034]}
LIMITS = {0.0: [-500, 2_500], 1.0: [64 , 15_589]}

# Number of fractals we are dealing with
N_FRACTALS = 9

passive_iti_mu = 2.0
passive_iti_sd = 0
