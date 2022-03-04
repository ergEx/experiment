import math
import numpy as np
import pandas as pd
import pathlib
import os
from . import constants as con
from .utils import isoelastic_utility,inverse_isoelastic_utility, shuffle_along_axis, plot_sequence, calculate_dx1
from .utils import create_experiment, create_trial_order
from .utils import create_gambles, create_gamble_pairs, create_gambles_v2, create_gamble_pairs_v2
from .utils import is_g_deterministic, is_nobrainer, is_statewise_dominated, is_stochastically_dominated
from .utils import growth_factor_to_fractal, random_reorder_axis


def passive_sequence_v1(eta:float, c:float, repeats:int, active_mode:int, n_fractals:int=con.N_FRACTALS,
                     x_0:int=con.X0, x_lower:int=0.001, x_upper:int=10**9, debug:bool=False, bound=False):
    if active_mode == 1:
        gamma_range = np.linspace(-c, c, n_fractals)
    elif active_mode == 2:
        dx1_list = [calculate_dx1(eta,-100,1000) for eta in [-2.0, -1.5, -0.5, -0.25, 0.5, 1.5, 2.5]]
        gamma_range = [float(isoelastic_utility(1000+dx1,eta)-isoelastic_utility(1000,eta)) for dx1 in dx1_list]
        gamma_range.append(float(isoelastic_utility(1000-100,eta)-isoelastic_utility(1000,eta)))
        gamma_range.append(0)
    else:
        raise ValueError('Only v1 and v2 are supported in the active phase')

    gamma_range = np.array(gamma_range)
    gamma_lower = isoelastic_utility(x_lower, eta) #- isoelastic_utility(x_0,eta)
    gamma_upper = isoelastic_utility(x_upper, eta) #- isoelastic_utility(x_0,eta)
    gamma_0     = isoelastic_utility(x_0, eta)

    accepted = False
    n_rejected = 0
    idx = 0
    while not accepted:
        idx += 1

        n_trials = n_fractals * repeats #Might change this to be an input instad of 'repeats'
        fractals = np.random.randint(n_fractals, size=n_trials)
        gamma_array = gamma_range[fractals]
        part_sum =  gamma_0 + np.cumsum(gamma_range[fractals])
        part_wealth_sum = inverse_isoelastic_utility(part_sum,eta)
        if bound:
            if  all(i > gamma_lower for i in part_sum)  \
            and all(i < gamma_upper for i in part_sum):
                accepted = True
                #print(f"Sequence n. {idx} was accepted!")
            else:
                n_rejected += 1
        else:
            accepted = True

        if debug:
            plot_sequence(part_sum, idx, gamma_lower, gamma_upper,
                            x_lower, x_upper, eta, x_0)

    return fractals, gamma_array, part_sum, part_wealth_sum, gamma_lower, gamma_upper


def passive_sequence_v2(eta:float, c:float, repeats:int,
                        active_mode:int=1,
                        n_fractals:int = con.N_FRACTALS, x_0:int = con.X0,
                        x_lower:int = 0.001, x_upper:int = 10**9,
                        debug:bool = False, bound = True):

    if active_mode == 1:
        gamma_range = np.linspace(-c, c, n_fractals).tolist()
        gamma_range.append(999)
    elif active_mode == 2:
        dx1_list = [calculate_dx1(eta,-100,1000) for eta in [-2.0, -1.5, -0.5, -0.25 , 0.5, 1.5, 2.5]]
        gamma_range = [float(isoelastic_utility(1000+dx1,eta)-isoelastic_utility(1000,eta)) for dx1 in dx1_list]
        gamma_range.append(float(isoelastic_utility(1000-100,eta)-isoelastic_utility(1000,eta)))
        gamma_range.append(0)
        gamma_range.append(999)
    else:
        raise ValueError('Only v1 and v2 are supported in the active phase')

    gamma_range = np.array(gamma_range)
    gamma_lower = isoelastic_utility(x_lower, eta) #- isoelastic_utility(x_0,eta)
    gamma_upper = isoelastic_utility(x_upper, eta) #- isoelastic_utility(x_0,eta)

    fractals = []
    part_sum = []

    for x0 in [x_0, x_0 / 10,  x_0 * 10]:

        gamma_0 = isoelastic_utility(x0, eta)

        # if ((gamma_0 + repeats * c) > gamma_upper
        #   or (gamma_0 + repeats * (-c)) < gamma_lower):
        #    raise ValueError('Number of repeats is too high for given c-value')

        fractal_order = shuffle_along_axis(np.array(range(n_fractals)), 0) if active_mode == 1 else shuffle_along_axis(np.array(range(n_fractals)), 0)

        for fractal in fractal_order:
            tmp_seq = [fractal] * repeats
            fractals.extend(tmp_seq)
            tmp_cum_sum = gamma_0 + np.cumsum(gamma_range[tmp_seq])
            part_sum.extend(tmp_cum_sum)
            part_sum.extend(gamma_0) #Reset wealth

            fractals.append(n_fractals) #Show blank fractal

    gamma_array = gamma_range[fractals]
    part_wealth_sum = inverse_isoelastic_utility(np.array(part_sum),eta)

    if debug:
        plot_sequence(part_sum, 1, gamma_lower, gamma_upper,
                        x_lower, x_upper, eta, x_0)

    return fractals, gamma_array, part_sum, part_wealth_sum, gamma_lower, gamma_upper


def active_sequence(c:float,eta:float, n_trials:int, n_fractals:int=con.N_FRACTALS,
                    simulation:bool = False, n_simulations:int = 1, pilot_mode:int = 1):

    if pilot_mode == 1: #Original version (symetric growth rates)
        gambles, fractal_dict = create_gambles(c,n_fractals)
        gambles = [
            gamble for gamble in gambles
            if not is_g_deterministic(gamble)
            ]
        gamble_pairs = create_gamble_pairs(gambles)
        gamble_pairs = [
            gamble_pair for gamble_pair in gamble_pairs
            if not is_statewise_dominated(gamble_pair)
            and not is_nobrainer(gamble_pair)
            ]
    elif pilot_mode == 2: #Non parametric approach
        gambles,fractal_dict = create_gambles_v2(etas=[-2.0, -1.5, -0.5, -0.25, 0.5, 1.5, 2.5],dynamic=eta,dx2=-100,x=1000)
        gamble_pairs = create_gamble_pairs_v2(gambles)
    else:
        raise ValueError('Only v1 and v2 are supported in the active phase')

    # gamble_pairs = shuffle_along_axis(np.array(gamble_pairs),axis=2) #Shuffeling growth_rates within gambles
    # gamble_pairs = random_reorder_axis(np.array(gamble_pairs),axis=1) #Shuffeling gambles with gamble_pairs

    experiment  = create_experiment(gamble_pairs)

    trial_order = create_trial_order(
            n_simulations=1,
            n_gamble_pairs=experiment.shape[-1],
            n_trials=n_trials
        )

    gamma_array = np.empty([n_trials, 4], dtype=float)
    fractals =  np.empty([n_trials, 4], dtype=float)
    coin_toss = shuffle_along_axis(np.concatenate((np.zeros([math.ceil(n_trials/2), 1], dtype=int),
                                                   np.ones([math.ceil(n_trials/2), 1], dtype=int)),
                                                   axis=0)[: n_trials], 0)
    flags = np.empty([n_trials, 2], dtype=int)
    timings = np.empty([n_trials, 3], dtype=float)
    timings[:, 0] = np.zeros(n_trials) + 3 # ITI
    timings[:, 1] = np.zeros(n_trials) + 1.5 # Onset Gamble 1
    timings[:, 2] = np.zeros(n_trials) + 1.5 # Onset Gamble 2
    timings = shuffle_along_axis(timings, 0)

    for ii, trial in enumerate(trial_order):
        tmp = experiment[:,:,trial].flatten()


        fractals[ii, :] = [growth_factor_to_fractal(g, c, n_fractals) for g in tmp] if pilot_mode == 1 else [fractal_dict[g] for g in tmp]

        gamma_array[ii, :] = tmp
        flags[ii, 0] = is_nobrainer(gamble_pairs[trial_order[ii][0]])
        flags[ii, 1] = is_stochastically_dominated(gamble_pairs[trial_order[ii][0]])

    return fractals, gamma_array, coin_toss, flags, timings, fractal_dict


def generate_dataframes(eta:float,
                        c:float,
                        n_trials_active:int,
                        n_repeats_passive:int,
                        passive_mode:int = 1,
                        active_mode:int = 1,
                        speed_up:float = 1):

    if passive_mode == 1:
        passive_sequence = passive_sequence_v1
    elif passive_mode == 2:
        passive_sequence = passive_sequence_v2
    else:
        raise ValueError("Passive sequence has to be 1 or 2")

    #Do everything for the active phase
    (a_seq_fractals, a_seq_gamma,
     a_seq_cointoss, a_seq_flags, a_seq_timings, _) = active_sequence(c=c,
                                                                   eta=eta,
                                                                   n_trials=n_trials_active,
                                                                   pilot_mode=active_mode)

    a_df_fractals = pd.DataFrame(a_seq_fractals,
                                 columns=['fractal_left_up', 'fractal_left_down',
                                          'fractal_right_up', 'fractal_right_down'])
    a_df_gamma = pd.DataFrame(a_seq_gamma,
                              columns=['gamma_left_up', 'gamma_left_down',
                                       'gamma_right_up', 'gamma_right_down'])

    a_df_cointoss = pd.DataFrame(a_seq_cointoss, columns=['gamble_up'])

    a_df_flags = pd.DataFrame(a_seq_flags, columns=['no_brainer',
                                                    'stochasticly_dominated'])

    a_df_misc = pd.DataFrame(data={'trial': range(n_trials_active),
                                   'eta': [eta]*n_trials_active})

    a_df_timings = pd.DataFrame(a_seq_timings / speed_up, columns=['iti',
                                                        'onset_gamble_pair_left',
                                                        'onset_gamble_pair_right'])

    a_df = pd.concat([a_df_misc,a_df_fractals, a_df_gamma, a_df_cointoss,
                      a_df_flags, a_df_timings], axis=1)

    p_seq_fractals, seq_gamma, p_seq_gamma, p_seq_wealth, _, _ = passive_sequence(eta=eta,
                                                    c=c,
                                                    repeats=n_repeats_passive,
                                                    active_mode= active_mode)
    # Calculate number of trials:
    n_trials_passive = len(p_seq_fractals)
    p_df = pd.DataFrame(data={'trial': range(n_trials_passive),
                            'eta': [eta] * n_trials_passive,
                            'gamma': seq_gamma,
                            'fractal': p_seq_fractals,
                            'iti': np.zeros(n_trials_passive) + 3 / speed_up, # to debug
                            'fractal_duration': np.zeros(n_trials_passive) + 1.5 / speed_up, # to debug
                            'p_seq_gamma': p_seq_gamma,
                            'p_seq_wealth':p_seq_wealth})

    ## Meta Info written here
    l_avg_u = a_df_gamma[['gamma_left_up']].sum(axis=0)
    l_avg_l = a_df_gamma[['gamma_left_down']].sum(axis=0)
    r_avg_u = a_df_gamma[['gamma_right_up']].sum(axis=0)
    r_avg_l = a_df_gamma[['gamma_right_down']].sum(axis=0)

    optimal_path = (a_df_gamma[['gamma_left_up', 'gamma_left_down']].mean(1) <
                    a_df_gamma[['gamma_right_up', 'gamma_right_up']].mean(1)) * 1 # to cast

    max_gammas = np.concatenate([a_df_gamma[['gamma_left_up', 'gamma_left_down']].max(1).values[:, np.newaxis],
                                a_df_gamma[['gamma_right_up', 'gamma_right_down']].max(1).values[:, np.newaxis]],
                                axis=1)
    max_sum_gamma = np.take_along_axis(max_gammas, optimal_path.values[:, np.newaxis] * 1,
                                    axis=1).sum()

    meta = ("Passive: \n______________________ \n"
            + f"Sequence generated using passive version {passive_mode}"
            + f" and active version {active_mode} \n"
            + f"Trials: {n_trials_passive}\nmin: {min(p_seq_wealth)}\nmax: {max(p_seq_wealth)}"
            + "\n\n\nActive: \n______________________ \n"
            + f"n. trials: {n_trials_active} \n"
            + f"Left upper avg: {np.mean(l_avg_u)}\n"
            +  f"Left lower avg: {np.mean(l_avg_l)}\n"
            +  f"Right upper avg: {np.mean(r_avg_u)}\n"
            +  f"Right lower avg: {np.mean(r_avg_l)}\n"
            +  f"Time Optimal Max: {max_sum_gamma}\n"
            +  f"Cointoss: {np.mean(a_df_cointoss.mean(axis=0))}\n"
            +  f"No brainers: {np.mean(a_df_flags['no_brainer'].sum(axis=0))}\n"
            +  f"Stochastically dominated: {np.mean(a_df_flags['stochasticly_dominated'].sum(axis=0))}\n" )

    return p_df, a_df, meta
