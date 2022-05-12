import numpy as np
import pandas as pd
import math
from . import constants as con
from .utils import (isoelastic_utility, inverse_isoelastic_utility, calculate_growth_rates,
                    shuffle_along_axis, create_gambles, create_gamble_pairs, create_experiment,
                    create_trial_order)

def passive_sequence_v1(lambd:float,
                        repeats:int,
                        x_0:int,
                        indifference_etas:np.array,
                        indifference_x_0:np.array,
                        indifference_dx2:int):

    gamma_range, gamma1_list, gamma2_list, fractal_dict= calculate_growth_rates(indifference_etas=indifference_etas,
                                                                                lambd=lambd,
                                                                                dx2=indifference_dx2,
                                                                                x=indifference_x_0)
    gamma_0 = isoelastic_utility(x_0,lambd)

    n_fractals = len(gamma_range)
    n_trials = n_fractals * repeats
    fractals = np.random.randint(n_fractals, size=n_trials)
    gamma_array = [gamma_range[fractal] for fractal in fractals]
    part_sum =  gamma_0 + np.cumsum(gamma_array)
    part_wealth_sum = inverse_isoelastic_utility(part_sum,lambd)

    return fractals, gamma_array, part_sum, part_wealth_sum,gamma1_list, gamma2_list, fractal_dict


def passive_sequence_v2(lambd:float,
                        repeats:int,
                        x_0:int,
                        indifference_etas:np.array,
                        indifference_x_0:np.array,
                        indifference_dx2:int):

    gamma_range,gamma1_list,gamma2_list,fractal_dict = calculate_growth_rates(indifference_etas, lambd, indifference_dx2, indifference_x_0)
    gamma_range.append(999) #blank fractal for resetting wealth
    n_fractals = len(gamma_range)

    fractals = []
    part_sum = []

    for x0 in [x_0, x_0 * 10]:

        gamma_0 = isoelastic_utility(x0,lambd)

        fractal_order = shuffle_along_axis(np.array(range(n_fractals-1)), 0)

        for fractal in fractal_order:

            part_sum.extend(gamma_0) #Reset wealth
            fractals.append(n_fractals-1) #Show blank fractal

            tmp_seq = [fractal] * repeats
            fractals.extend(tmp_seq)
            tmp_cum_sum = gamma_0 + np.cumsum([gamma_range[fractal]]*repeats)
            part_sum.extend(tmp_cum_sum)

    gamma_array = np.array([gamma_range[fractal] for fractal in fractals])
    fractals = np.array(fractals)
    fractals[gamma_array == 999] = 9

    part_wealth_sum = inverse_isoelastic_utility(np.array(part_sum),lambd)

    return fractals, gamma_array, part_sum, part_wealth_sum, gamma1_list, gamma2_list, fractal_dict

def active_sequence(n_trials:int,
                    gamma1_list:np.array,
                    gamma2_list:np.array,
                    fractal_dict:dict,
                    n_simulations:int=1):

    gambles = create_gambles(gamma1_list, gamma2_list)
    gambles = shuffle_along_axis(gambles,1)
    gamble_pairs = create_gamble_pairs(gambles)
    experiment  = create_experiment(gamble_pairs)
    trial_order = create_trial_order(
            n_simulations=n_simulations,
            n_gamble_pairs=experiment.shape[-1],
            n_trials=n_trials
        )

    gamma_array = np.empty([n_trials, 4], dtype=float)
    fractals =  np.empty([n_trials, 4], dtype=float)
    coin_toss = shuffle_along_axis(np.concatenate((np.zeros([math.ceil(n_trials/2), 1], dtype=int),
                                                   np.ones([math.ceil(n_trials/2), 1], dtype=int)),
                                                   axis=0)[: n_trials], 0)
    timings = np.empty([n_trials, 3], dtype=float)
    timings[:, 0] = np.zeros(n_trials) + 3 # ITI
    timings[:, 1] = np.zeros(n_trials) + 1.5 # Onset Gamble 1
    timings[:, 2] = np.zeros(n_trials) + 1.5 # Onset Gamble 2
    timings = shuffle_along_axis(timings, 0)

    for ii, trial in enumerate(trial_order):
        tmp = experiment[:,:,trial].flatten()
        fractals[ii, :] =  [fractal_dict[g] for g in tmp]
        gamma_array[ii, :] = tmp

    return fractals, gamma_array, coin_toss, timings, fractal_dict


def generate_dataframes(lambd:float,
                        x_0:int,
                        n_trials_active:int,
                        n_repeats_passive:int,
                        passive_mode:int = 1,
                        speed_up:float = 1,
                        indifference_etas:np.array = con.INDIFFERENCE_ETAS,
                        indiffrence_x_0:np.array = con.INDIFFERENCE_X_0,
                        indifference_dx2:int = con.INDIFFERENCE_DX2):

    if passive_mode == 1:
        passive_sequence = passive_sequence_v1
    elif passive_mode == 2:
        passive_sequence = passive_sequence_v2
    else:
        raise ValueError("Passive sequence has to be 1 or 2")

    (p_seq_fractals, p_seq_gamma,
    p_seq_part_sum, p_seq_part_wealth_sum,
    gamma1_list, gamma2_list, fractal_dict) = passive_sequence(lambd=lambd,
                                                            repeats=n_repeats_passive,
                                                            x_0=x_0,
                                                            indifference_etas=indifference_etas,
                                                            indifference_x_0=indiffrence_x_0,
                                                            indifference_dx2=indifference_dx2)

    n_trials_passive = len(p_seq_fractals)
    p_df = pd.DataFrame(data={'trial': range(n_trials_passive),
                              'lambda': [lambd] * n_trials_passive,
                              'gamma': p_seq_gamma,
                              'fractal': p_seq_fractals,
                              'iti': np.zeros(n_trials_passive) + 3 / speed_up, # to debug
                              'fractal_duration': np.zeros(n_trials_passive) + 1.5 / speed_up, # to debug
                              'p_seq_gamma': p_seq_part_sum,
                              'p_seq_wealth':p_seq_part_wealth_sum})

    (a_seq_fractals, a_seq_gamma,
     a_seq_cointoss, a_seq_timings, _) = active_sequence(n_trials=n_trials_active,
                                                         gamma1_list=gamma1_list,
                                                         gamma2_list=gamma2_list,
                                                         fractal_dict=fractal_dict)

    a_df_fractals = pd.DataFrame(a_seq_fractals,
                                 columns=['fractal_left_up', 'fractal_left_down',
                                          'fractal_right_up', 'fractal_right_down'])
    a_df_gamma = pd.DataFrame(a_seq_gamma,
                              columns=['gamma_left_up', 'gamma_left_down',
                                       'gamma_right_up', 'gamma_right_down'])

    a_df_cointoss = pd.DataFrame(a_seq_cointoss, columns=['gamble_up'])

    a_df_misc = pd.DataFrame(data={'trial': range(n_trials_active),
                                   'lambda': [lambd]*n_trials_active})

    a_df_timings = pd.DataFrame(a_seq_timings / speed_up, columns=['iti',
                                                        'onset_gamble_pair_left',
                                                        'onset_gamble_pair_right'])

    a_df = pd.concat([a_df_misc,a_df_fractals, a_df_gamma,
                      a_df_cointoss,a_df_timings], axis=1)

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
            + f"Sequence generated using passive version {passive_mode} \n"
            + f"Trials: {n_trials_passive}\nmin: {min(p_seq_part_wealth_sum)}\nmax: {max(p_seq_part_wealth_sum)}"
            + "\n\n\nActive: \n______________________ \n"
            + f"n. trials: {n_trials_active} \n"
            + f"Left upper avg: {np.mean(l_avg_u)}\n"
            + f"Left lower avg: {np.mean(l_avg_l)}\n"
            + f"Right upper avg: {np.mean(r_avg_u)}\n"
            + f"Right lower avg: {np.mean(r_avg_l)}\n"
            + f"Time Optimal Max: {max_sum_gamma}\n"
            + f"Cointoss: {np.mean(a_df_cointoss.mean(axis=0))}\n" )

    return p_df, a_df, meta
