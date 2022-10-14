import math
import numpy as np
import pandas as pd
from . import constants as con
from .utils import calculate_growth_rates, create_experiment, create_gamble_pairs_one_gamble, \
    create_gamble_pairs_two_gambles, create_gambles_one_gamble, create_gambles_two_gambles, create_trial_order, \
    indiference_eta, inverse_isoelastic_utility, is_g_deterministic, is_nobrainer, is_statewise_dominated, \
    isoelastic_utility, shuffle_along_axis, wealth_change


def passive_sequence_one_gamble(lambd:float,
                        n_trials_before_reset:int,
                        n_resets:int,
                        x_0:int,
                        indifference_etas:np.array,
                        indifference_x_0:np.array,
                        indifference_dx2:int):

    gamma_range, gamma1_list, gamma2_list, fractal_dict= calculate_growth_rates(indifference_etas=indifference_etas,
                                                                                lambd=lambd,
                                                                                dx2=indifference_dx2,
                                                                                x=indifference_x_0)

    gamma_range = np.array(gamma_range)
    gamma_0 = isoelastic_utility(x_0,lambd)

    n_fractals = len(gamma_range)
    fractals = np.random.randint(n_fractals, size=n_trials_before_reset*n_resets)
    gamma_array = [gamma_range[fractal] for fractal in fractals]
    part_sum = np.zeros(n_trials_before_reset*n_resets)
    for resets in range(n_resets):
        index_1 = n_trials_before_reset*resets
        index_2 = n_trials_before_reset*(resets+1)
        part_sum[index_1:index_2] =  gamma_0 + np.cumsum(gamma_range[fractals[index_1:index_2]])
    part_wealth_sum = inverse_isoelastic_utility(part_sum,lambd)

    return fractals, gamma_array, part_sum, part_wealth_sum,gamma1_list, gamma2_list, fractal_dict

def passive_sequence_two_gambles(lambd:float,
                        c_dict:dict,
                        assymetry_dict:dict,
                        n_trials_before_reset:int,
                        n_resets:int,
                        n_fractals:int,
                        x_0:int):
    c = c_dict[lambd]
    gamma_range = np.array(np.linspace(-c, c, n_fractals))
    assymetry_array = assymetry_dict[lambd]
    gamma_range = np.sum([gamma_range,assymetry_array], axis=0)
    fractal_dict = {ii : n for n, ii in enumerate(gamma_range)}
    gamma_0     = isoelastic_utility(x_0, lambd)

    fractals = np.random.randint(n_fractals, size=n_trials_before_reset*n_resets)
    gamma_array = gamma_range[fractals]
    part_sum = np.zeros(n_trials_before_reset*n_resets)
    for resets in range(n_resets):
        index_1 = n_trials_before_reset*resets
        index_2 = n_trials_before_reset*(resets+1)
        part_sum[index_1:index_2] =  gamma_0 + np.cumsum(gamma_range[fractals[index_1:index_2]])
    part_wealth_sum = inverse_isoelastic_utility(part_sum,lambd)

    return fractals, gamma_array, part_sum, part_wealth_sum, fractal_dict, gamma_range

def active_sequence_one_gamble(n_trials:int,
                    gamma1_list:np.array,
                    gamma2_list:np.array,
                    fractal_dict:dict,
                    n_simulations:int=1):

    gambles = create_gambles_one_gamble(gamma1_list, gamma2_list)
    gambles = shuffle_along_axis(gambles,1)
    gamble_pairs = create_gamble_pairs_one_gamble(gambles)
    experiment  = create_experiment(gamble_pairs)
    trial_order = create_trial_order(
            n_simulations=n_simulations,
            n_gamble_pairs=experiment.shape[-1],
            n_trials=n_trials
        )

    gamma_array = np.empty([n_trials, 4], dtype=float)
    fractals =  np.empty([n_trials, 4], dtype=float)
    coin_toss = np.random.randint(2, size=(n_trials,1))

    timings = np.empty([n_trials, 3], dtype=float)
    timings[:, 0] = np.zeros(n_trials) + 2.0 # ITI
    timings[:, 1] = np.zeros(n_trials) + 1.5 # Onset Gamble 1
    timings[:, 2] = np.zeros(n_trials) + 1.5 # Onset Gamble 2
    timings = shuffle_along_axis(timings, 0)

    for ii, trial in enumerate(trial_order):
        tmp = experiment[:,:,trial].flatten()
        fractals[ii, :] =  [fractal_dict[g] for g in tmp]
        gamma_array[ii, :] = tmp

    return fractals, gamma_array, coin_toss, timings, fractal_dict


def active_sequence_two_gambles(lambd:float,
                                n_trials:int,
                                gamma_range:np.array,
                                fractal_dict:dict,
                                filtering:bool=False,
                                n_simulations:int=1):

    gambles = create_gambles_two_gambles(gamma_range)
    gambles = [
        gamble for gamble in gambles
        if not is_g_deterministic(gamble)
        ]
    gambles = shuffle_along_axis(np.array(gambles),1)
    gamble_pairs = create_gamble_pairs_two_gambles(gambles)
    gamble_pairs = [
        gamble_pair for gamble_pair in gamble_pairs
        if not is_statewise_dominated(gamble_pair)
        and not is_nobrainer(gamble_pair)
        ]

    if filtering:
        tmp = list()
        for gamble_pair in gamble_pairs:
            x_updates = wealth_change(x=1000,
                                      gamma=[gamble_pair[0][0], gamble_pair[0][1],
                                            gamble_pair[1][0], gamble_pair[1][1]],
                                            lambd=lambd)
            try:
                root = indiference_eta(x_updates[0], x_updates[1], x_updates[2], x_updates[3])
                if -0.5 < root < 1.5:
                    tmp.append(gamble_pair)
            except Exception:
                pass

        gamble_pairs = tmp

    experiment  = create_experiment(gamble_pairs)
    trial_order = create_trial_order(
            n_simulations=n_simulations,
            n_gamble_pairs=experiment.shape[-1],
            n_trials=n_trials
        )

    gamma_array = np.empty([n_trials, 4], dtype=float)
    fractals =  np.empty([n_trials, 4], dtype=float)
    coin_toss = np.random.randint(2, size=(n_trials,1))

    timings = np.empty([n_trials, 3], dtype=float)
    timings[:, 0] = np.zeros(n_trials) + 2.0 # ITI
    timings[:, 1] = np.zeros(n_trials) + 1.3 # Onset Gamble 1
    timings[:, 2] = np.zeros(n_trials) + 1.3 # Onset Gamble 2
    timings = shuffle_along_axis(timings, 0)

    for ii, trial in enumerate(trial_order):
        tmp = experiment[:,:,trial].flatten()
        fractals[ii, :] =  [fractal_dict[g] for g in tmp]
        gamma_array[ii, :] = tmp

    return fractals, gamma_array, coin_toss, timings, fractal_dict


def active_sequence_two_gambles_train_tracks(lambd:float,
                    n_trials:int,
                    gamma_range:np.array,
                    fractal_dict:dict,
                    n_simulations:int=1,
                    filtering:bool = False):

    gamma_array_train_tracks = dict()
    fractals_train_tracks = dict()

    coin_toss = np.random.randint(2, size=(n_trials,1))
    timings = np.empty([n_trials, 3], dtype=float)
    timings[:, 0] = np.zeros(n_trials) + 2.0 # ITI
    timings[:, 1] = np.zeros(n_trials) + 1.3 # Onset Gamble 1
    timings[:, 2] = np.zeros(n_trials) + 1.3 # Onset Gamble 2
    timings = shuffle_along_axis(timings, 0)

    for i,r in enumerate([list(range(int(len(gamma_range)-2))),list(range(int(len(gamma_range)))),list(range(2,int(len(gamma_range))))]):
        gambles = create_gambles_two_gambles(gamma_range[r])
        gambles = [
            gamble for gamble in gambles
            if not is_g_deterministic(gamble)
            ]
        gambles = shuffle_along_axis(np.array(gambles),1)
        gamble_pairs = create_gamble_pairs_two_gambles(gambles)
        gamble_pairs = [
            gamble_pair for gamble_pair in gamble_pairs
            if not is_statewise_dominated(gamble_pair)
            and not is_nobrainer(gamble_pair)
            ]

        if filtering:
            tmp = list()
            for gamble_pair in gamble_pairs:
                x_updates = wealth_change(x=1000,
                                        gamma=[gamble_pair[0][0], gamble_pair[0][1],
                                                gamble_pair[1][0], gamble_pair[1][1]],
                                                lambd=lambd)
                try:
                    root = indiference_eta(x_updates[0], x_updates[1], x_updates[2], x_updates[3])
                    if -0.5 < root < 1.5:
                        tmp.append(gamble_pair)
                except Exception:
                    pass

            gamble_pairs = tmp

        experiment  = create_experiment(gamble_pairs)
        trial_order = create_trial_order(
                n_simulations=n_simulations,
                n_gamble_pairs=experiment.shape[-1],
                n_trials=n_trials
            )

        gamma_array = np.empty([n_trials, 4], dtype=float)
        fractals =  np.empty([n_trials, 4], dtype=float)

        for ii, trial in enumerate(trial_order):
            tmp = experiment[:,:,trial].flatten()
            fractals[ii, :] =  [fractal_dict[g] for g in tmp]
            gamma_array[ii, :] = tmp

        gamma_array_train_tracks[i] = gamma_array
        fractals_train_tracks[i] = fractals

    return fractals_train_tracks, gamma_array_train_tracks, coin_toss, timings, fractal_dict


def generate_dataframes(lambd:float,
                        mode:int,
                        n_trials_active:int=con.n_trials_active,
                        n_trials_passive_before_reset:int=con.n_trials_passive,
                        n_resets_passive:int=con.n_resets_passive,
                        speed_up:float=1,
                        gamble_filter:bool = False
                        c_dict=con.c_dict,
                        assymetry_dict=con.assymetry_dict
                        ):

    if mode == 3: #Gamble pair version with train tracks
        (p_seq_fractals, p_seq_gamma,
        p_seq_part_sum, p_seq_part_wealth_sum,
        fractal_dict, gamma_range) = passive_sequence_two_gambles(lambd=lambd,
                                                                    c_dict=c_dict,
                                                                    assymetry_dict=assymetry_dict,
                                                                    n_trials_before_reset=n_trials_passive_before_reset,
                                                                    n_resets=n_resets_passive,
                                                                    n_fractals=con.N_FRACTALS,
                                                                    x_0=con.x_0)


        (a_seq_fractals, a_seq_gamma,
        a_seq_cointoss, a_seq_timings, _) = active_sequence_two_gambles_train_tracks(lambd=lambd,
                                                                        n_trials=n_trials_active,
                                                                        gamma_range=gamma_range,
                                                                        fractal_dict=fractal_dict,
                                                                        filtering=gamble_filter)
    elif mode == 2: #One gamble version
        (p_seq_fractals, p_seq_gamma,
        p_seq_part_sum, p_seq_part_wealth_sum,
        gamma1_list, gamma2_list, fractal_dict) = passive_sequence_one_gamble(lambd=lambd,
                                                                    n_trials_before_reset=n_trials_passive_before_reset,
                                                                    n_resets=n_resets_passive,
                                                                    x_0=con.x_0,
                                                                    indifference_etas=con.indifference_etas,
                                                                    indifference_x_0=con.indifference_x_0,
                                                                    indifference_dx2=con.indifference_dx2)

        (a_seq_fractals, a_seq_gamma,
        a_seq_cointoss, a_seq_timings, _) = active_sequence_one_gamble(n_trials=n_trials_active,
                                                            gamma1_list=gamma1_list,
                                                            gamma2_list=gamma2_list,
                                                            fractal_dict=fractal_dict)

    elif mode == 1 or mode == 4: #Gamble pair version
        (p_seq_fractals, p_seq_gamma,
        p_seq_part_sum, p_seq_part_wealth_sum,
        fractal_dict, gamma_range) = passive_sequence_two_gambles(lambd=lambd,
                                                                    c_dict=c_dict,
                                                                    assymetry_dict=assymetry_dict,
                                                                    n_trials_before_reset=n_trials_passive_before_reset,
                                                                    n_resets=n_resets_passive,
                                                                    n_fractals=con.N_FRACTALS,
                                                                    x_0=con.x_0)


        (a_seq_fractals, a_seq_gamma,
        a_seq_cointoss, a_seq_timings, _) = active_sequence_two_gambles(lambd=lambd,
                                                                        n_trials=n_trials_active,
                                                                        gamma_range=gamma_range,
                                                                        fractal_dict=fractal_dict,
                                                                        filtering=gamble_filter)
    else:
        raise ValueError("Mode has to be 1, 2, 3 or 4")


    n_trials_passive = len(p_seq_fractals)
    p_df = pd.DataFrame(data={'trial': range(n_trials_passive),
                              'lambda': [lambd] * n_trials_passive,
                              'gamma': p_seq_gamma,
                              'fractal': p_seq_fractals,
                              'iti': np.zeros(n_trials_passive) + 2.0 / speed_up, # to debug
                              'fractal_duration': np.zeros(n_trials_passive) + 1.5 / speed_up, # to debug
                              'p_seq_gamma': p_seq_part_sum,
                              'p_seq_wealth':p_seq_part_wealth_sum})

    a_df_cointoss = pd.DataFrame(a_seq_cointoss, columns=['gamble_up'])

    a_df_misc = pd.DataFrame(data={'trial': range(n_trials_active),
                                    'lambda': [lambd]*n_trials_active})

    a_df_timings = pd.DataFrame(a_seq_timings / speed_up, columns=['iti',
                                                            'onset_gamble_pair_left',
                                                            'onset_gamble_pair_right'])
    if mode in [1, 2, 4]:
        a_df_fractals = pd.DataFrame(a_seq_fractals,
                                    columns=['fractal_left_up', 'fractal_left_down',
                                            'fractal_right_up', 'fractal_right_down'])
        a_df_gamma = pd.DataFrame(a_seq_gamma,
                                columns=['gamma_left_up', 'gamma_left_down',
                                        'gamma_right_up', 'gamma_right_down'])


        a_df = pd.concat([a_df_misc,a_df_fractals, a_df_gamma,
                        a_df_cointoss,a_df_timings], axis=1)

        ## Meta Info written here
        l_avg_u = a_df_gamma[['gamma_left_up']].mean(axis=0)
        l_avg_l = a_df_gamma[['gamma_left_down']].mean(axis=0)
        r_avg_u = a_df_gamma[['gamma_right_up']].mean(axis=0)
        r_avg_l = a_df_gamma[['gamma_right_down']].mean(axis=0)

        optimal_path = (a_df_gamma[['gamma_left_up', 'gamma_left_down']].mean(1) <
                        a_df_gamma[['gamma_right_up', 'gamma_right_up']].mean(1)) * 1 # to cast

        max_gammas = np.concatenate([a_df_gamma[['gamma_left_up', 'gamma_left_down']].max(1).values[:, np.newaxis],
                                    a_df_gamma[['gamma_right_up', 'gamma_right_down']].max(1).values[:, np.newaxis]],
                                    axis=1)
        max_sum_gamma = np.take_along_axis(max_gammas, optimal_path.values[:, np.newaxis] * 1,
                                        axis=1).sum()

        meta = ("Passive: \n______________________ \n"
                + f"Sequence generated using version {mode} \n"
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

    elif mode == 3:
        a_dfs = []
        for i in range(3):
            a_df_fractals = pd.DataFrame(a_seq_fractals[i],
                                    columns=['fractal_left_up', 'fractal_left_down',
                                            'fractal_right_up', 'fractal_right_down'])
            a_df_gamma = pd.DataFrame(a_seq_gamma[i],
                                columns=['gamma_left_up', 'gamma_left_down',
                                        'gamma_right_up', 'gamma_right_down'])


            a_df = pd.concat([a_df_misc,a_df_fractals, a_df_gamma,
                        a_df_cointoss,a_df_timings], axis=1)
            a_dfs.append(a_df)

        return p_df, a_dfs
