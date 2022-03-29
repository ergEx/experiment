"""
Plotting functions used in the dashboard.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ..configs import active_configs as acfg
from ..configs import passive_configs as pcfg

# %%
def rand_jitter(arr:np.ndarray):
    stdev = .02 * (np.nanmax(arr) - np.nanmin(arr))
    return arr + np.random.randn(len(arr)) * stdev

# %%
def pandas_save_loader(filename:str, task:str = 'passive'):
    if task == 'active':
        cat_dtypes = {'onset': float, 'duration': float, 'trial_type': str,
                      'event_type': str, 'response_time': float,
                      'response_button': str, 'response_is_last': float,
                      'response_time_optimal': float, 'gamble_up': str,
                      'wealth': float, 'delta_wealth': float, 'trial': float,
                      'gamma_left_up': float, 'gamma_left_down': float,
                      'gamma_right_up': float, 'gamma_right_down': float,
                      'fractal_left_up': str, 'fractal_left_down': str,
                      'fractal_right_up': str, 'fractal_right_down': str,
                      'eta': float, 'trial_time': float,
                      'participant_id': str, 'TR': float, 'no_response': float,
                      'chosen_expected_gamma': float, 'realized_gamma': float,
                      'expected_duration': float, 'selected_side': str}

    elif task == 'passive':
        cat_dtypes = {'onset': float, 'duration': float, 'trial_type': str,
                      'event_type': str, 'response_time': float,
                      'response_button': str, 'response_late': float,
                      'response_correct': float, 'no_response': float,
                      'wealth': float, 'delta_wealth': float, 'trial': float,
                      'gamma': float, 'fractal': str, 'eta': float,
                      'trial_time': float, 'participant_id': str, 'TR': float,
                      'expected_duration': float, 'gamma_left': float,
                      'gamma_right': float, 'fractal_left': str,
                      'fractal_right': str, 'chosen_gamma': float,
                      'selected_side': str, 'run': float, 'part': float}

    else:
        raise ValueError('task has to be passive or active!')

    dataframe = pd.read_csv(filename, sep='\t', na_values='n/a', dtype=cat_dtypes)

    return dataframe


def plot_type_bar(dataframe, ax, ex_type='Event'):
    if ex_type == 'Event':
        event_types = dataframe.query('event_type != "TR"')
        ct_vc = event_types.event_type.value_counts()

    elif ex_type == 'Trial':
        trial_types = dataframe.query('event_type == "TrialEnd"')
        ct_vc = trial_types.trial_type.value_counts()

    ax.bar(np.arange(len(ct_vc)), ct_vc)
    ax.set_xticks(np.arange(len(ct_vc)))
    ax.set_xticklabels(list(ct_vc.index))


    ax.set(ylabel=f'{ex_type} Type', xlabel='',
                title=f'{ex_type} Types')

    return ax


def plot_expected_gamma(dataframe, ax, direction='horizontal'):
    gammas = dataframe.query('event_type=="WealthUpdate"')

    if direction == 'horizontal':
        gammas_1 = gammas[['gamma_left_up', 'gamma_left_down']].mean(1)
        gammas_2 = gammas[['gamma_right_up', 'gamma_right_down']].mean(1)
    elif direction == 'vertical':
        gammas_1 = gammas[['gamma_left_up', 'gamma_right_up']].mean(1)
        gammas_2 = gammas[['gamma_left_down', 'gamma_right_down']].mean(1)
    elif direction == 'nobrainer':
        gammas_1 = gammas['gamma_left']
        gammas_2 = gammas['gamma_right']
    else:
        raise ValueError('Diretion must be in ["horizontal", "vertical"]')

    ax.hist(gammas_1.values, alpha=0.5, density=True, color='b')
    ax.hist(gammas_2.values, alpha=0.5, density=True, color='orange')

    legend = ax.get_legend_handles_labels()

    if direction == 'horizontal':
        ax.legend(['Left', 'Right'])
    elif direction =='vertical':
        ax.legend(['Up', 'Down'])


    xlim = [np.min([gammas_1.min(), gammas_2.min()]),
            np.max([gammas_1.max(), gammas_2.max()])]

    # Add 2 % of range to sides for visuals
    xlim[0] = xlim[0] - np.abs(0.2 * xlim[0])

    xlim[1] = xlim[1] + np.abs(0.2 * xlim[1])
    ax.set(ylabel='Density', xlabel='Average Gamma',
            title=f'Average Gamma per Side ', xlim=xlim)

    return ax


def plot_prob_heads(dataframe, ax):
    coin_tosses = dataframe.query('event_type=="Coin"').gamble_up

    ct_vc = coin_tosses.value_counts()
    ax.bar(np.arange(len(ct_vc)), ct_vc)
    ax.set_xticks(np.arange(len(ct_vc)))
    ax.set_xticklabels(list(ct_vc.index))
    ax.set(ylabel='Count', xlabel='Coin Position',
           title=f'Coin Toss Distribution')

    return ax


def plot_wealth_trajectory(dataframe, ax):

    wealth_trajectory = dataframe.query('event_type=="WealthUpdate"')

    wealth_traj = wealth_trajectory.reset_index()
    ax.plot(wealth_traj.wealth)

    ax.set(ylabel='Wealth', xlabel='Trial',
        title=f'Wealth Trjactory over Trials')

    return ax


def plot_event_durations(dataframe, ax, event='ITI'):

    durations = dataframe.query("event_type==@event").duration

    ax.hist(durations.values, density=True, bins=20, edgecolor='k',
            color='blue', alpha=0.5)

    xlim = ax.get_xlim()

    ax.set(ylabel='Duration in s', xlabel='s',
        title=f'Durations for {event}', xlim=xlim)

    return ax


def plot_event_duration_error(dataframe, ax, event='ITI'):

    durations = dataframe.query("event_type==@event")

    timing_error = (durations.duration.values -
                    durations.expected_duration.values) * 1000

    ax.hist(timing_error, density=True, bins=20, edgecolor='k',
            color='blue', alpha=0.5, orientation='vertical')

    ax.axvline(timing_error.mean(), color='r')

    ax.axvline(np.percentile(timing_error, 5), color='g')
    ax.axvline(np.percentile(timing_error, 95), color='g')


    ax.set(ylabel='Counts', xlabel='ms',
        title=f'Timing Error for {event}')

    ax.legend(['Timing Error', 'Mean', '5-95 Percentiles'])

    return ax


def plot_tr_timings_through(dataframe, ax):

    trs = dataframe.query('event_type == "TR"').copy()

    trt_1 = trs.response_time.values
    trt_1 = trt_1 - trt_1[0]

    trs.loc[:, 'response_time'] = trt_1 # trs.response_time.values - trs.response_time.values[0] # reset values.

    timings = trs.response_time.diff()

    emp_tr = timings.mean()

    ax.bar(np.arange(len(timings)), timings)

    ax.axhline(emp_tr, color='r')
    ax.axhline(np.percentile(timings, 5), color='g')
    ax.axhline(np.percentile(timings, 95), color='g')
    ax.legend(['timings', 'mean', '5-95 Percentiles'])

    max_time = trs.onset.values[-1]

    expected_tr = int(max_time / emp_tr)

    ax.set(ylabel='TR diff in s', xlabel='TRs',
            title=f'TR={emp_tr} timings\nCaught {int(trs.TR.values[-1]):4.4f} of {expected_tr} + 1 TRs')


    return ax


def plot_reaction_time_distribution(dataframe, ax, task='active'):

    # Get RT data
    rt = dataframe.query('event_type=="Response"').response_time

    ax.hist(rt.values, density=True, bins=20, edgecolor='k', color='blue', alpha=0.5)
    #rt.plot.kde(ax=ax)
    xlim = ax.get_xlim()

    if task == 'active':
        xlim = [-0.05,  acfg.timeResponse + 0.05]
    elif task == 'passive':
        xlim = [-0.05, pcfg.timeResponseWindow + 0.05]

    ax.set(ylabel='Density', xlabel='RT in s', title='RT distribution', xlim=xlim)

    return ax


def plot_response_button_distribution(dataframe, ax, task='active'):

    response_button = dataframe.query('event_type=="Response"').response_button

    rb_vc = response_button.value_counts()
    ax.bar(np.arange(len(rb_vc)), rb_vc)
    ax.set_xticks(np.arange(len(rb_vc)))
    ax.set_xticklabels(list(rb_vc.index))
    ax.set(ylabel='Counts', xlabel='Button Presses', title='Buttons Pressed')

    return ax


def plot_time_optimal_responses(dataframe, ax, task='active'):

    if task == 'active':
        to_response = dataframe.query('event_type=="TrialEnd"').response_time_optimal
    elif task == 'nobrainer':
        to_response = dataframe.query('event_type=="TrialEnd"').response_correct

    tr_vc = to_response.value_counts()
    ax.bar(np.arange(len(tr_vc)), tr_vc)
    ax.set_xticks(np.arange(len(tr_vc)))
    ax.set_xticklabels(list(tr_vc.index))
    ax.set(ylabel='Counts', xlabel='Time Optimal Responses', title='Time Optimal Choice')

    return ax


def plot_realized_gambles(dataframe, ax):

    realized_gamble = dataframe.query('event_type=="TrialEnd"').realized_gamma
    expected_gamble = dataframe.query('event_type=="TrialEnd"').chosen_expected_gamma
    # Violation:
    rpe = np.abs(realized_gamble.values - expected_gamble.values)
    rpe[np.isnan(rpe)] = 0
    ax.scatter(rand_jitter(expected_gamble.values),
               rand_jitter(realized_gamble.values),
               s=20, c=rpe, edgecolors='b')
    ylim = ax.get_ylim()
    ax.axline((np.min(realized_gamble), np.min(realized_gamble)),
              (np.max(realized_gamble), np.max(realized_gamble)),
              linewidth=2, ls='--')
    ax.axhline(0, ls='--', color='k')
    ax.axvline(0, ls='--', color='k')
    xlim = ax.get_xlim()
    xdist = xlim[1] - xlim[0]
    ylim = ax.get_ylim()
    ydist = ylim[1] - ylim[0]
    # Add counts for each field:
    right_up = np.sum((expected_gamble > 0) & (realized_gamble > 0))
    left_up = np.sum((expected_gamble <= 0) & (realized_gamble > 0))
    right_down = np.sum((expected_gamble > 0) & (realized_gamble <= 0))
    left_down = np.sum((expected_gamble <= 0) & (realized_gamble <= 0))

    ax.text(xlim[1] - 0.25 * xdist, ylim[1] - 0.25 * ydist, f"{right_up:2.0f}")
    ax.text(xlim[1] - 0.75 * xdist, ylim[1] - 0.25 * ydist, f"{left_up:2.0f}")
    ax.text(xlim[1] - 0.25 * xdist, ylim[1] - 0.75 * ydist, f"{right_down:2.0f}")
    ax.text(xlim[1] - 0.75 * xdist, ylim[1] - 0.75 * ydist, f"{left_down:2.0f}")

    ax.set(ylabel='Realized', xlabel='Expected',
           title='Expectation vs. Realization', xlim=ylim)

    return ax


def plot_late_responses(dataframe, ax, task='passive'):
    if task == 'passive':
        responses = dataframe.query('event_type == "Response"').response_late
    elif task == 'active':
        responses = dataframe.query('event_type == "TrialEnd"').no_response

    rs_c = responses.value_counts()
    ax.bar(np.arange(len(rs_c)), rs_c) #.plot.bar(ax=ax)
    ax.set_xticks(np.arange(len(rs_c)))
    ax.set_xticklabels(list(rs_c.index))

    ax.set(ylabel='Count', xlabel='Pressed or Not',
           title=f'None or Late Responses')

    return ax


def plot_rt_versus_difficulty(dataframe, ax, task='active'):
    reaction = dataframe.query('event_type == "Response"')

    reaction = reaction.response_time.values
    gammas = dataframe.query('event_type=="WealthUpdate" and no_response == False')

    if task == 'active':
        gammas_1 = gammas[['gamma_left_up', 'gamma_left_down']].mean(1)
        gammas_2 = gammas[['gamma_right_up', 'gamma_right_down']].mean(1)
    elif task=='nobrainer':
        gammas_1 = gammas['gamma_left']
        gammas_2 = gammas['gamma_right']

    button = gammas.selected_side == 'left'
    distance = np.abs(gammas_1.values - gammas_2.values)

    ax.scatter(distance, rand_jitter(reaction), s=20, c=button, edgecolors='b')

    ax.set(title='Reaction Time vs. Difficulty', xlabel='Gamble Distance',
           ylabel='RT')

    return ax


def plot_choice_probability(dataframe, ax):

    gammas = dataframe.query('event_type=="WealthUpdate" and no_response == False')

    gammas_1 = gammas[['gamma_left_up', 'gamma_left_down']].mean(1)

    button = gammas.selected_side == 'left'

    choices = np.unique(gammas_1)
    probs = np.zeros(choices.shape)

    for n, ch in enumerate(choices):
        probs[n] = np.mean(button[gammas_1==ch])

    # ax.bar(choices, probs)
    ax.plot(choices, probs, '--*')

    ax.set(title='Choice Probability', xlabel='Unique Gammas',
           ylabel='Probability')

    return ax


def plot_to_trajectory(dataframe, ax):
    gammas = dataframe.query('event_type == "TrialEnd"')
    optimal_path = (gammas[['gamma_left_up', 'gamma_left_down']].mean(1) <
                    gammas[['gamma_right_up', 'gamma_right_down']].mean(1)) * 1 # to cast
    max_gammas = np.concatenate([gammas[['gamma_left_up',
                                        'gamma_left_down']].max(1).values[:, np.newaxis],
                                 gammas[['gamma_right_up',
                                        'gamma_right_down']].max(1).values[:, np.newaxis]],
                                axis=1)
    max_gamma_path = np.take_along_axis(max_gammas,
                                       optimal_path.values[:, np.newaxis] * 1,
                                       axis=1).cumsum()

    min_gammas = np.concatenate([gammas[['gamma_left_up',
                                        'gamma_left_down']].min(1).values[:, np.newaxis],
                                 gammas[['gamma_right_up',
                                        'gamma_right_down']].min(1).values[:, np.newaxis]],
                                axis=1)
    min_gamma_path = np.take_along_axis(min_gammas,
                                        (optimal_path.values[:, np.newaxis]) * 1,
                                        axis=1).cumsum()

    exp_gammas = np.concatenate([gammas[['gamma_left_up',
                                        'gamma_left_down']].mean(1).values[:, np.newaxis],
                                 gammas[['gamma_right_up',
                                        'gamma_right_down']].mean(1).values[:, np.newaxis]],
                                axis=1)
    exp_gamma_path = np.take_along_axis(exp_gammas,
                                        (optimal_path.values[:, np.newaxis]) * 1,
                                        axis=1).cumsum()

    ax.plot(max_gamma_path[:, np.newaxis])
    ax.plot(min_gamma_path[:, np.newaxis])
    ax.plot(exp_gamma_path[:, np.newaxis])
    ax.plot(gammas['realized_gamma'].cumsum().values[:, np.newaxis])

    ax.legend(['Max', 'Min', 'Expectation', 'Realized'], loc='upper center',
              bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=False)

    ax.set(title='Gamma Trajectories', xlabel='Trial', ylabel='Cumulative Gamma')
    return ax
# %%

def passive_report(fname:str, target_dir:str = 'data/reports'):
    dataframe = pandas_save_loader(fname, 'passive')
    dataframe = dataframe.query('part == 0')

    fig, axes = plt.subplots(4, 3, figsize=(12.5, 15), sharex=False, sharey=False)

    axes = axes.flatten()
    ii = 0
    ax = plot_reaction_time_distribution(dataframe, axes[ii], task='passive')
    ii += 1
    ax = plot_response_button_distribution(dataframe, axes[ii], task='passive')
    ii += 1
    try:
        ax = plot_tr_timings_through(dataframe, axes[ii])
    except IndexError:
        axes[ii].axis('off')

    ii += 1
    ax = plot_type_bar(dataframe, axes[ii], ex_type='Event')
    ii += 1
    ax = plot_wealth_trajectory(dataframe, axes[ii])
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='ITI')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='WheelSpin')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='WealthUpdate')
    ii += 1
    ax = plot_late_responses(dataframe, axes[ii], task='passive')
    ii += 1 # 9
    ax = plot_type_bar(dataframe, axes[ii], ex_type='Trial')

    fname_bare = os.path.split(fname)[-1][:-4]

    plt.suptitle(fname_bare)
    plt.tight_layout()
    plt.savefig(target_dir + os.sep +  fname_bare + '.png')


def active_report(fname, target_dir='data/reports'):

    dataframe = pandas_save_loader(fname, task='active')

    fig, axes = plt.subplots(7, 3, figsize=(12.5, 17), sharex=False, sharey=False)

    axes = axes.flatten()
    ii = 0
    ax = plot_reaction_time_distribution(dataframe, axes[ii], task='active')
    ii += 1
    print(ii, 'rt done')
    ax = plot_response_button_distribution(dataframe, axes[ii])
    ii += 1
    print(ii, 'button done')

    try:
        ax = plot_tr_timings_through(dataframe, axes[ii])
    except IndexError:
        axes[ii].axis('off')

    ii += 1
    ax = plot_type_bar(dataframe, axes[ii], ex_type='Event')
    ii += 1
    ax = plot_wealth_trajectory(dataframe, axes[ii])
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='ITI')
    ii += 1
    try:
        ax = plot_event_duration_error(dataframe, axes[ii], event='GambleLeft')
    except IndexError:
        axes[ii].axis('off')

    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='GambleRight')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='SideSelection')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='Coin')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='FractalSelection')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='WealthUpdate')
    ii += 1
    ax = plot_prob_heads(dataframe, axes[ii])
    ii += 1
    ax = plot_expected_gamma(dataframe, axes[ii], direction='horizontal')
    ii += 1
    ax = plot_expected_gamma(dataframe, axes[ii], direction='vertical')
    ii += 1
    ax = plot_time_optimal_responses(dataframe, axes[ii])
    ii += 1
    ax = plot_realized_gambles(dataframe, axes[ii])
    ii += 1
    ax = plot_late_responses(dataframe, axes[ii], task='active')
    ii += 1
    ax = plot_choice_probability(dataframe, axes[ii])
    ii += 1 # 19
    ax = plot_rt_versus_difficulty(dataframe, axes[ii])
    ii += 1 # 20
    ax = plot_to_trajectory(dataframe, axes[ii])

    fname_bare = os.path.split(fname)[-1][:-4]

    plt.suptitle(fname_bare)
    plt.tight_layout()
    plt.savefig(target_dir + os.sep +  fname_bare + '.png')


def nobrainer_report(fname, target_dir='data/reports'):
    dataframe = pandas_save_loader(fname, task='passive')
    dataframe = dataframe.query('part == 1')
    trials_data = dataframe.query('event_type == "TrialEnd"')
    to_answer = trials_data.response_correct

    print(f"Proportion correct: {(to_answer.mean() * 100):4.2f} % || "
          f"Correct responses: {np.int(to_answer.sum())} / {to_answer.shape[0]}")

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 17), sharex=False, sharey=False)

    axes = axes.flatten()
    ii = 0
    ax = plot_reaction_time_distribution(dataframe, axes[ii], task='active')
    ii += 1
    ax = plot_response_button_distribution(dataframe, axes[ii])
    ii += 1
    try:
        ax = plot_tr_timings_through(dataframe, axes[ii])
    except IndexError:
        axes[ii].axis('off')

    ii += 1
    ax = plot_type_bar(dataframe, axes[ii], ex_type='Event')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='ITI')
    ii += 1
    ax = plot_event_duration_error(dataframe, axes[ii], event='WealthUpdate')
    ii += 1
    ax = plot_expected_gamma(dataframe, axes[ii], direction='nobrainer')
    ii += 1
    ax = plot_rt_versus_difficulty(dataframe, axes[ii], 'nobrainer')
    ii += 1
    ax = plot_time_optimal_responses(dataframe, axes[ii], 'nobrainer')

    fname_bare = os.path.split(fname)[-1][:-4]
    fname_bare = fname_bare.replace('passive', 'nobrainer')

    plt.suptitle(fname_bare)
    plt.tight_layout()
    plt.savefig(target_dir + os.sep +  fname_bare + '.png')
