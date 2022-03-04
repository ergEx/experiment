import numpy as np
import os
import matplotlib.pyplot as plt
from .dashboard import pandas_save_loader
import pandas as pd
import argparse
from typing import List


def heatmap(ax:plt.Axes, data:np.ndarray) -> plt.Axes :
    """Heatmap with labeled fields.

    Args:
        ax (plt.Axes): Plotting axes
        data (np.ndarray): Data matrix to be plotted.

    Returns:
        plt.Axes: [description]
    """
    ax.pcolormesh(data)

    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                     horizontalalignment='center',
                     verticalalignment='center')

    return ax


def covar_to_correlation(covar_matrix:np.ndarray) -> np.ndarray:
    """Calculates correlations from covariance matrix.

    Args:
        covar_matrix (np.ndarray): The covariance matrix.

    Returns:
        np.ndarray: the correlation matrix.
    """

    D = np.sqrt(np.diag(covar_matrix))
    outer_v = np.outer(D, D)
    correlation = covar_matrix / outer_v

    return correlation


def plot_heatmap(ax: plt.Axes, design_matrix: pd.DataFrame,
                 columns: List[str], correlation: bool=True) -> plt.Axes:
    """Plots a heatmap using a nilearn design matrix.

    Args:
        ax (plt.Axes): Axis to plot on.
        design_matrix (pd.DataFrame): Nilean design matrix
        columns (List[str]): List of columns to include.
        correlation (bool, optional): Plot correlation or covar. Defaults to True.

    Returns:
        plt.Axes: Returns modified axis.
    """

    covar = np.cov(design_matrix[columns].T)
    title_str = 'Covariance Matrix'

    if correlation:
        covar = covar_to_correlation(covar)
        title_str = 'Correlation Matrix'

    ax = heatmap(ax, covar)
    ax.set(xticks=np.arange(covar.shape[0]) + 0.5,
        yticks=np.arange(covar.shape[0])  + 0.5)
    ax.set_xticklabels(labels=columns, rotation=45 )
    ax.set_yticklabels(labels=columns)
    ax.invert_yaxis()

    ax.set_title(title_str)

    return ax


def plot_efficency(ax: plt.Axes, design_matrix: pd.DataFrame,
                   columns:List[str]) -> plt.Axes:
    """Calculates the efficency of a given regressor using a 1/30 Hz sinewave.

    Args:
        ax (plt.Axes): axis to plot on
        design_matrix (pd.DataFrame): nilearn design matrix
        columns (List[str]): columns to include

    Returns:
        plt.Axes: axis with plot.
    """

    n_frames = np.array(design_matrix.index)
    sinewave = np.sin(2 * np.pi * 1/30 * n_frames)

    efficiency = np.var(design_matrix[columns].values, 0) / np.var(sinewave)
    ax.bar(np.arange(1, efficiency.shape[0] + 1), efficiency)
    ax.set_xticks(np.arange(1, efficiency.shape[0] + 1))
    ax.set_xticklabels(columns, rotation=15)
    ax.set_title("Regressor Efficency\nvar / var(sin)")
    text = [[f'{i}'] for i in efficiency]

    ax.table(colLabels=['Efficency'], rowLabels=columns,
            cellText=text, cellLoc='center', bbox=[0.2, -0.5, 0.8, 0.4])

    return ax


def plot_design_matrix(ax: plt.Axes, design_matrix: pd.DataFrame,
                       columns: List[str]) -> plt.Axes:
    """Plots a design matrix

    Args:
        ax (plt.Axes): axis to plot on
        design_matrix (pd.DataFrame): nilearn design matrix
        columns (List[str]): columns to include

    Returns:
        plt.Axes: axis with plot.
    """

    ax.pcolormesh(design_matrix[columns], cmap='Greys_r')
    ax.invert_yaxis()
    ax.set(xticklabels=columns,
        xticks=np.arange(len(columns)) + 0.5)
    ax.xaxis.set_tick_params(labeltop='on')
    ax.set(yticklabels=np.round(np.array(design_matrix.index), 3)[::50],
        yticks=np.arange(len(design_matrix.index), step=50))


def passive_mri_report(fname:str, TR: float=0.592,
                       target_dir: str='data/reports'):
    """MRI test analysis in terms of efficency and design matrix.

    Args:
        fname (str): name to the logfile
        TR (float, optional): The TR. Defaults to 0.592.
        target_dir (str, optional): Target Directory. Defaults to 'data/reports'.
    """
    try:
        from nilearn.glm.first_level import make_first_level_design_matrix
    except:
        ImportError('Please install nilearn before running efficency analysis.')

    dataframe = pandas_save_loader(fname, task='passive')

    filter_idx = dataframe.eval("event_type != 'TrialEnd' and event_type != 'TR'" )

    dataframe_filtered = dataframe[filter_idx]
    dataframe_filtered.loc[:, 'trial_type'] = dataframe_filtered.event_type.values
    dataframe_filtered.loc[:, 'duration'] = 0

    max_time = dataframe.onset.values[-1]

    design_cols = ['onset', 'duration', 'trial_type']
    n_frames = np.arange(max_time / TR) * TR
    design_matrix = make_first_level_design_matrix(n_frames,
                                                dataframe_filtered[design_cols],
                                                drift_model=None, hrf_model='spm')

    columns = ['FractalOnset', 'ITI', 'ReminderOnset', 'Response',
               'ResponseCue', 'WealthUpdate', 'WheelSpin']

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    axes = axes.flatten()

    axes[0] = plot_heatmap(axes[0], design_matrix, columns, True)
    axes[1] = plot_heatmap(axes[1], design_matrix, columns, False)
    axes[2] = plot_efficency(axes[2], design_matrix, columns)
    axes[3] = plot_design_matrix(axes[3], design_matrix, columns)

    fname_bare = os.path.split(fname)[-1][:-4]

    plt.suptitle(fname_bare)
    plt.tight_layout()
    plt.savefig(target_dir + os.sep + 'MRI_' + fname_bare + '.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "CLI to generate fMRI efficency reports")
    parser.add_argument('-f', '--fname', type=str, help='Filename.',
                        default='data/sub-0_sess-001_task-passive_eta-0.0_events.tsv')

    parser.add_argument('-targ', '--targetdir', type=str,
                        help='Target folder',
                        default='data/reports')
    parser.add_argument('-tr', '--TR', type=float,
                        help='Time of repetition.',
                        default=0.592)

    args = parser.parse_args()
    fname = str(args.fname)
    TR = float(args.TR)
    target_dir = str(args.targetdir)

    print(f"Creating report for {fname}")
    passive_mri_report(fname, TR=TR, target_dir=target_dir)

    print("DONE!")
