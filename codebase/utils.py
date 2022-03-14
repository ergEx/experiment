# Util functions stolen from https://github.com/kbonna/ergodicity-experiments

import numpy as np
import math
from itertools import combinations,combinations_with_replacement
import matplotlib.pyplot as plt


class DotDict(dict):
    """Small helper class, so attributed of dictionary can be accessed using
        dot.notation"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def isoelastic_utility(x:np.ndarray, eta:float) -> np.ndarray:
    """Isoelastic utility for a given wealth.

    Args:
        x (array):
            Wealth vector.
        eta (float):
            Risk-aversion parameter.

    Returns:
        Vector of utilities corresponding to wealths. For log utility if wealth
        is less or equal to zero, smallest float possible is returned. For other
        utilites if wealth is less or equal to zero, smallest possible utility,
        i.e., specicfic lower bound is returned.
    """
    if eta >= 2:
        return ValueError("Not implemented for eta geq 2!")

    if np.isscalar(x):
        x = np.asarray((x, ))

    u = np.zeros_like(x, dtype=float)

    if np.isclose(eta, 1):
        u[x > 0] = np.log(x[x > 0])
        u[x <= 0] = np.finfo(float).min
    elif eta < 1:
        bound = (-1) / (1 - eta)
        u[x > 0] = (np.power(x[x > 0], 1-eta) - 1) / (1 - eta)
        u[x <= 0] = bound
    elif 1 < eta < 2:
        bound = (-1) / (1 - (eta-1))
        u[x > 0] = (np.power(np.log(x[x > 0]), 1 - (eta - 1))) / (1 - (eta - 1))
        u[x <= 0] = bound
    return u


def inverse_isoelastic_utility(u:np.ndarray, eta:float) -> np.ndarray:
    """Inverse isoelastic utility function mapping from utility to wealth.

    Args:
        u (array):
            Utility vector.
        eta (float):
            Risk-aversion parameter.

    Returns:
        Vector of wealths coresponding to utilities. For
    """

    if eta >= 2:
        return ValueError("Not implemented for eta geq 2!")

    if np.isscalar(u):
        u = np.asarray((u, ))

    x = np.zeros_like(u, dtype=float)

    if np.isclose(eta, 1):
        x = np.exp(u)
    elif eta < 1:
        bound = (-1) / (1 - eta)
        x[u > bound] = np.power(u[u > bound] * (1 - eta) + 1, 1 / (1 - eta))
    elif 1 < eta < 2:
        bound = (-1) / (1 - (eta - 1))
        x[u > bound] = np.exp(np.power(u[u > bound] * (1 - (eta - 1)) , 1 / (1 - (eta - 1))))
    return x


def wealth_change(x, gamma, eta):
    """Apply isoelastic wealth change.

    Args:
        x (float):
            Initial wealth vector.
        gamma (float):
            Growth rate.
        eta (float):
            Wealth dynamic parameter.
    """
    return inverse_isoelastic_utility(isoelastic_utility(x, eta) + gamma, eta)


def shuffle_along_axis(a, axis):
    """Randomly shuffle multidimentional array along specified axis."""
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def random_reorder_axis(a, axis):
    """Shuffle along axis, keeping all other axis intact.

    Args:
        a ([type]): [description]
        axis ([type]): [description]
    """
    idx = np.arange(a.shape[axis])
    np.random.shuffle(idx)
    return np.take(a, idx, axis=axis)


def create_gambles(c, n_fractals=9):
    """Create list of all gambles.

    Args:
        c (float):
            Max growth rate for gamble space.
        n_fractals (int):
            Number of growth rate samples.
    Returns:
        List of arrays. Each gamble is represented as (2, ) array with growth
        rates. For n fractals, n(n+1)/2 gambles are created. Order of growth
        rates doesn't matter since probabilities are assigned equally to both
        wealth changes.
    """
    gamma_range = np.linspace(-c, c, n_fractals)
    fractal_dict = {ii : n for n, ii in enumerate(gamma_range)}
    return [
        np.array([gamma_1, gamma_2])
        for gamma_1, gamma_2
        in combinations_with_replacement(gamma_range, 2)
    ], fractal_dict


def create_gamble_pairs(gambles):
    """Create list of all unique gamble pairs.

    Args:
        gambles (list of arrays):
            List of gambles.

    Returns:
        List of arrays. Each gamble pair is represented as (2, 2) array with
        four growth rates for both gambles. Rows corresponds to gambles, columns
        correspond to individual growth rates within a gamble. All pairs contain
        two unique gambles. For n gambles, n(n-1)/2 gamble pairs are created.
    """
    return [
        np.concatenate((gamble_1[np.newaxis], gamble_2[np.newaxis]), axis=0)
        for gamble_1, gamble_2 in combinations(gambles, 2)
        ]


def create_trial_order(n_simulations, n_gamble_pairs, n_trials):
    """Generates randomized trial order for paralell simulations.

    Args:
        n_simulations (int):
            Number of paralell simulations.
        n_gamble_pairs (int):
            Number of unique, available gamble pairs.
        n_trials (int):
            Number of experimental trials.

    Returns:
        Array of shape n_trials x n_simulations with indices corresponding to
        gamble pairs.
    """
    # indicates how many times gamble pairs should be repeated to span n_trials
    repetition_factor = math.ceil(n_trials / n_gamble_pairs)

    basic_order = np.arange(n_gamble_pairs)[:, np.newaxis]
    trial_order = np.repeat(basic_order, n_simulations, axis=1)
    trial_order = shuffle_along_axis(trial_order, axis=0)
    trial_order = np.tile(trial_order, (repetition_factor, 1))
    trial_order = trial_order[:n_trials]
    trial_order = shuffle_along_axis(trial_order, axis=0)

    return trial_order


def create_experiment(gamble_pairs):
    """Creates experiment array.

    Args:
        gamble_pairs (list of arrays):
            List of gamble pairs.

    Returns:
        Array of size (2, 2, n_trials). First two dimensions correspond to
        gamble pair, third dimension correspond to subsequent trials.
    """
    return np.stack(gamble_pairs, axis=2)


def is_nobrainer(gamble_pair):
    """Decision if a gamble pair is nobrainer."""
    return len(set(gamble_pair[0]).intersection(set(gamble_pair[1]))) != 0


def is_statewise_dominated(gamble_pair):
    """Decision if a gamble is strictly statewise dominated by the other gamble in a gamble pair"""
    return (np.greater(max(gamble_pair[0]), max(gamble_pair[1])) and np.greater(min(gamble_pair[0]), min(gamble_pair[1])) or
           np.greater(max(gamble_pair[1]), max(gamble_pair[0])) and np.greater(min(gamble_pair[1]), min(gamble_pair[0])) )


def is_stochastically_dominated(gamble_pair):
    """Decision if one gamble is first order stochastically dominated by the other in a gamble pair"""
    F_1 = np.cumsum(np.sort(gamble_pair[0]))
    F_2 = np.cumsum(np.sort(gamble_pair[1]))
    return all(x >= y for x, y in zip(F_1, F_2)) or all(x >= y for x, y in zip(F_2, F_1))


def is_g_deterministic(gamble):
    """Decision if gamble is deterministic, i.e., composed of two same fractals.

    Args:
        gamble (np.array):
            Gamble array of shape (2, 0).

    Returns:
        Boolean decision value.
    """
    return gamble[0] == gamble[1]


def calculate_c(eta):
    """Function to that provides a c-value based purely on eta

    This is just a dummy function - will be rewritten!

    """
    c_values = {0: 428, 1: 0.806}
    if eta not in c_values.keys():
        return ValueError(f"Eta value of {eta} cannot be chosen, choose between: {list(c_values.keys())}")
    return c_values[eta]

