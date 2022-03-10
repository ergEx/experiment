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
        return ValueError("Not implemented for eta > 1!")

    if np.isscalar(x):
        x = np.asarray((x, ))

    u = np.zeros_like(x, dtype=float)

    if np.isclose(eta, 1):
        u[x > 0] = np.log(x[x > 0])
        u[x <= 0] = np.finfo(float).min
    else:
        bound = (-1) / (1 - eta)
        u[x > 0] = (np.power(x[x > 0], 1-eta) - 1) / (1 - eta)
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
        return ValueError("Not implemented for eta > 1!")

    if np.isscalar(u):
        u = np.asarray((u, ))

    x = np.zeros_like(u, dtype=float)

    if np.isclose(eta, 1):
        x = np.exp(u)
    else:
        bound = (-1) / (1 - eta)
        x[u > bound] = np.power(u[u > bound] * (1 - eta) + 1, 1 / (1 - eta))
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


def calculate_dx1(eta:float, dx2:int, x:int):
    """Calculate change in wealth that corresponds to given indifference-eta
       given dx2 and x.

    Args:
        eta (float):
            Indifference-eta, ie. riskpreference being tested.
        dx2 (int):
            Wealth change of other fractals.
        x (int):
            Wealth level investigated.
    Returns:
        wealth change (float)
    """
    if np.isclose(eta, 1):
        return math.exp(1*math.log(x)-math.log(x + dx2)) - x
    else:
        return (2 * x ** (1 - eta) - (x + dx2) ** (1 - eta)) ** (1 / (1 - eta)) - x


def create_gambles(etas:list, dynamic:float, dx2:int, x:int):
    """Create list of all gambles.

    Args:
        eta (list):
            List of indifference-etas, ie. riskpreferences being tested.
        dx2 (int):
            Wealth change of other fractals.
        x (int):
            Wealth level investigated.
    Returns:
        List of arrays. Each gamble is represented as (2, ) array with growth
        rates.
    """
    dx1_list = [calculate_dx1(eta, dx2, x) for eta in etas]
    gamma_list = [float(isoelastic_utility(x + dx1, dynamic)-isoelastic_utility(x, dynamic)) for dx1 in dx1_list]
    gamma_2 = float(isoelastic_utility(x + dx2, dynamic)-isoelastic_utility(x, dynamic))
    gamble_list = [np.array(
        [gamma_1, gamma_2]) for gamma_1 in gamma_list]

    fractal_dict = {}
    for i, gamma in enumerate(gamma_list):
        fractal_dict[gamma] = i

    fractal_dict[gamma_2] = i + 1
    fractal_dict[0] = i + 2

    return gamble_list,fractal_dict


def create_gamble_pairs(gambles):
    """Dummy function to make v2 compatable with v1

    Args:
        gamble (list of arrays):
            List of gambles.

    Returns:
        List of arrays. Each gamble pair is represented as (2, 2) array with
        four growth rates, but where growth rates of gamble 2 is always 0.
        """
    return [
        np.concatenate((gamble_1[np.newaxis], np.array([[0, 0]])), axis=0)
        for gamble_1 in gambles
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