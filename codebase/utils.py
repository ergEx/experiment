import itertools
import math
import numpy as np
from scipy.optimize import fsolve


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
            Risk-aversion.

    Returns:
        Vector of utilities corresponding to wealths. For log utility if wealth
        is less or equal to zero, smallest float possible is returned. For other
        utilites if wealth is less or equal to zero, smallest possible utility,
        i.e., specicfic lower bound is returned.
    """

    if np.isscalar(x):
        x = np.asarray((x, ))

    u = np.zeros_like(x, dtype=float)

    if np.isclose(eta, 1):
        u[x > 0] = np.log(x[x > 0])
        u[x <= 0] = np.finfo(float).min
    elif np.isclose(eta, 0): #allow negative values in additive dynamic
        u = (np.power(x, 1-eta) - 1) / (1 - eta)
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
            Risk-aversion.

    Returns:
        Vector of wealths coresponding to utilities.
    """

    if eta > 1:
        return ValueError("Not implemented for eta > 1!")

    if np.isscalar(u):
        u = np.asarray((u, ))

    x = np.zeros_like(u, dtype=float)

    if np.isclose(eta, 1):
        x = np.exp(u)
    elif np.isclose(eta, 0): #allow for negative values in additive dynamic
        x = np.power(u * (1 - eta) + 1, 1 / (1 - eta))
    else:
        bound = (-1) / (1 - eta)
        x[u > bound] = np.power(u[u > bound] * (1 - eta) + 1, 1 / (1 - eta))
    return x


def wealth_change(x:np.array, gamma:np.array, lambd:float) -> np.ndarray:
    """Apply isoelastic wealth change.

    Args:
        x (array):
            Initial wealth vector.
        gamma (gamma):
            Growth rates.
        lambd (float):
            Wealth dynamic.

    Returns:
        Vector of updated wealths.
    """

    if np.isscalar(x):
        x = np.asarray((x, ))

    if np.isscalar(gamma):
        gamma = np.asarray((gamma, ))

    return inverse_isoelastic_utility(isoelastic_utility(x, lambd) + gamma, lambd)


def shuffle_along_axis(a:np.array, axis:int) -> np.ndarray:
    """Randomly shuffle multidimentional array along specified axis.

    Args:
        a (array)
        axis (int)

    Returns:
        array
    """
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def random_reorder_axis(a:np.array, axis:int) -> np.ndarray:
    """Shuffle along axis, keeping all other axis intact.

    Args:
        a (array)
        axis (int)

    Returns:
        array
    """
    idx = np.arange(a.shape[axis])
    np.random.shuffle(idx)
    return np.take(a, idx, axis=axis)


def calculate_dx1(indifference_eta:float, dx2:int, x_0:int) -> float:
    """Calculate change in wealth that corresponds to given indifference-eta
       given dx2 and x.

    Args:
        indifference_eta (float):
            Indifference-etas, ie. riskpreference being tested.
        dx2 (int):
            Wealth change of other fractal.
        x_0 (int):
            Wealth level investigated.
    Returns:
        additive wealth change (dx1; float)
    """
    if np.isclose(indifference_eta, 1):
        return round(math.exp(2 * math.log(x_0) - math.log(x_0 + dx2)) - x_0)
    else:
        return round((2 * x_0 ** (1 - indifference_eta) - (x_0 + dx2) ** (1 - indifference_eta)) ** (1 / (1 - indifference_eta)) - x_0)


def calculate_growth_rates(indifference_etas:np.array, lambd:float, dx2:int, x:np.array):
    """Create list of all gambles.

    Args:
        indifference_etas (array):
            Array of indifference-etas, ie. riskpreferences being tested.
        lambd (float):
            Wealth dynamic.
        dx2 (int):
            Wealth change of other fractal.
        x (array):
            Array of reference wealth levels; 0th entry is main reference and 1st is secondary reference.
    Returns:
        List of arrays. Each gamble is represented as (2, ) array with growth
        rates.
    """
    if len(x) > 2:
        return ValueError("You can choose max two reference wealths!")

    dx1_list = [calculate_dx1(eta, dx2, x[0]) for eta in indifference_etas[lambd]]
    gamma1_list = [float(isoelastic_utility(x[0] + dx1, lambd)-isoelastic_utility(x[0], lambd)) for dx1 in dx1_list]

    dx2_list = [dx2, calculate_dx1(0.5, calculate_dx1(0.5, dx2, x[0]), x[1])] if len(x)==2 else [dx2]
    gamma2_list = [float(isoelastic_utility(x[0] + dx2, lambd)-isoelastic_utility(x[0], lambd)) for dx2 in dx2_list]

    gamma_list = gamma1_list + gamma2_list + [0]
    #print(gamma_list)

    fractal_dict = {}
    for i, gamma in enumerate(gamma_list):
        fractal_dict[gamma] = i

    return gamma_list, gamma1_list, gamma2_list, fractal_dict


def create_gambles_two_gambles(gamma_range:np.array) -> np.ndarray:
    """Create list of all gambles.
    Args:
        gamma_range (array):
            List of growth rates

    Returns:
        List of arrays. Each gamble is represented as (2, ) array with growth
        rates. For n fractals, n(n+1)/2 gambles are created. Order of growth
        rates doesn't matter since probabilities are assigned equally to both
        wealth changes.
    """
    return [
        np.array([gamma_1, gamma_2])
        for gamma_1, gamma_2
        in itertools.combinations_with_replacement(gamma_range, 2)
    ]


def create_gamble_pairs_two_gambles(gambles:np.array) -> np.ndarray:
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
        for gamble_1, gamble_2 in itertools.combinations(gambles, 2)
        ]


def create_trial_order(n_simulations:int, n_gamble_pairs:int, n_trials:int) -> np.array:
    """Generates randomized trial order allowing for paralell simulations.

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


def create_experiment(gamble_pairs:np.array) -> np.ndarray:
    """Creates experiment array.

    Args:
        gamble_pairs (list of arrays):
            List of gamble pairs.

    Returns:
        Array of size (2, 2, n_trials). First two dimensions correspond to
        gamble pair, third dimension correspond to subsequent trials.
    """
    return np.stack(gamble_pairs, axis=2)


def is_g_deterministic(gamble:np.array) -> bool:
    """Decision if gamble is deterministic, i.e., composed of two same fractals.
    Args:
        gamble (np.array):
            Gamble array of shape (2, 0).
    Returns:
        Boolean decision value.
    """
    return gamble[0] == gamble[1]


def is_nobrainer(gamble_pair:np.array) -> bool:
    """Decision if a gamble pair is nobrainer."""
    return len(set(gamble_pair[0]).intersection(set(gamble_pair[1]))) != 0


def is_statewise_dominated(gamble_pair:np.array) -> bool:
    """Decision if a gamble is strictly statewise dominated by the other gamble in a gamble pair"""
    return (np.greater(max(gamble_pair[0]), max(gamble_pair[1])) and np.greater(min(gamble_pair[0]), min(gamble_pair[1])) or
           np.greater(max(gamble_pair[1]), max(gamble_pair[0])) and np.greater(min(gamble_pair[1]), min(gamble_pair[0])) )


def indiference_eta(x1:float, x2:float, x3:float, x4:float) -> list:
    """Calculates indiference_etas for gamble pairs, ie. at which riskaversion is an agent indifferent between the choices
    Args:
        x1 (float): after trial wealth if upper left is realized
        x2 (float): after trial wealth if lower left is realized
        x3 (float): after trial wealth if upper right is realized
        x4 (float): after trial wealth if lower right is realized

    Returns:
        Indifference eta (float).
    """
    if x1<0 or x2<0 or x3<0 or x4<0:
        print(x1,x2,x3,x4)
        raise ValueError(f"Isoelastic utility function not defined for negative values")

    func = lambda x : ((isoelastic_utility(x1,x)+ isoelastic_utility(x2,x)) / 2
                     - (isoelastic_utility(x3,x)+ isoelastic_utility(x4,x)) / 2)
    root_initial_guess = -20
    root = fsolve(func, root_initial_guess)

    return root, func
