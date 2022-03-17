""" Agents for passive and active phase, so that experiments can be run without
User input.
"""
import numpy as np
from psychopy import event
from typing import List, Union
from ...utils import isoelastic_utility, wealth_change
from ... import constants as con
# %%
def softmax(x, beta):
    """Softmax choice function.

    Args:
        beta (float):
            Precision parameter (inverse temperature)

    Returns:
        Choice probability.
    """
    return np.power(1 + np.exp((-beta) * x), -1)


def calculate_beta(eta_dynamic, eta_agent, c, x_0, x_limit, p_threshold):
    """Estimate normalized precision parameter.

    Here, a softmax precision (or inverse temperature) for isoelastic agent is
    estimated. It is defined at precision in which an agent endowed with initial
    wealth x0 and facing two extreme, opposite fractals with growth rates -c
    and c would choose the winning fractal with probability p_threshold.
    Wealth changes that exceed [0, x_limit] range are trimmed down before
    calculation.

    Args:
        eta_dynamic (float):
            Wealth dynamic parameter.
        eta_agent (float):
            Isoelastic agent's risk attitude.
        c (float):
            Maximal fractal growth rate.
        x_0 (float):
            Initial wealth.
        x_limit (float):
            Wealth upper-bound.
        p_threshold (float):
            Matched probability of choosing better fractal. Should lie within
            range (0, 1).

    Returns:
        Precision parameter beta (float).
    """
    # extreme wealth changes
    x_f = wealth_change(np.array([x_0, x_0]), np.array([-c, c]), eta_dynamic)
    x_f[x_f < 0] = 0
    x_f[x_f > x_limit] = x_limit

    # utility changes
    u_f = isoelastic_utility(x_f, eta_agent)
    du = u_f[1] - u_f[0]

    beta = np.log(p_threshold / (1 - p_threshold)) / du
    return beta


class PassiveAutoPilot:
    """Class for a passive autopilot / agent - pressing the specified button
    after normaly distributed time.
    """
    def __init__(self, mean_rt:float, sd_rt:float, active:bool = False,
                 responseButton:str = 'space') -> None:
        """Initiate

        Args:
            mean_rt (float): Mean of normal distribution
            sd_rt (float): SD of normal distribution.
            active (bool, optional): If the agent is active or not. Defaults to False.
            responseButton (str, optional): Response key. Defaults to 'space'.
        """
        self.mean_rt = mean_rt
        self.sd_rt = sd_rt
        self.active = active
        self.responseButton = responseButton

    def start_timer(self, time:float):
        """Start timer for agent, sets RT and time after which response occurs.

        Args:
            time (float): Time from which to count down until response.
        """
        self.response_time = np.abs(self.sd_rt * np.random.randn() + self.mean_rt)
        self.start_time = time
        self.press_time = self.start_time + self.response_time

    def press(self):
        """Ushers a key press.
        """
        event._onPygletKey(symbol=self.responseButton, modifiers=0,
                            emulated=True)


class ActiveAutoPilot(PassiveAutoPilot):
    """Agent for the active phase of the experiment.

    """
    def __init__(self, mean_rt:float, sd_rt:float, active:bool = False,
                 mode:str = 'random', buttonLeft:str = 'left',
                 buttonRight:str = 'right', p_threshold:float=0.99,
                 n_presses:int = 1) -> None:
        """Active agent.

        Args:
            mean_rt (float): Mean of normal distribution
            sd_rt (float): SD of normal distribution.
            active (bool, optional): If the agent is active or not. Defaults to False.
            mode (str, optional): The mode the agent operates in. Defaults to 'random'.
            buttonLeft (str, optional): Response button for a left response. Defaults to 'left'.
            buttonRight (str, optional): Response button for a right response. Defaults to 'right'.

        Raises:
            ValueError: If mode is not in ['random', time_optimal', 'eta_-1', 'eta_0', 'eta_1']
        """

        self.mode = mode
        self.active = active
        self.buttonLeft = buttonLeft
        self.buttonRight = buttonRight
        self.wealth = None
        self.eta_agent = None
        self.p_threshold = p_threshold
        self.n_presses = n_presses

        if self.mode == 'random':
            self.decision_fun = self.random_agent
        elif self.mode == 'time_optimal':
            self.decision_fun = self.time_optimal_agent
        elif self.mode in ['eta_-1.0', 'eta_0.0', 'eta_1.0', 'eta_0.5']:
            self.decision_fun = self.dynamic_agent
        elif self.mode in ['soft_-1.0', 'soft_0.0', 'soft_1.0', 'soft_0.5']:
            self.decision_fun = self.softmax_agent
        else:
            raise ValueError("mode has to be one of: ['random', time_optimal',"
                            + "'eta_-1.0', 'eta_0.0', 'eta_1.0', 'soft_-1.0',"
                            + " 'soft_0.0', 'soft_1.0']")

        if self.mode in ['eta_-1.0', 'eta_0.0', 'eta_0.5', 'eta_1.0', 'soft_-1.0',
                         'soft_0.0', 'soft_1.0', 'soft_0.5']:
            self.eta_agent = float(self.mode.split('_')[-1])
            print(f'Agent decides according to eta = {self.eta_agent}')

        super().__init__(mean_rt, sd_rt, active)

    def start_timer(self, time:float, wealth:float,
                    gamblePairs:Union[np.ndarray, List[float]], eta:float):
        """Start timer for agent, sets RT and time after which response occurs.

        Args:
            time (float): Time from which to count down until response.
            wealth (float): Current wealth
            gamblePairs (Union[np.ndarray, List[float]]): np.ndarray or list of
                length four defining the two gamblepairs. The order is
                left-up, left-down, right-up, right-down.
            eta (float): The wealth dynamic of the experiment.
        """

        self.response_time = np.abs(self.sd_rt * np.random.randn() + self.mean_rt)
        self.start_time = time
        self.wealth = wealth
        self.press_time = self.start_time + self.response_time
        self.button = self.decision_fun(gamblePairs, eta, self.eta_agent)
        self.pressed = 0

    def press(self):
        """Ushers a key press, depending on whether it is a left or right response.
        """
        if self.pressed < self.n_presses:
            if self.button:
                event._onPygletKey(symbol=self.buttonLeft, modifiers=0,
                                    emulated=True)
            else:
                event._onPygletKey(symbol=self.buttonRight, modifiers=0,
                                    emulated=True)

            self.button = None

            self.pressed += 1

    def random_agent(self, gamblePairs:None=None, eta:None=None,
                     eta_agent:None=None) -> bool:
        """ Creates a random response.
        Args:
            gamblePairs (None, optional): Not used. Defaults to None.
            eta (None, optional): Not used. Defaults to None.
            eta_agent (None, optional): Not used. Defaults to None.

        Returns:
            bool: Which response key. True = Left, False = Right.
        """

        button = np.random.rand() < 0.5

        return button


    def time_optimal_agent(self, gamblePairs:Union[np.ndarray, List[float]],
                           eta:None=None, eta_agent:None=None) -> bool:
        """Agent creating time optimal reactions.

        Args:
            gamblePairs (Union[np.ndarray, List[float]]): np.ndarray or list of
                length four defining the two gamblepairs. The order is
                left-up, left-down, right-up, right-down.
            eta (None, optional): Not used. Defaults to None.
            eta_agent (None, optional): Not used. Defaults to None.

        Returns:
            bool: Which response key. True = Left, False = Right.
        """

        expectation_left = np.mean(gamblePairs[:2])
        expectation_right = np.mean(gamblePairs[2:])

        return expectation_left > expectation_right


    def dynamic_agent(self, gamblePairs:Union[np.ndarray, List[float]],
                      eta:float, eta_agent:float):
        """Agent creating responses after a specified dynamic.

        Args:
            gamblePairs (Union[np.ndarray, List[float]]): np.ndarray or list of
                length four defining the two gamblepairs. The order is
                left-up, left-down, right-up, right-down.
            eta (float): The wealth dynamic of the experiment.
            eta_agent (float): The dynamic the agent uses to decide.

        Returns:
            bool: Which response key. True = Left, False = Right.
        """
        wealth_updates = np.array([wealth_change(self.wealth, g, eta)
                                   for g in gamblePairs])

        agent_base = isoelastic_utility(self.wealth, eta_agent)

        agent_expectation = np.array([isoelastic_utility(wc, eta_agent) - agent_base
                                      for wc in wealth_updates])

        return np.mean(agent_expectation[:2]) > np.mean(agent_expectation[2:])


    def softmax_agent(self, gamblePairs:Union[np.ndarray, List[float]],
                      eta:float, eta_agent:float):
        """Agent creating responses after a specified dynamic.

        Args:
            gamblePairs (Union[np.ndarray, List[float]]): np.ndarray or list of
                length four defining the two gamblepairs. The order is
                left-up, left-down, right-up, right-down.
            eta (float): The wealth dynamic of the experiment.
            eta_agent (float): The dynamic the agent uses to decide.

        Returns:
            bool: Which response key. True = Left, False = Right.
        """

        c_eta = {0.0: 250, 1.0 : 0.4}
        beta =  calculate_beta(eta_dynamic=eta, eta_agent=eta_agent, x_0=con.X0,
                               x_limit=con.X_UPPER, p_threshold=self.p_threshold,
                c=c_eta(eta))

        wealth_updates = np.array([wealth_change(self.wealth, g, eta)
                                   for g in gamblePairs])

        agent_base = isoelastic_utility(self.wealth, eta_agent)

        agent_expectation = np.array([isoelastic_utility(wc, eta_agent) - agent_base
                                      for wc in wealth_updates])

        diff_sides = np.mean(agent_expectation[:2]) - np.mean(agent_expectation[2:])
        print(diff_sides)
        p_left = softmax(diff_sides, beta)
        print(p_left)
        return np.random.rand() < p_left
