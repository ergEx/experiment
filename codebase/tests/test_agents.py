from ..experiment.exp import ActiveAutoPilot
import numpy as np
# Class could be tested via fixtures, not sure how though^^

def test_AAP_timing():
    """Simple test of active auto pilot if times are created correctly.
    """
    AAP = ActiveAutoPilot(1, 0, active=True)
    AAP.start_timer(1, 0, None, None)
    assert AAP.press_time == 2
    assert AAP.start_time == 1


def test_AAP_decision_function_random():
    '''
    '''
    AAP = ActiveAutoPilot(1, 0, active=True, mode='random')
    presses = [AAP.random_agent() for _ in range(1000)]
    # Does it make sense?
    assert np.isclose(np.mean(presses), 0.5, rtol=0.1, atol=0.1)
    assert np.max(presses) == 1
    assert np.min(presses) == 0
    assert len(presses) == 1000


def test_APP_time_optimal_decision():
    AAP = ActiveAutoPilot(1, 0, active=True, mode='time_optimal')

    # Right > left
    assert AAP.time_optimal_agent(np.array([1, 1, 2, 2])) == 0
    # Left > right
    assert AAP.time_optimal_agent(np.array([2, 2, 1, 1])) == 1
    # Equal size, put out right
    assert AAP.time_optimal_agent(np.array([2, 2, 2, 2])) == 0


def test_APP_dynamic_agent_0():
    AAP = ActiveAutoPilot(1, 0, active=True, mode='eta_0.0')
    assert AAP.eta_agent == 0
    # Right > left
    assert AAP.time_optimal_agent(np.array([1, 1, 2, 2])) == 0
    # Left > right
    assert AAP.time_optimal_agent(np.array([2, 2, 1, 1])) == 1
    # Equal size, put out right
    assert AAP.time_optimal_agent(np.array([2, 2, 2, 2])) == 0


def test_APP_dynamic_agent_1():
    AAP = ActiveAutoPilot(1, 0, active=True, mode='eta_1.0')
    assert AAP.eta_agent == 1
    # Right > left
    assert AAP.time_optimal_agent(np.array([1, 1, 2, 2])) == 0
    # Left > right
    assert AAP.time_optimal_agent(np.array([2, 2, 1, 1])) == 1
    # Equal size, put out right
    assert AAP.time_optimal_agent(np.array([2, 2, 2, 2])) == 0


def test_APP_dynamic_agent_m1():
    AAP = ActiveAutoPilot(1, 0, active=True, mode='eta_-1.0')
    assert AAP.eta_agent == -1
    # Right > left
    assert AAP.time_optimal_agent(np.array([1, 1, 2, 2])) == 0
    # Left > right
    assert AAP.time_optimal_agent(np.array([2, 2, 1, 1])) == 1
    # Equal size, put out right
    assert AAP.time_optimal_agent(np.array([2, 2, 2, 2])) == 0