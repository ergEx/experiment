from ..utils import inverse_isoelastic_utility, isoelastic_utility
from ..utils import wealth_change, shuffle_along_axis
import numpy as np


def test_inverse_isoelastic_utility_inputs():
    """Test for scalar and array inputs
    """
    inverse_isoelastic_utility(1, 1)
    inverse_isoelastic_utility(np.ones((3,1)), 1)


def test_isoelastic_utility_inputs():
    """Test for scalar and array inputs
    """
    isoelastic_utility(1, 1)
    isoelastic_utility(np.ones((3,1)), 1)


def test_additive_wealth_change():
    """Test for additive wealth change
    """

    wealth = wealth_change(1000, 100, 0)
    assert wealth == 1100


def test_multiplicative_wealth_change():
    """Test for multiplicative wealth change
    """

    wealth = wealth_change(1000, np.log(1.1), 1)
    assert np.isclose(wealth, 1100)


def test_shuffle_along_axis_0():
    """Test to shuffle along axis 0, THIS FAILS!
    """

    test_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    shuff_array = shuffle_along_axis(test_array, axis=0)

    assert np.isin(test_array[:, 0], shuff_array[:, 0]).all()
    assert np.isin(test_array[:, 1], shuff_array[:, 1]).all()


def test_shuffle_along_axis_1():
    """Test to shuffle along axis 1
    """

    test_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    shuff_array = shuffle_along_axis(test_array, axis=1)

    assert np.isin(test_array[0, :], shuff_array[0, :]).all()
    assert np.isin(test_array[1, :], shuff_array[1, :]).all()


def test_shuffle_along_axis_0_w_concat():
    """Test to shuffle along axis 0, THIS FAILS!
    """

    test1 = np.array([1, 2])
    test2 = np.array([3, 4])
    test_array = np.concatenate((test1[np.newaxis], test2[np.newaxis]), axis=0)
    shuff_array = shuffle_along_axis(test_array, axis=0)

    assert np.isin(test_array[:, 0], shuff_array[:, 0]).all()
    assert np.isin(test_array[:, 1], shuff_array[:, 1]).all()


def test_shuffle_along_axis_1_w_concat():
    """Test to shuffle along axis 1
    """

    test1 = np.array([1, 2])
    test2 = np.array([3, 4])
    test_array = np.concatenate((test1[np.newaxis], test2[np.newaxis]), axis=0)
    shuff_array = shuffle_along_axis(test_array, axis=1)

    assert np.isin(test_array[0, :], shuff_array[0, :]).all()
    assert np.isin(test_array[1, :], shuff_array[1, :]).all()



def test_suffle_along_axis_3d_0():
    """Test to 3d along axis1"""

    test_array = np.arange(24).reshape(2, 3, 4)
    shuff_array = shuffle_along_axis(test_array, axis=0)

    # Assert along 2nd axis
    for idx in range(3):
        assert np.isin(test_array[:, idx, :], shuff_array[:, idx, :]).all()
    # Assert along 3rd axis
    for idx in range(4):
        assert np.isin(test_array[:, :, idx], shuff_array[:, :, idx]).all()


def test_suffle_along_axis_3d_1():
    """Test to 3d along axis1"""

    test_array = np.arange(24).reshape(2, 3, 4)
    shuff_array = shuffle_along_axis(test_array, axis=1)

    # Assert along 1st axis
    for idx in range(2):
        assert np.isin(test_array[idx, :, :], shuff_array[idx, :, :]).all()
    # Assert along 3rd axis
    for idx in range(4):
        assert np.isin(test_array[:, :, idx], shuff_array[:, :, idx]).all()


def test_suffle_along_axis_3d_2():
    """Test to 3d along axis1"""

    test_array = np.arange(24).reshape(2, 3, 4)
    shuff_array = shuffle_along_axis(test_array, axis=2)

    # Assert along 1st axis
    for idx in range(2):
        assert np.isin(test_array[idx, :, :], shuff_array[idx, :, :]).all()
    # Assert along 2nd axis
    for idx in range(3):
        assert np.isin(test_array[:, idx, :], shuff_array[:, idx, :]).all()

"""
def test_create_gamble_pairs():
    # From:
    # https://stackoverflow.com/questions/14766194/testing-whether-a-numpy-array-contains-a-given-row
    gambles = [np.array([0, 1]), np.array([2, 3]),
               np.array([4, 5]), np.array([6, 7]),
               np.array([8, 9]), np.array([10, 11])]

    gamble_pairs = create_gamble_pairs(gambles)
    for gps in gamble_pairs:
        for ga in range(len(gps)):
            print(ga, gps[ga, :])
            # Test whether gamble_pairs are a row in gambles
            assert any((gambles[:] == gps[ga, :]).all(1))
"""