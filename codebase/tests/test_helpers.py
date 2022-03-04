import numpy as np
import pytest
from ..experiment.exp.helper import assign_fractals, format_wealth


def test_assign_fractals_xnumber():
    # Test if an xnumber can be extracted.
    for eta in [-1, -0.5, 0, 0.5, 1]:
        assign_fractals("X00123", eta, path='stimuli/')


def test_assign_fractals_other():
    # Test if an xnumber can be extracted.
    for eta in [-1, -0.5, 0, 0.5, 1]:
        assign_fractals("abcd", eta, path='stimuli/')
        assign_fractals("dge3", eta, path='stimuli/')
        assign_fractals("aet123", eta, path='stimuli/')


def test_assign_fractals_etastring():
    # Test if an xnumber can be extracted.
    for eta in ['-1.0', '-0.5', '0.0', '0.5', '1.0']:
        assign_fractals("abcd", eta, path='stimuli/')
        assign_fractals("dge3", eta, path='stimuli/')
        assign_fractals("aet123", eta, path='stimuli/')
        assign_fractals("aet1123", eta, path='stimuli/')


def test_raise_eta_error():
    # Test if an xnumber can be extracted.
    for eta in ['-1', '-.5', '0', '1.4']:

        with pytest.raises(ValueError):
            assign_fractals("abcd", eta, path='stimuli/')


def test_format_wealth_default():

    assert format_wealth(1_000) == '001,000'
    assert format_wealth(10_000) == '010,000'
    assert format_wealth(10_000_000) == '10,000,000'


def test_format_wealth_float():

    assert format_wealth(1_000, '8.2f') == ' 1000.00'
    assert format_wealth(10_000, '14.2f') == '      10000.00'
    assert format_wealth(10_000_000, '14.2f') == '   10000000.00'
