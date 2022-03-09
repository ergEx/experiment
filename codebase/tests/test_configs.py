import pytest
from ..experiment.configs import check_attribute_present, check_attribute_type


def test_attribute_present_replaced():
    """Checks if attribute is replaced.
    """

    configs = {}

    configs = check_attribute_present(configs, 'test', 'test')

    assert configs['test'] == 'test'


def test_attribute_present_error():
    """Checks if error is correctly raised.
    """
    configs = {}

    with pytest.raises(KeyError):
        configs = check_attribute_present(configs, 'test', None)


def test_attribute_type():
    """Checks if attribute is correctly set.
    """
    configs = {'test': 'test'}

    check_attribute_type(configs, 'test', str)

    check_attribute_type(configs, 'test', (str, int))


def test_attribute_type_error():
    """Check if error is correctly raised.
    """
    configs = {'test': 'test'}

    with pytest.raises(ValueError):
        configs = {'test': 'test'}
        check_attribute_type(configs, 'test', int)
