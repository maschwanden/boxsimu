# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:23:46 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
from keyword import iskeyword
from attrdict import AttrDict


class Condition(AttrDict):
    """Represent environmental parameters.

    Conditions (subclass of AttrDict) group environemental
    parameters of individual boxes or the whole BoxModelSystem.
    Conditions are constants. That means that they are assumed
    to not change with time and that their values must not be
    callables.

    Args:
        dict or **kwargs: keys must be valid python variable
            names, values must not be callables.
    """

    def __init__(self, *args, **kwargs):
        super(Condition, self).__init__(*args, **kwargs)

        for key, value in self.items():
            if not key.isidentifier() or iskeyword(key):
                raise ValueError('Keys must be valid python expressions.')

    def set_surrounding_condition(self, surrounding_condition):
        """Set all not defined parameters based on the surrounding condition.

        All key-value-conditions that are not set on the current
        condition but are defined in the surrounding_condition
        are set. If a key-value-condition is set on the current
        condition and the surrounding_condition the current
        key-value-condtion has priority (thus is not changed).
        """
        for key, value in surrounding_condition.items():
            if not hasattr(self, key):
                setattr(self, key, value)
