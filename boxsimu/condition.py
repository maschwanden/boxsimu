# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 13:42UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

The Condition class is used to group constant environmental parameters. 

"""

import copy
from keyword import iskeyword
from attrdict import AttrDict


class Condition(AttrDict):
    """Represent environmental parameters that are constant in time.

    Conditions (subclass of AttrDict) group environemental
    parameters of individual boxes or the whole BoxModelSystem.
    Conditions are constants. That means that they are assumed
    to not change in time and that their values must not be
    callables.

    Args:
        dict or **kwargs: keys must be valid python variable
            names, values must not be callables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            if not key.isidentifier() or iskeyword(key):
                raise ValueError('Keys must be valid python expressions.')

    def set_superset_condition(self, superset_condition):
        """Set all not defined parameters based on the superset condition.

        All condition parameters that are not set on the current
        condition but are defined in the superset_condition
        are set. If a key-value-condition is set on the current
        condition and the surrounding_condition the current
        key-value-condtion has priority (thus is not changed).

        Args:
            superset_condition (Condition): All condition parameters that are 
                not set on the current instance are filled with values from
                the superset_condition instance.        
        
        """
        for key, value in superset_condition.items():
            if not hasattr(self, key):
                setattr(self, key, value)
