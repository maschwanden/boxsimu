# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:23:46 2016

@author: aschi
"""

import copy
from attrdict import AttrDict


class Condition(AttrDict):
    """ Simple class that's used for grouping all environmental parameters of
    the whole system or individual boxes. 
    
    Conditions are constants.
    """

    def set_superior_condition(self, superior_condition):
        """ Set the superior condition.
        
        For a box where the conditions are set to "cond1" this method can be 
        used update the local condition by setting all attributes of "cond1" 
        that are set by the superior but not the (local" "cond1".
        """
        for key, value in superior_condition.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        
