# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 08:42:29 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import numpy as np


def _1Darray_to_2Darray_method(vector_method):
    """Return method that returns 2Darray from 1Darrays from vector_method."""
    def _2Darray_method(self, *args):
        tmp_vector_list = [0] * self.N_variables
        for variable_name, variable in self.variables.items():
            vector = vector_method(self, variable, *args, numpy_array=True)
            tmp_vector_list[variable.id] = vector
        return np.array(tmp_vector_list).T
    return _2Darray_method


def np_pint_dot(a, b):
    """Calculate dot product of numpy arrays of pint Quantities.
    
    Calculat the dot product of two numpy arrays of pint Quantities while
    the units are conserved.

    Note: If numpy.dot is used the resulting array has no units anymore.

    """
    a_units = a.units
    b_units = b.units

    return np.dot(a.magnitude, b.magnitude) * a_units * b_units



