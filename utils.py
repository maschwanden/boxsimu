# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 08:42:29 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import re
import numpy as np


def OneDarray_to_TwoDarray_method(vector_method):
    """Return method that returns 2Darray from 1Darrays from vector_method."""
    def _2Darray_method(self, *args):
        tmp_vector_list = [0] * self.N_variables
        for variable_name, variable in self.variables.items():
            vector = vector_method(self, variable, *args, numpy_array=True)
            tmp_vector_list[variable.id] = vector
        return np.array(tmp_vector_list).T
    return _2Darray_method


def dot(a, b):
    """Calculate dot product of numpy arrays of pint Quantities.
    
    Calculat the dot product of two numpy arrays of pint Quantities while
    the units are conserved.

    Note: If numpy.dot is used the resulting array has no units anymore.

    """
    a_units = 1
    b_units = 1
    a_vec = a
    b_vec = b
    try:
        a_units = a.units
        a_vec = a.magnitude
    except AttributeError:
        pass
    try:
        b_units = b.units
        b_vec = b.magnitude
    except AttributeError:
        pass
    return np.dot(a_vec, b_vec) * a_units * b_units


def get_valid_filename_from_string(string): 
    tmp_str = str(string).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', tmp_str)        


def get_valid_svg_id_from_string(string):
    tmp_str = str(string).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', tmp_str)        



