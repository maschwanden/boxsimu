
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:51:29 2017

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""


class BoxSimuBaseException(Exception):
    """Base Error for Boxsimu Project."""
    pass


class VariableNotQuantifiedError(BoxSimuBaseException):
    """Raise if Variable is not quantified but should be."""
    pass


class FluidNotQuantifiedError(BoxSimuBaseException):
    """Raise if Fluid is not quantified but should be."""
    pass
