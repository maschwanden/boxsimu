# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:36:37 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy

from . import action as bs_action
from . import errors as bs_errors


class Process(bs_action.BaseAction):
    """Destruction or creation of variable with a specified (dynamic) rate.

    An internal process can be a production or destruction of a variable.

    Args:
        name (str): Human readable string describing the process.
        variable (Variable): Variable that is processed.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the substance is processed. Note: Must have dimensions of
            [M/T].

    Attributes:
        name (str): Human readable string describing the process.
        variable (Variable): Variable that is processed.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is processed. Note: Must have dimensions of
            [M/T].
    """

    def __init__(self, name, variable, rate):
        self.name = name
        self.variable = variable
        self.rate = rate

        self.units = None

    @classmethod
    def get_all_of_variable(cls, variable, processes):
        return [p for p in processes if p.variable == variable]

    @classmethod
    def get_all_of_box(cls, box, processes):
        return [p for p in processes if p.box == box]


class Reaction(bs_action.BaseAction):
    """Transform masses of different Variables into each other.

    A reaction can for example be:
    - Photosynthesis (carbon + nitrate + phosphate -> phytoplankton mass)
    - Remineralization (phytoplankton mass -> carbon + nitrate + phsopate)
    - Any other transformation of variables defined in the box/system.

    Args:
        name (str): Human readable string describing the reaction.
        variable_reaction_coefficients (dict {Variable: float}): 
            Dict with instances of Variable as keys and the corresponding
            reaction coefficients as values.
            E.g. : A + 3B -> 2C + 4D 
            -> variable_reaction_coefficients={A: -1, B: -3, C: 2, D: 4}
        rate (pint.Quantity or callable that returns pint.Quantity): Rate [M/T] 
            at which the a variable with a reaction coefficient of 1 (one) 
            reacts. Note: Must have dimensions of [M/T].
            E.g. : A variable has a reaction coefficient of 3, the rate of mass
            transformed of this variable is: reaction_rate * 3.

    Attributes:
        name (str): Human readable string describing the reaction.
        variable_reaction_coefficients (dict {Variable: float}): 
            Dict with instances of Variable as keys and the corresponding
            reaction coefficients as values.
            E.g. : A + 3B -> 2C + 4D 
            -> variable_reaction_coefficients={A: -1, B: -3, C: 2, D: 4}
        variables (list of Variable): Variables that react in this Reaction.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate [M/T] 
            at which the a variable with a reaction coefficient of 1 (one) 
            reacts. Note: Must have dimensions of [M/T].
            E.g. : A variable has a reaction coefficient of 3, the rate of mass
            transformed of this variable is: reaction_rate * 3.

    """

    def __init__(self, name, variable_reaction_coefficients, rate):
        self.name = name
        self.variables = []
        self.variable_reaction_coefficients = variable_reaction_coefficients
        
        for variable, coeff in variable_reaction_coefficients.items():
            self.variables.append(variable)

        self.rate = rate
        self.units = None

    def __call__(self, time, context, variable):
        rate_func = self.get_rate_function_of_variable(variable)
        rate = rate_func(time, context)
        if not self.units:
            self.units = rate.units
        return rate.to_base_units()

    def get_coeff_of(self, variable):
        """Return the reaction coefficient of variable."""
        if variable not in self.variables:
            raise ValueError('{} is not a Variable!'.format(variable))
        for i, var in enumerate(self.variables):
            if var == variable:
                return self.variable_coeffs[i]

    def get_rate_function_of_variable(self, variable):
        """Return a rate function with which a variable reacts."""
        coeff = self.get_coeff_of(variable)

        def _rate_func(t, c):
            rr = self.rate
            if callable(self.rate):
                rr = self.rate(t, c)
            return rr * coeff
        return _rate_func

