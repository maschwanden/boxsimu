# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 10:36UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy

from . import dimensionality_validation as bs_dim_val
from . import entities as bs_entities
from . import errors as bs_errors


class BaseProcess:
    """Base Class for Process and Reaction."""
    def __str__(self):
        return '<BaseProcess {}>'.format(self.name)

    def __hash__(self):
        return hash(repr(self))
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.name > other.name
        return False

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.name < other.name
        return False


class Process(BaseProcess):
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

    def __call__(self, time, context):
        """Return rate of the process [M/T]."""
        rate = self.rate  # Default rate
        if callable(self.rate):
            rate = self.rate(time, context)
        if bs_dim_val.is_mole_per_time(rate):
            if self.variable.molar_mass:
                rate = rate * self.variable.molar_mass
        return rate.to_base_units()

    @classmethod
    def get_all_of_variable(cls, variable, processes):
        return [p for p in processes if p.variable == variable]

    @classmethod
    def get_all_of_box(cls, box, processes):
        return [p for p in processes if p.box == box]


class Reaction(BaseProcess):
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
        self.variable_reaction_coefficients = variable_reaction_coefficients
        
        self.variables = []
        for variable, coeff in variable_reaction_coefficients.items():
            self.variables.append(variable)

        self.rate = rate

    def __call__(self, time, context, variable):
        """Return rate of the variable transformation [M/T]."""
        if not isinstance(variable, bs_entities.Variable):
            raise ValueError('{} is not a Variable!'.format(variable))
        if variable not in self.variables:
            var_coeff = 0
        else:
            var_coeff = self.variable_reaction_coefficients[variable]

        rate = self.rate  # Default rate
        if callable(self.rate):
            rate = self.rate(time, context)
        if bs_dim_val.is_mole_per_time(rate):
            if self.variable.molar_mass:
                rate = rate * self.variable.molar_mass
        rate = rate.to_base_units()
        return var_coeff * rate

    def get_reverse_reaction(self, name, rate=None):
        """Return the reverse reaction of the instance.
        
        For example: If a reaction Photosynthesis is defined, the reverse 
        reaction (remineralization) can be obtained by the following call:
            photosynthesis.get_reverse_reaction(name='remineralization',
                    rate=...)
        
        """
        reverse_reaction = copy.deepcopy(self)
        reverse_reaction.name = name
        for variable, coeff in self.variable_reaction_coefficients.items():
            reverse_reaction.variable_reaction_coefficients[variable] = -coeff
        if rate:
            reverse_reaction.rate = rate
        return reverse_reaction

