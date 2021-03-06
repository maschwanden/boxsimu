# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 10:36UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
import numpy as np

from . import validation as bs_validation
from . import descriptors as bs_descriptors
from . import entities as bs_entities
from . import errors as bs_errors
from . import function as bs_function
from . import ur


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

    name = bs_descriptors.ImmutableIdentifierDescriptor('name')
    rate = bs_descriptors.PintQuantityDescriptor('rate', 
            ur.kg/ur.second, 0*ur.kg/ur.second)

    def __init__(self, name, variable, rate, description=None):
        self.name = name
        self.variable = variable
        self.rate = bs_function.UserFunction(rate, ur.kg/ur.second)
        if not description:
            self.description = name

    def __call__(self, time, context):
        """Return rate of the process [M/T]."""
        return self.rate(time, context, system)

    @classmethod
    def get_all_of_variable(cls, variable, processes):
        """Return all processes defined for variable."""
        return [p for p in processes if p.variable == variable]

    @classmethod
    def get_all_of_box(cls, box, processes):
        """Return all processes defined for box."""
        return [p for p in processes if p.box == box]


class Reaction(BaseProcess):
    """Transform masses of different Variables into each other.

    A reaction can for example be:
    - Photosynthesis (carbon + nitrate + phosphate -> phytoplankton mass)
    - Remineralization (phytoplankton mass -> carbon + nitrate + phsopate)
    - Any other transformation of variables defined in the box/system.

    Args:
        name (str): Human readable string describing the reaction.
        reaction_coefficients (dict {Variable: float}): 
            Dict with instances of Variable as keys and the corresponding
            reaction coefficients as values.
            E.g. : A + 3B -> 2C + 4D 
            -> reaction_coefficients={A: -1, B: -3, C: 2, D: 4}
        rate (pint.Quantity or callable that returns pint.Quantity): Rate [M/T] 
            at which the a variable with a reaction coefficient of 1 (one) 
            reacts. Note: Must have dimensions of [M/T].
            E.g. : A variable has a reaction coefficient of 3, the rate of mass
            transformed of this variable is: reaction_rate * 3.

    Attributes:
        name (str): Human readable string describing the reaction.
        reaction_coefficients (dict {Variable: float}): 
            Dict with instances of Variable as keys and the corresponding
            reaction coefficients as values.
            E.g. : A + 3B -> 2C + 4D 
            -> reaction_coefficients={A: -1, B: -3, C: 2, D: 4}
        variables (list of Variable): Variables that react in this Reaction.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            [M/T] at which the a variable with a reaction coefficient of 1 
            (one) reacts. Note: Must have dimensions of [M/T].
            E.g. : A variable has a reaction coefficient of 3, the rate of mass
            transformed of this variable is: reaction_rate * 3.

    """

    name = bs_descriptors.ImmutableIdentifierDescriptor('name')

    def __init__(self, name, reaction_coefficients, rate, description=None):
        self.name = name
        for var, coeff in reaction_coefficients.items():
            if not isinstance(var, bs_entities.Variable):
                raise bs_errors.DictKeyNotInstanceOfError(
                        'reaction_coefficients', 'Variable')
            if not (isinstance(coeff, int) or isinstance(coeff, float)):  
                raise bs_errors.DictValueNotInstanceOfError(
                        'reaction_coefficients', 'int or float')
        self.reaction_coefficients = reaction_coefficients
        
        self.variables = []
        for variable, coeff in reaction_coefficients.items():
            self.variables.append(variable)
        self.rate = bs_function.UserFunction(rate, ur.kg/ur.second)
        if not description:
            self.description = name

    def __call__(self, time, context, system, variables):
        """Return reaction rates of all variables [M/T].
        
        Args:
            time (pint.Quantity [T]): Time of the simulation.
            context (AttrDict): Condition and Variables of the Box/Flow/Flux.
            system (BoxModelSystem): System that is solved. Allows the user
                to access all Variables in all Boxes of the system.
            variables (list of Variable): Variables for which the reaction
                rates should be returned.
            
        """
        rate = self.rate(time, context, system)
        var_coeff_list = []
        for variable in variables:
            if variable in self.variables:
                var_coeff_list.append(
                        self.reaction_coefficients[variable])
            else:
                var_coeff_list.append(0)
        var_coeffs = np.array(var_coeff_list)
        return var_coeffs * rate

    def get_reverse_reaction(self, name, rate=None, description=None):
        """Return the reverse reaction of the instance.
        
        For example: If a reaction Photosynthesis is defined, the reverse 
        reaction (remineralization) can be obtained by the following call:
            photosynthesis.get_reverse_reaction(name='remineralization',
                    rate=...)

        Args:
            name (str): Human readable string describing the reaction.
            description (str): Human readable string describing the box.   
            rate (pint.Quantity or callable that returns pint.Quantity): Rate 
                [M/T] at which the a variable with a reaction coefficient of 1 
                (one) reacts. Note: Must have dimensions of [M/T].
                E.g. : A variable has a reaction coefficient of 3, the rate of 
                mass transformed of this variable is: reaction_rate * 3.
        
        """
        reverse_reaction_coefficients = {variable: -coeff 
                for variable, coeff in self.reaction_coefficients.items()}
        reverse_reaction_rate = rate if rate else self.rate
        return Reaction(name, reverse_reaction_coefficients,
                reverse_reaction_rate, description=description)

    def get_reaction_text(self):
        """Return the reaction as text. 

        The reaction is printed in the following format:
        <coeff 1><var 1> + <coeff 2><var 2> + ... -> <coeff i><var i> + ...

        """
        educt_str_list = []
        product_str_list = []
        for variable, coeff in self.reaction_coefficients.items():
            if coeff > 0:
                product_str_list.append('{}{}'.format(abs(coeff), 
                    variable.name))
            else:
                educt_str_list.append('{}{}'.format(abs(coeff), 
                    variable.name))

        reaction_text = ' + '.join(educt_str_list) + ' -> ' + ' + '.join(
                product_str_list)
        return reaction_text




