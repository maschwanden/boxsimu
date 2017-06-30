# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:36:37 2016

@author: aschi
"""

from action import BaseAction


class Process(BaseAction):
    """ Represents a process.
    
    An internal process can be a production or destruction of a variable.
    
    Attributes:
    - name (str): Name of the internal process. Should be a short describtive text.
    - box (Box): Box in which the process takes place.
    - variable (Variable): Instance of Variable that represents the processed substance/entity.
    - rate (pint Quantity or callable that returns pint Quantity): rate at which 
    the substance is processed.
    """
    
    def __init__(self, name, variable, rate):
        self.name = name
        if not variable.quantified:
            raise ValueError('Variables were not quantified until now!'\
                    'Use the method q() on the instance to quantify the variable!')
        self.variable = variable
        self.rate = rate
        
        self.units = None
    
    @classmethod
    def get_all_of_variable(cls, variable, processes):
        return  [p for p in processes if p.variable==variable]

    @classmethod
    def get_all_of_box(cls, box, processes):
        return  [p for p in processes if p.box==box]


class Reaction(BaseAction):
    """ Represents a reaction that transforms variable-masses.

    A reaction can for example be: 
    - Photosynthesis (carbon + nitrate + phosphate -> phytoplankton mass)
    - Remineralization (phytoplankton mass -> carbon + nitrate + phsopate)
    - Any other transformation of variables defined in the box/system.

    Attributes:
    - name (str): Name of the Reaction. Short descriptive text.
    - box (Box): Box in which the reaction takes place.
    - variable_reaction_coeff_dict (dict): 
        Dict with instances of Variable as keys and the corresponding
        reaction coefficients as values. 
        E.g. : A + 3B -> 2C + 4D -> variable_reaction_coeff_dict={A: -1, B: -3, C: 2, D: 4}
    - rate (pint Quantity or callable that returns pint Quantity): 
    Reaction rate must have units of [M/T]. The reaciton rates 
    defines the mass that is transformed of a variable with a var_coeff 
    of 1. This means if a variable has a var_coeff of 3, the rate of mass 
    transformed of this variable is: reaction_rate * 3. 
    """

    def __init__(self, name, variable_reaction_coeff_dict, rate):
        self.name = name
        self.variables = []
        self.variable_coeffs = []
        self.variable_reaction_coeff_dict = copy.deepcopy(variable_reaction_coeff_dict)
        
        for variable, coeff in variable_reaction_coeff_dict.items():
            if not variable.quantified:
                raise errors.VariableNotQuantifiedError('Variable was not quantified!')
            self.variables.append(variable)
            self.variable_coeffs.append(coeff)

        self.rate = rate

        self.units = None

    def __call__(self, time, context, variable):
        rate_func = self.get_rate_function_of_variable(variable)
        rate = rate_func(time, context)
        if not self.units:
            self.units = rate.units
        return rate.to_base_units()
    
    def get_coeff_of(self, variable):
        if not variable in self.variables:
            raise ValueError('{} is not an instance of Variable!'.format(variable))
        for i, var in enumerate(self.variables):
            if var == variable:
                return self.variable_coeffs[i]
    
    def get_rate_function_of_variable(self, variable):
        coeff = self.get_coeff_of(variable)
        def _rate(t, c):
            rr = self.reactions_rate
            if callable(self.reactions_rate):
                rr = self.reactions_rate(t, c)
            return  rr * coeff
        return _rate

