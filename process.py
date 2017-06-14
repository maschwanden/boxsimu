# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:36:37 2016

@author: aschi
"""

from utils import (dimensionality_check, dimensionality_check_err,
                   magnitude_in_current_units, magnitude_in_base_units)
from action import BaseAction


class Process(BaseAction):
    """ Represents a process.
    
    An internal process can be a production, consumption, destruction, etc. 
    of a given variable.
    
    Attributes:
    - name: Name of the internal process. Should be a short describtive text.
    - box: Box in which the process takes place.
    - variable: Instance of Variable that represents the processed substance/entity.
    - rate: rate at which the substance is processed.
    """
    
    def __init__(self, name, variable, rate):
        self.name = name
        self.variable = variable
        self.rate = rate
        
        self.box = None
        self.units = None
    
    
    @classmethod
    def get_all_of_variable(cls, variable, processes):
        return  [p for p in processes if p.variable==variable]


    @classmethod
    def get_all_of_box(cls, box, processes):
        return  [p for p in processes if p.box==box]


class Reaction:
    """ Represents a reaction that transforms variable-mass into other variable-mass.

    A reaction can for example be: Photosynthesis (carbon + nitrate + phosphate -> phytoplankton mass),
    or Remineralization (phytoplankton mass -> carbon + nitrate + phsopate), or any other transformation
    of variables.

    Attributes:
    - name: Name of the Reaction. Should be a short describitive text.
    - box: Box in which the reaction takes place.
    - variables: Instance of Variable that represents the processed substance/entity.
    - variable_coeffs: List of (mass-based)Coefficents for the (mass-based)Reaction, must be given in 
                       the same same order as the variables
    - reaction_rate: Reaction rate must have units of [1/T] 
    """

    def __init__(self, name, variables, variable_coeffs, reactions_rate):
        self.name = name
        self.variables = variables
        self.variable_coeffs = variable_coeffs
        self.reactions_rate = reactions_rate
        self.box = None
    
    @classmethod
    def get_all_of_box(cls, box, reactions):
        return  [l for r in reactions if r.box==box]

    
