# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:23:46 2016

@author: aschi
"""
import copy
from keyword import iskeyword
from attrdict import AttrDict

import errors
from transport import Flow
from process import Process
from condition import Condition


class Box:
    """ Represents a box in a single- or multibox model.
    
    The mass and volume of a box is defined by the fluid it contains. Fluids have
    a certain (initial) mass that can change if the outflow and inflow of the box 
    are not equal. Additionally every box has specific processes that 
    take place and potentially alter the concentrations of containting 
    variables. Finally every box has specific (environmental) conditions like
    temperature etc. that are specific for this box (in contrast to "global 
    conditions" that are default for every box that doesnt overwrites a global 
    condtion).
    
    Attributes:
    - name: Name of the box. Should use python synthax for variables.
    - name_long: Long name of the box. Short description of the box.
    - fluid: Fluid within this Box.  
    - processes: Instances of the class Process that describe how certain 
                 substances within the Box are transformed, consumed, or 
                 produced.
    - condition: Instance of the class Condition that represents the (physical,
                 chemical, biological) conditions within the modelled Box.
    - substances: Substances that should be analysed.
    - system: System to which this Box belongs.
    """

    def __init__(self, name, name_long, fluid, condition=None, variables=[], 
                 processes=[], reactions=[]):
        self.ID = None

        if not name.isidentifier() or iskeyword(name):
            raise ValueError('Name must be a valid python variable name!')

        self.name = name
        self.name_long = name_long

        if not fluid.quantified:
            raise errors.FluidNotQuantifiedError('Fluid was not quantified!')
        self.fluid = copy.deepcopy(fluid)
        self.condition = copy.deepcopy(condition) or Condition()
        self.processes = copy.deepcopy(processes)
        self.reactions = copy.deepcopy(reactions)
        self.variables = AttrDict()

        for variable in variables:
            if not variable.quantified:
                raise errors.VariableNotQuantifiedError('Variable was not quantified!')
            self.variables[variable.name] = variable
        
        # Not needed now.
        # self.surface = 0
        
        variable_names = [var_name for var_name, var in self.variables.items()]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError('Variable names have to be unique!')

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    @property
    def mass(self):
        return self.fluid.mass + sum([var.mass for var in self.variables])
     
    @property
    def cond(self):
        return self.condition
    
    @property
    def var(self):
        return self.variables

    def get_volume(self, time=None, context=None):
        return self.fluid.get_volume(time, context)

        
            
            
