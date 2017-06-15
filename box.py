# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:23:46 2016

@author: aschi
"""
import copy
from attrdict import AttrDict

from transport import Flow
from process import Process
from condition import Condition

from utils import dimensionality_check, dimensionality_check_err


class Box:
    """ Represents a box in a single- or multibox model.
    
    Boxes have a certain (initial) volume that can change if the outflow and 
    inflow are not equal. Additionally every box has specific processes that 
    take place and potentially alter the concentrations of containting 
    substances. Finally every box has specific (environmental) conditions like
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
                 processes=[], reactions=[] ):
        self.ID = None
        self.name = name
        self.name_long = name_long
        self.fluid = copy.copy(fluid)
        self.condition = condition if condition else Condition()
        
        self.variables = AttrDict({variable.name: variable for variable in variables})  # copy.deepcopy(variables)
        self.processes = copy.deepcopy(processes)
        self.reactions = copy.deepcopy(reactions)
        self.system = None
        self.surface = 0
        
        # Set the box attribute of the fluid, substances, and process in order
        # to give them access to the condition of the box
        self.fluid.box = self
        for var_name, var in self.variables.items():
            var.box = self
        for process in self.processes:
            process.box = self
        
        variable_names = [var_name for var_name, var in self.variables.items()]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError('Variable names have to be unique!')
      
    @property
    def volume(self):
        return self.fluid.volume
    
    @property
    def cond(self):
        return self.condition
    
    @property
    def var(self):
        return self.variables
        
    def info(self,):
        print(self.name)
        print('_____________________')
        print('Box Volume: {}'.format(self.volume))
        print()
                 
    def set_global_condition(self, global_condition):
        self.condition.set_superior_condition(global_condition)

    @property
    def context(self):
        """
        Returns a context for evaluating user-defined functions.
        
        box_context returns a context containing the box condition and 
        variables. In addition all boxes of the whole boxmodel-system are also 
        added to the context in order to give the user-defined function access 
        to condition and variables of other boxes.
        
        """
        context = self.condition
        for var_name, var in self.variables.items():
            setattr(context, var_name, var)
        for box_name, box in self.system.boxes.items():
            setattr(context, 'box_'+box.name, box)
        # Add current box to context as 'box1'
        setattr(context, 'box1', self)
        return context
    

    
    
            
            
