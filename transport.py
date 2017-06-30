# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:57:31 2016

@author: aschi
"""

from scipy import integrate

from pint import UnitRegistry
ur = UnitRegistry(autoconvert_offset_to_baseunit = True)

import utils
import errors
import box
from action import BaseAction


class BaseTransport(BaseAction):
    """ Represents a basic transport of a substance or solvent.
    
    Attributes:
    - name (str): Name of the BaseTransport. Short descriptive text.
    - source_box (Box): 
    - target_box (Box): 
    - rate: 
    """
    
    def __init__(self, name, source_box, target_box, rate):
        self.name = name
        self.source_box = source_box
        self.target_box = target_box
        self.rate = rate
        
        self.units = None
    
    def __str__(self):
        return '<BaseTransport {}: {}>'.format(self.name, self.rate)
    
    @classmethod
    def get_all_from(cls, source, flows):
        """ Returns all instances that stem from source (source can either be a 
        box instance or None if the flow stems from outside the system).
        """        
        if source == None:
            return [f for f in flows if f.source_box==None]
        else:
            return [f for f in [fb for fb in flows if fb.source_box]
                    if f.source_box.name==source.name]

    @classmethod
    def get_all_to(cls, target, flows):
        """ Returns all instances that go to target (target can either be a 
        box instance or None if the flow goes out of the system).
        """  
        if target == None:
            return [f for f in flows if f.target_box==None]
        else:
            return [f for f in [fb for fb in flows if fb.target_box] 
                    if f.target_box.name==target.name]
    

class Flow(BaseTransport):
    """ Represents the flow of the medium from one box into another. 
    
    For the evaluation of the rate at runtime the following rules apply:
    - if the flow has a source box the context of this source box is used as context
    - if the flow has only a target box (therefore it is a flow from the outside of 
    the system into the system) the global context of the system is used as context!

    Attributes:
    - 
    """

    def __init__(self, name, source_box, target_box, rate, tracer_transport=True):
        if source_box == target_box:
            raise ValueError('target_box and source_box must not be equal!')
        
        if not (isinstance(source_box, box.Box) or isinstance(target_box, box.Box)):
            raise ValueError('At least one of the two parameters source_box'\
                    'and target_box must be an instance of the class Box!')
        if isinstance(source_box, box.Box) and isinstance(target_box, box.Box):
            if source_box.fluid != target_box.fluid:
                raise ValueError('target_box.fluid and source_box.fluid must be equal!')
        # specify whether tracers are transported with this flow
        # if set to true a flow from one box to another will also transport variables
        # (tracers) from the source box to the target box. If set to False only fluid
        # mass will be transported and no variable mass is removed from the source box.
        self.tracer_transport = tracer_transport 

        self.variables = []
        self.concentrations = {}
    
        super(Flow, self).__init__(name, source_box, target_box, rate)

    def __str__(self):
        return '<BaseTransport {}: {}>'.format(self.name, self.rate)
    
    def add_transported_variable(self, variable, concentration):
        """ Adds a variable to the flow: This variable is then transported into the
        target box. Only for Flows that have no source box!
        """

        if self.source_box:
            raise ValueError('Fixed variable concentrations can only be set for Flows with no '\
                    'source_box set (source_box=None)!')
        conc = concentration.to_base_units()
        utils.dimensionality_check_mass_dimless(conc)

        if not variable.quantified:
            raise errors.VariableNotQuantifiedError('Variable was not quantified!')
        self.variables.append(variable)
        self.concentrations[variable.name] = conc
             
        
class Flux(BaseTransport):
    """ Represents the transport process of a certain variable from one box 
    into another, or from outside the system into the system (or reverse).
    
    This transport can be a simple mass flux (e.g. sedimentation of particles) 
    or diffusion.
    
    Attributes:
    - name: Name of the transport process. Should be a short describtive text.
    - source_box: Instance of Box or None if the transport is coming from outside the system.
    - target_box: Instance of Box or None if the transport is going outside the system.
    - variable: Instance of Variable that represents the transported substance.
    - rate: Rate of the flux of the variable.
    """
    
    def __init__(self, name, source_box, target_box, variable, rate):
        self.name = name
        if source_box == target_box:
            raise ValueError('target_box and source_box must not be equal!')
        if not (isinstance(source_box, Box) and isinstance(target_box, Box)):
            raise ValueError('The parameters source_box and target_box must be'\
                             'instances of the class Box!')

        if not variable.quantified:
            raise errors.VariableNotQuantifiedError('Variable was not quantified!')
        self.variable = variable
        
        super(Flux, self).__init__(name, source_box, target_box, rate)






