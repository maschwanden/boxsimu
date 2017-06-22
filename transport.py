# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:57:31 2016

@author: aschi
"""

from scipy import integrate

from pint import UnitRegistry
ur = UnitRegistry(autoconvert_offset_to_baseunit = True)

from utils import (dimensionality_check, dimensionality_check_err, 
                   dimensionality_check_volume_flux, 
                   dimensionality_check_mass_flux,
                   dimensionality_check_mass_flux_err,
                   dimensionality_check_mass_dimless,
                   magnitude_in_current_units, magnitude_in_base_units)
from action import BaseAction


class BaseTransport(BaseAction):
    """ Represents a basic transport of a substance or solvent."""
    
    def __init__(self, name, source_box, target_box, rate):
        self.name = name
        self.source_box = source_box
        self.target_box = target_box
        self.rate = rate
        
        self.units = None
    
    def __call__(self, *args, **kwargs):
        rate = self.rate
        if callable(rate):
            rate = rate(*args, **kwargs)
        if not self.units:
            self.units = rate.units
        return rate
    
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
    """ Represents the flow of the medium from one box into another. """

    def __init__(self, name, source_box, target_box, rate, tracer_transport=True):
        super(Flow, self).__init__(name, source_box, target_box, rate)

        # specify whether tracers are transported with this flow
        # if set to true a flow from one box to another will also transport variables
        # (tracers) from the source box to the target box. If set to False only fluid
        # mass will be transported and no variable mass is removed from the source box.
        self.tracer_transport = tracer_transport 

        self.variables = []
        self.concentrations = {}
    
    def __str__(self):
        return '<BaseTransport {}: {}>'.format(self.name, self.rate)
    
    def mass_flow_rate(self, time, context):
        """ Calculates the total volume flow rate depnding on time, and context.
        """
        flow = self.__call__(time, context)
        
        if dimensionality_check_volume_flux(flow):
            if self.source_box:
                flow = flow*self.source_box.fluid.rho
            else:
                flow = flow*self.target_box.fluid.rho
        
        flow_rate = flow.to_base_units()
        dimensionality_check_mass_flux_err(flow_rate) 
        return flow_rate

    def add_transported_variable(self, variable, concentration):
        """ Adds a variable to the flow: This variable is then transported into the
        target box. Only for Flows that have no source box!
        """
        if self.source_box:
            raise ValueError('Fixed variable concentrations can only be set for Flows with no '\
                    'source_box set (source_box=None)!')
        dimensionality_check_mass_dimless(concentration)
        self.variables.append(variable)
        self.concentrations[variable.name] = concentration
             
        
class Flux(BaseTransport):
    """ Represents the transport process of a certain variable from one box 
    into another, or from outside the system into the system (or reverse).
    
    This transport can be a simple mass flux (e.g. sedimentation of particles) 
    or diffusion.
    
    Attributes:
    - name: Name of the transport process. Should be a short describtive text.
    - variable: Instance of Variable that represents the transported substance.
    - source_box: Instance of Box or None if the transport is coming from outside the system.
    - target_box: Instance of Box or None if the transport is going outside the system.
    - rate: Rate of the flux of the variable.
    """
    
    def __init__(self, name, source_box, target_box, variable, rate):
        self.name = name
        self.source_box = source_box
        self.target_box = target_box
        self.variable = variable
        self.rate = rate
        
        self.units = None
        
    def mass_flux_rate(self, time, context):
        """ Calculates the total flux rate depnding on time, and context.
        """
        flux = self.__call__(time, context)
        flux_rate = flux.to_base_units()
        dimensionality_check_mass_flux_err(flux_rate) 
        return flux_rate
