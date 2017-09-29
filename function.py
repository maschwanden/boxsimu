# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 2017 at 10:29UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

The class UserFunction is a wrapper for user-defined functions.

The user of boxsimu can define dynamic rate-functions for Processes
Reactions, Flows, Fluxes. Additionally the density of fluids can also be
a dynamic function. 

User-defined functions are called during the simulation
with three parameters/arguments:
    time (pint.Quantity [T]): Time of the simulation.
    condition (Condition): Condition of the Box/Flow/Flux.
    system (BoxModelSystem): System that is solved. Allowes the user
        to access all Variables in all Boxes of the system.
"""

import pint

from . import dimensionality_validation as bs_dim_val


class UserFunction:
    """User-defined function for dynamic rates or equations.
    
    UserFunction is a wrapper for user-defined functions, used 
    for example in instances of the classes Process, Reaction, Flow, 
    and Flux. These user-defined functions are called during the 
    simulation of the system by the solver with three arguments:
    time, condition, and system.

    Args:
        expression (pint.Quantity or callable that returns pint.Quantity): 
            User-defined function or constant that 
        units (pint.Quantity): Quantity of the desired dimensions.

    """

    def __init__(self, expression, units):
        self.units = units
        if not callable(expression):
            bs_dim_val.raise_if_not(expression, units)
            self.expression = expression.to_base_units()
            self.dimensionality_verified = True
            self.call_func = self.static_call
            self.is_static = True
            self.is_dynamic = False
        else:
            self.dimensionality_verified = False
            self.expression = expression
            self.call_func = self.dynamic_call
            self.is_static = False
            self.is_dynamic = True
            
    def static_call(self, *args):
        """Call method for static UserFunction."""
        return self.expression
    
    def dynamic_call(self, *args):
        """Call method for dynamic UserFunction."""
        expression = self.expression(*args)
        if not self.dimensionality_verified:
            bs_dim_val.raise_if_not(expression, units)
            self.dimensionality_verified = True
        return expression.to_base_units()
    
    def __call__(self, *args):
        """UserFunction is called with args: time, condition, system.
        
        A UserFunction instance is called from an instance of the class 
        Solver during a simulation of a BoxModelSystem.

        Args:
            time (pint.Quantity [T]): Time of the simulation.
            condition (Condition): Condition of the Box/Flow/Flux.
            system (BoxModelSystem): System that is solved. Allows the user
                to access all Variables in all Boxes of the system.
            
        """
        return self.call_func(*args)


