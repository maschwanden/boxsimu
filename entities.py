# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 07:56:18 2016

@author: aschi
"""
import copy
import numpy as np

import utils
import errors

class BaseMassEntity:
    def __init__(self, name):
        self.name = name

        self.mass = None 
        self.units = None
        self._quantified = False

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError('Equality can only be checked for instance '\
                    'of the same class.')
        return self.name == other.name

    def __add__(self, other):
        if self == other:
            total_mass = self.mass + other.mass
            new_entity = self.q(total_mass)
            return new_entity
        else:
            raise ValueError('Summing not possible! Objects are not compatible!')

    @property
    def quantified(self):
        if self._quantified:
            pass
        elif not self.mass is None:  # If self._quantified is False but mass is still set
            utils.dimensionality_check_mass_err(self.mass)
            self._quantified = True
        return self._quantified

    def q(self, mass):
        """ Returns a 'quantified' deepcopy of the current variable instance. """
        utils.dimensionality_check_mass_err(mass) 
        self_copy = copy.deepcopy(self)
        self_copy.mass = mass
        self_copy.units = mass.units
        self_copy._quantified = True
        return self_copy

    def get_empty_copy(self):
        """ Returns a deepcopy of the current variable instance with zero mass. """
        self_copy = copy.deepcopy(self)
        self_copy.mass *= 0
        return self_copy


class Fluid(BaseMassEntity):
    """ Represents a Fluid that fills a Box.
    
    The Fluid within a Box defines the Box's volume, and mainly also its mass.
    Fluid mass is exchanged between the boxes of a BoxModelSystem and with the
    exterior of the system. The Fluid mass can be diminshed to zero but not below.

    Attributes:
    - name: Name of the fluid. Should be a short descriptive text.
    - rho_expr (pint quantity or function that returns pint quantity):
            Density of the fluid.
    - mass (pint Quantity): Mass of the fluid.
    """
    
    def __init__(self, name, rho_expr):
        if callable(rho_expr) or utils.dimensionality_check_density(rho_expr):
            self.rho_expr = rho_expr
        else:
            raise ValueError('rho_expr must either be a callable function or '
                             'have the dimension of weight per volume')
        super(Fluid, self).__init__(name)
            
    def get_rho(self, time=None, context=None):
        if callable(self.rho_expr):
            if time is None or context is None:
                raise ValueError('If rho is given as a dynamic expression '\
                        '(function), appropriate parameters for time and '\
                        'context must be given.')
            rho = self.rho_expr(time, context)
        else:
            rho = self.rho_expr
        return rho
    
    def get_volume(self, time=None, context=None):
        return self.mass / self.rho(time, context)
            

class Variable(BaseMassEntity):
    """ Quantity that is analysed.
    
    Attributes:
    - name: Name of the variable. Should be a short describtive text.
    """
    
    def __init__(self, name):
        self.ID = None
        super(Variable, self).__init__(name)


        
    
