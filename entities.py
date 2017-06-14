# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 07:56:18 2016

@author: aschi
"""
import copy

import numpy as np

from utils import (dimensionality_check, dimensionality_check_err, 
                   dimensionality_check_density, dimensionality_check_mass_err)


class Fluid:
    """ Represents a Fluid within a Box.
    
    Attributes:
    - name: Name of the fluid. Should be a short describtive text.
    - rho_expr (pint quantity or function that returns pint quantity):
            Density of the fluid.
    - volume (pint quantity): Volume of the whole fluid.
    - mass (pint quantity): Mass of the whole fluid.
    """
    
    def __init__(self, name, rho_expr, mass):
        self.name = name
        if callable(rho_expr) or dimensionality_check_density(rho_expr):
            self.rho_expr = rho_expr
        else:
            raise ValueError('rho_expr must either be a callable function or '
                             'have the dimension of weight per volume')
        self.mass = mass
        self.box = None
            
    @property
    def rho(self):
        if callable(self.rho_expr):
            time = None
            rho = self.rho_expr(time, self.box.context)
        else:
            rho = self.rho_expr
        return rho
    
    @property
    def volume(self):
        return self.mass / self.rho
            
    def info(self,):
        print(self.name)
        print('_____________________')
        print('Volume: {}'.format(self.volume))
        print('Mass: {}'.format(self.mass))
        print('Density: {}'.format(self.rho))


class Variable:
    """ Quantity that is analysed.
    
    Attributes:
    - name: Name of the variable. Should be a short describtive text.
    """
    
    ALLOWED_CONCENTRATION_TYPES = ['volumetric', 'areal']
    
    def __init__(self, name, mass, concentration_type='volumetric'):
        self.ID = None
        self.name = name

        dimensionality_check_mass_err(mass)    
        self.mass = mass

        self.concentration_type = concentration_type
        self.units = mass.units
        self.box = None
        
        
    def __eq__(self, other):
        return self.name == other.name
    
    def __add__(self, other):
        if self.name == other.name:
            total_mass = self.mass + other.mass
            return Variable(name=self.name, mass=total_mass)
        else:
            raise ValueError('Variables are not compatible!')
    
    @property        
    def concentration(self):
        if self.box:
            if self.box.volume and self.concentration_type=='volumetric':
                return self.mass / self.box.volume
            if self.box.area and self.concentration_type=='areal':
                return self.mass / self.box.surface
        return np.nan



        
    
