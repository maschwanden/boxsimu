# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 07:56UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
import numpy as np
from keyword import iskeyword

# import all submodules with prefix 'bs' for BoxSimu
from . import dimensionality_validation as bs_dim_val
from . import ur


class BaseEntity:
    """Base entity class.

    Args:
        name (str): Human readable string describing the entity.
        
    Attributes:
        name (str): Human readable string describing the entity.
        mass (pint.Quantity): Mass of the entity.  
        molar_mass (pint.Quantity): Molar mass of the variable.
    """
    def __init__(self, name, molar_mass=None):
        self.name = name

        self.molar_mass = molar_mass
        if self.molar_mass:
            bs_dim_val.raise_if_not_molar_mass(self.molar_mass) 

        self._mass = None
        self._quantified = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError('Equality can only be checked for instance '
                             'of the same class.')
        if self.name == other.name:
            if self.molar_mass == other.molar_mass:
                return True
            elif self.molar_mass is None or other.molar_mass is None:
                return True
            else:
                return False
        else:
            return False

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.name > other.name
        return false

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.name < other.name
        return false

    def __add__(self, other):
        if self == other:
            total_mass = self.mass + other.mass
            new_entity = self.q(total_mass)
            return new_entity
        else:
            raise ValueError('Addition not possible, objects are not compatible!')

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        bs_dim_val.raise_if_not_mass(value)
        self._quantified = True
        self._mass = value

    @property
    def quantified(self):
        return self._quantified

    def q(self, value):
        """Returns a 'quantified' deepcopy of the current instance."""
        # value can be a pint.Quantity of units mole or kg
        bs_dim_val.raise_if_not(value, ur.kg, ur.mole)
        self_copy = copy.deepcopy(self)

        if bs_dim_val.is_mass(value):
            self_copy.mass = value
        else:
            if self.molar_mass:
                self_copy.mass = self.molar_mass * value
            else:
                raise ValueError('Entity cannot be quantified. '
                        'Molar mass is not set.')
        self_copy.molar_mass = self.molar_mass
        return self_copy


class Fluid(BaseEntity):
    """Represent a Fluid of a Box.

    The Fluid within a Box defines the Box's volume, and mainly also its mass.
    Fluid mass is exchanged between the boxes of a BoxModelSystem and with the
    exterior of the system. The Fluid mass can be diminshed to zero but not below.

    Attributes:
        name (str): Human readable string describing the entity.
        rho_expr (pint.Quantity or callable that returns pint.Quantity):
            Density of the Fluid. Note: Must have dimensions of [M/L^3].
        mass (pint.Quantity): Mass of the fluid.

    """

    def __init__(self, name, rho_expr):
        if callable(rho_expr) or bs_dim_val.is_density(rho_expr):
            self.rho_expr = rho_expr
        else:
            raise ValueError('rho_expr must either be callable or a pint.Quantity '
                    'with dimensions of [M/L^3].')
        super().__init__(name)

    def get_rho(self, context=None):
        """Return the density of the Fluid."""
        if callable(self.rho_expr):
            if context is None:
                raise ValueError('If rho is given as a dynamic expression '
                    '(function), an appropriate context must be given.')
            rho = self.rho_expr(context)
        else:
            rho = self.rho_expr
        return rho.to_base_units()

    def get_volume(self, context=None):
        """Return the volume of the Fluid."""
        return (self.mass / self.get_rho(context)).to_base_units()


class Variable(BaseEntity):
    """Tracer/Substance that is analysed.

    Attributes (in addition to attributes from base class BaseEntity):
        id (int): Id of the variable in the system. Ids are assigned 
            alphabetically by the BoxModelSystem instance.

    """

    def __init__(self, name, molar_mass=None):
        self.id = None
        super().__init__(name)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        # Make name an immutable attribute
        if hasattr(self, '_name'):
            raise AttributeError('Can\'t set immutable attribute')
        if not value.isidentifier() or iskeyword(value):
            raise ValueError('Name must be a valid python expression!')
        self._name = value

