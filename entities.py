# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 07:56UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
import numpy as np

# import all submodules with prefix 'bs' for BoxSimu
from . import dimension_validation as bs_dim_val


class BaseMassEntity:
    """Mass-based Entity class.

    Args:
        name (str): Human readable string describing the entity.
        
    Attributes:
        name (str): Human readable string describing the entity.
        mass (pint.Quantity): Mass of the entity.  
    """
    def __init__(self, name):
        self.name = name

        self.mass = None
        self.units = None
        self._quantified = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError('Equality can only be checked for instance '
                             'of the same class.')
        return self.name == other.name

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
    def quantified(self):
        if self._quantified:
            pass
        elif self.mass is None:  # If not self._quantified but mass is not still set
            bs_dim_val.dimensionality_check_mass_err(self.mass)
            self._quantified = True
        return self._quantified

    def q(self, mass):
        """Returns a 'quantified' deepcopy of the current variable instance."""
        bs_dim_val.dimensionality_check_mass_err(mass)
        self_copy = copy.deepcopy(self)
        self_copy.mass = mass
        self_copy.units = mass.units
        self_copy._quantified = True
        return self_copy


class Fluid(BaseMassEntity):
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
        if callable(rho_expr) or bs_dim_val.dimensionality_check_density(rho_expr):
            self.rho_expr = rho_expr
        else:
            raise ValueError('rho_expr must either be callable or a pint.Quantity '
                    'with dimensions of [M/L^3].')
        super(Fluid, self).__init__(name)

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


class Variable(BaseMassEntity):
    """Tracer/Substance that is analysed.

    Attributes (in addition to attributes from base class BaseMassEntity):
        id (int): Id of the variable in the system. Ids are assigned 
            alphabetically by the BoxModelSystem instance.
    """

    def __init__(self, name):
        self.id = None
        super(Variable, self).__init__(name)

