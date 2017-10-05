# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 07:56UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
import numpy as np
from keyword import iskeyword

# import all submodules with prefix 'bs' for BoxSimu
from . import validation as bs_validation
from . import descriptors as bs_descriptors
from . import function as bs_function
from . import ur


class BaseEntity:
    """Base entity class.

    Args:
        name (str): Human readable string describing the entity.
        molar_mass (pint.Quantity): Molar mass of the variable.
 
    Attributes:
        name (str): Short, human readable string describing the entity.
            Must be a valid Python expression, meaning a valid 
            Python-variable name.
        description (str): Human readable string describing the entity.
        mass (pint.Quantity [M]): Mass of the entity.
        molar_mass (pint.Quantity [M/N]): Molar mass of the variable.

    """

    name = bs_descriptors.ImmutableIdentifierDescriptor('name')
    mass = bs_descriptors.QuantifiedPintQuantityDescriptor(
            'mass', ur.kg, 0*ur.kg)
    molar_mass = bs_descriptors.PintQuantityDescriptor(
            'molar_mass', ur.kg/ur.mole, 0*ur.kg/ur.mole)

    def __init__(self, name, molar_mass=None, description=None):
        self.name = name
        self.molar_mass = molar_mass
        self._quantified = False
        if not description:
            self.description = name

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
    def quantified(self):
        return self._quantified

    def q(self, value):
        """Returns a 'quantified' deepcopy of the current instance."""
        # value can be a pint.Quantity of units mole or kg
        bs_validation.raise_if_not(value, ur.kg, ur.mole)
        self_copy = copy.deepcopy(self)

        if bs_validation.is_mass(value):
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
    """Represent a Fluid (Solvent for Variables) of a Box.

    The Fluid within a Box defines the Box's volume, and mainly also its 
    mass. Fluid mass is exchanged between the boxes of a BoxModelSystem 
    and with the exterior of the system. The Fluid mass can be diminshed 
    to zero but not below.

    Attributes:
        name (str): Human readable string describing the entity.
        rho_expr (pint.Quantity or callable that returns pint.Quantity):
            Density of the Fluid. Note: Must have dimensions of [M/L^3].
        mass (pint.Quantity): Mass of the fluid.

    """

    def __init__(self, name, rho, description=None):
        self.rho = bs_function.UserFunction(rho, ur.kg/ur.meter**3)
        super().__init__(name, description=description)

    def get_rho(self, time, context, system):
        """Return the density of the Fluid."""
        return self.rho(time, context, system)

    def get_volume(self, time, context, system):
        """Return the volume of the Fluid."""
        rho = self.rho(time, context, system)
        return (self.mass / rho).to_base_units()


class Variable(BaseEntity):
    """Tracer/Substance that is analysed.

    Args:
        name (str): Human readable string describing the variable.
        name_symolic (str): Human readable short string that clearly 
            represents the variable.
            Defaults to the same value as name.
        mobility (boolean or function that returns boolean): Specifies
            whether the variable is mobile, and thus whether it is
            passively transported with fluid flows.

    Attributes (in addition to attributes from base class BaseEntity):
        id (int): Id of the variable in the system. Ids are assigned 
            alphabetically by the BoxModelSystem instance.
        is_mobile (boolean): Specifies whether the variable is mobile, and
            thus whether it is passively transported with fluid flows.

    """

    id = bs_descriptors.ImmutableDescriptor('id')

    def __init__(self, name, molar_mass=None, mobility=True, 
            description=None):
        self.mobility = mobility
        super().__init__(name, molar_mass=molar_mass, description=description)

    def is_mobile(self, time, context, system):
        mobile = self.mobility 
        if callable(self.mobility):
            mobile = self.mobility(time, context, system)
        if not type(mobile) == bool:
            raise ValueError('Variable.mobility expression must return bool.')
        return mobile
        
