# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 2017 18:57UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

Functions for the Validation of the dimensionality of pint Quantities.
Dimensions:
    M: Mass
    L: Length
    T: Time
    N: Number/Moles

"""

from pint.errors import DimensionalityError

from pint import UnitRegistry
ur = UnitRegistry()

from . import errors as bs_errors # import WrongUnitsDimensionalityError


def is_quantity_of_dimensionality(quantity, *units):
    """Check if quantity has dimensionality equal to one element in *units.

    Compare the dimensionality of quantity with the dimensionality of a 
    list of Quantities (*units).

    Args:
        quantity (pint.Quantity): Quantity to validate.
        *units (pint.Quantity): Quantity of the desired dimensions.

    Returns:
        True if dimensions are correct, False otherwise.

    """
    try:
        for u in units:
            if quantity.dimensionality == u.dimensionality:
                return True
    except AttributeError:
        dimensions = [u.dimensionality for u in units]
        if isinstance(quantity, float) or isinstance(quantity, int):
            if ur.dimensionless.dimensionality in dimensions:
                return True
    return False


def is_mass(quantity):
    """Check if quantity has dimensions of [M]."""
    return is_quantity_of_dimensionality(quantity, ur.kg)


def is_density(quantity):
    """Check if quantity has dimensions of [M/L^3]."""
    return is_quantity_of_dimensionality(quantity, ur.kg / ur.meter**3)


def is_molar_mass(quantity):
    """Check if quantity has dimensions of [M/N]."""
    return is_quantity_of_dimensionality(quantity, ur.kg / ur.mole)


def is_time(quantity):
    """Check if quantity has dimensions of [T]."""
    return is_quantity_of_dimensionality(quantity, ur.second)


def is_volume_per_time(quantity):
    """Check if quantity has dimensions of [L^3/T]."""
    return is_quantity_of_dimensionality(quantity, ur.meter**3 / ur.second)


def is_mass_per_time(quantity):
    """Check if quantity has dimensions of [M/T]."""
    return is_quantity_of_dimensionality(quantity, ur.kg / ur.second)


def is_mole_per_time(quantity):
    """Check if quantity has dimensions of [N/T]."""
    return is_quantity_of_dimensionality(quantity, ur.mole / ur.second)


def is_dimless(quantity):
    """Check if quantity has dimensions of [1]."""
    return is_quantity_of_dimensionality(quantity, ur.kg / ur.kg)


# EXCEPTION RAISING VALIDATION METHODS

def raise_if_not(quantity, *units):
    """Raise exception if quantity's dimensionality is not in *units.

    Args:
        quantity (pint.Quantity): Quantitiy to validate.
        *units (pint.Quantity): Quantity of the desired dimensions.

    """
    dimensions = [u.dimensionality for u in units]
    if not is_quantity_of_dimensionality(quantity, *units):
        if len(units) > 1:
            raise WrongUnitsDimensionalityError('Invalid units {}: '
                    'Must be given in one of the following dimensions: '
                    '{}.'.format(quantity.dimensionality, dimensions))
        else:
            raise WrongUnitsDimensionalityError('Invalid units {}: Must '
                    'be given in on of the following dimensions: {}'.format(
                        quantity.dimensionality, dimensions))


def raise_if_not_mass(quantity):
    """Raise DimensionalityError if quantity has not dimensions [M]."""
    raise_if_not(quantity, ur.kg)


def raise_if_not_density(quantity):
    """Raise DimensionalityError if quantity has not dimensions [M/L^3]."""
    raise_if_not(quantity, ur.kg / ur.meter**3)


def raise_if_not_molar_mass(quantity):
    """Raise DimensionalityError if quantity has not dimensions [M/N]."""
    raise_if_not(quantity, ur.kg / ur.mole)


def raise_if_not_time(quantity):
    """Raise DimensionalityError if quantity has not dimensions [T]."""
    raise_if_not(quantity, ur.second)


def raise_if_not_volume_per_time(quantity):
    """Raise DimensionalityError if quantity has not dimensions [L^3/T]."""
    raise_if_not(quantity, ur.meter**3 / ur.second)


def raise_if_not_mass_per_time(quantity):
    """Raise DimensionalityError if quantity has not dimensions [M/T]."""
    raise_if_not(quantity, ur.kg / ur.second)


def raise_if_not_mole_per_time(quantity):
    """Raise DimensionalityError if quantity has not dimensions [N/T]."""
    raise_if_not(quantity, ur.mole / ur.second)


def raise_if_not_dimless(quantity):
    """Raise DimensionalityError if quantity has not dimensions [1]."""
    raise_if_not(quantity, ur.dimensionless)

