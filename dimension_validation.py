# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:57:31 2017

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

Functions for the Validation of the dimensionality of pint Quantities.
Dimensions:
    M: Mass
    L: Length
    T: Time

"""

from pint.errors import DimensionalityError

from pint import UnitRegistry
ur = UnitRegistry()


def dimensionality_check(quantity, *units):
    """Check if quantity is of the correct dimensionality.

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


def dimensionality_check_err(quantity, *units):
    """Raise exception if quantity is not of the correct dimensionality.

    Args:
        quantity (pint.Quantity): Quantitiy to validate.
        *units (pint.Quantity): Quantity of the desired dimensions.

    """
    dimensions = [u.dimensionality for u in units]
    if not dimensionality_check(quantity, *units):
        if len(units) > 1:
            raise DimensionalityError('Invalid units {}: Must be given ' 
                    'in one of the following dimensions: {}.'.format(
                        quantity.dimensionality, dimensions))
        else:
            raise DimensionalityError('Invalid units {}: Must be given in '
                    'on of the following dimensions: {}'.format(
                        quantity.dimensionality, dimensions))


def dimensionality_check_mass(quantity):
    """Check if quantity has dimensions of [M]."""
    return dimensionality_check(quantity, ur.kg)


def dimensionality_check_density(quantity):
    """Check if quantity has dimensions of [M/L^3]."""
    return dimensionality_check(quantity, ur.kg / ur.meter**3)

def dimensionality_check_time(quantity):
    """Check if quantity has dimensions of [T]."""
    return dimensionality_check(quantity, ur.second)

def dimensionality_check_volume_per_time(quantity):
    """Check if quantity has dimensions of [L^3/T]."""
    return dimensionality_check(quantity, ur.meter**3 / ur.second)


def dimensionality_check_mass_per_time(quantity):
    """Check if quantity has dimensions of [M/T]."""
    return dimensionality_check(quantity, ur.kg / ur.second)

def dimensionality_check_dimless(quantity):
    """Check if quantity has dimensions of [1]."""
    return dimensionality_check(quantity, ur.kg / ur.kg)

def dimensionality_check_mass_err(quantity):
    """Raise DimensionalityError if quantity has not dimensions [M]."""
    dimensionality_check_err(quantity, ur.kg)


def dimensionality_check_density_err(quantity):
    """Raise DimensionalityError if quantity has not dimensions [M/L^3]."""
    dimensionality_check_err(quantity, ur.kg / ur.meter**3)

def dimensionality_check_time_err(quantity):
    """Raise DimensionalityError if quantity has not dimensions [T]."""
    dimensionality_check_err(quantity, ur.second)

def dimensionality_check_volume_per_time_err(quantity):
    """Raise DimensionalityError if quantity has not dimensions [L^3/T]."""
    dimensionality_check_err(quantity, ur.meter**3 / ur.second)

def dimensionality_check_mass_per_time_err(quantity):
    """Raise DimensionalityError if quantity has not dimensions [M/T]."""
    dimensionality_check_err(quantity, ur.kg / ur.second)

def dimensionality_check_dimless_err(quantity):
    """Raise DimensionalityError if quantity has not dimensions [1]."""
    dimensionality_check_err(quantity, ur.dimensionless)

