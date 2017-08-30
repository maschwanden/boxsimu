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
from functools import wraps

from pint.errors import DimensionalityError

from . import errors as bs_errors # import WrongUnitsDimensionalityError
from . import ur


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
            raise bs_errors.WrongUnitsDimensionalityError('Invalid units {}: '
                    'Must be given in one of the following dimensions: '
                    '{}.'.format(quantity.dimensionality, dimensions))
        else:
            raise bs_errors.WrongUnitsDimensionalityError('Invalid units {}: Must '
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


# VECTOR/List validation

def get_single_shared_unit(units, default_units=1):
    """Returns unit if all units of list are identical.

    If list is empty return the default units. If the list contains
    multiple units, raise DimensionalityError.

    """
    units_set = set(units)
    if len(units_set) == 1:
        res_units = units_set.pop()
    elif len(units_set) == 0:
        res_units = default_units
    else:
        raise DimensionalityError(units_set.pop(), units_set.pop())
    return res_units


# Function validation

def decorator_raise_if_not(*units):
    """Decorator to check the dimensionality of the function's return value."""
    try:
        iterator = iter(units)
    except TypeError:
        dimensionalities = units.dimensionality
    else:
        dimensionalities = [u.dimensionality for u in units]

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            if not is_quantity_of_dimensionality(result, *units):
                raise ValueError('The user-defined function <{}> doesn\'t '
                        'return a quantity with the needed '
                        'dimensionality: {}'.format(
                            function.__name__, dimensionalities))
            return result.to_base_units()
        return wrapper
    return decorator

decorator_raise_if_not_mass_per_time = decorator_raise_if_not(
        ur.kg / ur.second)
