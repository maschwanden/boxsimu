# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 08:42:29 2016

@author: aschi
"""

from pint import DimensionalityError

from pint import UnitRegistry
ur = UnitRegistry()


def dimensionality_check(quantitiy, *units):
    """ Checks if a given value has the correct dimensions in pint units.
    
    Attribute:
    - quantitiy (pint quantity): The quantitiy to check.
    - units (list of pint units): Reference units of the desired dimensions.
    
    Return:
    - Dimensionality check status (bool): True if dimensions are compatible, 
                                          False if not 
    """
    try:
        for u in units:
            if quantitiy.dimensionality == u.dimensionality:
                return True
    except AttributeError:
        pass
    return False


def dimensionality_check_err(quantitiy, *units):
    """ Checks if a given quantitiy has the correct dimensions in pint units 
    and throws a ValueError if the dimensions are incorrect.
    
    Attribute:
    - quantitiy (pint quantity): The quantitiy to check.
    - units (list of pint units): Reference units of the desired dimensions.
    """
    dimensions = [u.dimensionality for u in units]
    if not dimensionality_check(quantitiy, *units):
        if len(units) > 1:
            raise ValueError(
                'Invalid units {}: Must be given in one of the '
                'following dimensions: {}.'.format(quantitiy.dimensionality, 
                                                   dimensions)
                )
        else:
            raise ValueError(
                'Invalid units {}: Must be given in '
                'the dimension: {}'.format(quantitiy.dimensionality, dimensions)
            )


def dimensionality_check_mass(quantitiy):
    """ Checks if a given value has dimensions of mass in pint units."""
    return dimensionality_check(quantitiy, ur.kg)


def dimensionality_check_density(quantitiy):
    """ Checks if a given value has dimensions of density in pint units."""
    return dimensionality_check(quantitiy, ur.kg/ur.meter**3)


def dimensionality_check_volume_flux(quantitiy):
    """ Checks if a given value has dimensions of volume flux in pint units."""
    return dimensionality_check(quantitiy, ur.meter**3/ur.second)


def dimensionality_check_mass_flux(quantitiy):
    """ Checks if a given value has dimensions of mass flux in pint units."""
    return dimensionality_check(quantitiy, ur.kg/ur.second)


def dimensionality_check_mass_err(quantitiy):
    """ 
    Raises Error if a given value has not dimensions of mass in pint units.
    """
    dimensionality_check_err(quantitiy, ur.kg)


def dimensionality_check_density_err(quantitiy):
    """ 
    Raises Error if a given value has not dimensions of density in pint units.
    """
    dimensionality_check_err(quantitiy, ur.kg/ur.meter**3)


def dimensionality_check_volume_flux_err(quantitiy):
    """ 
    Raises Error if a given value has not dimensions of volume flux in pint units.
    """
    dimensionality_check_err(quantitiy, ur.meter**3/ur.second)


def dimensionality_check_mass_flux_err(quantitiy):
    """ 
    Raises Error if a given value has not dimensions of mass flux in pint units.
    """
    dimensionality_check_err(quantitiy, ur.kg/ur.second)


def dimensionality_check_mass_dimless(quantitiy):
    """ 
    Raises Error if a given value has not dimensions of mass flux in pint units.
    """
    dimensionality_check_err(quantitiy, ur.dimensionless)


def magnitude_in_current_units(quantity):
    """ Splits a pint.Quantity into its magnitude and the current unit. 
    Attribute:
    - quantity (pint.Quantity)
    
    Return:
    - magnitude (float): Magnitude of the parameter quantity
    - current_unit (pint.UnitsContainer): Units of quantity    
    """
    units = quantity.units
    mag = quantity.magnitude
    return mag, units


def magnitude_in_base_units(quantity):
    """ Splits a pint.Quantity into its magnitude in base units and the 
    corresponding base units.
    
    Attribute:
    - quantity (pint.Quantity)
    
    Returns:
    - magnitude (float): Magnitude in base units of the parameter quantity
    - base_units (pint.UnitsContainer): Base units of quantity
    """
    quantity_base_units = quantity.to_base_units()
    base_units = quantity_base_units.units
    magnitude = quantity_base_units.magnitude
    return magnitude, base_units
    

      
