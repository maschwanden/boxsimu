# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 2017 at 11:08UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

from . import dimensionality_validation as bs_dim_val
from keyword import iskeyword


class PintQuantityDescriptor:
    """Descriptor to check correct units of pint-quantity-attributes."""
    def __init__(self, name, units, default=None):
        self.name = '_' + name
        self.units = units
        self.default = default

    def __get__(self, instance, instance_type):
        return getattr(instance, self.name, self.default)
    
    def __set__(self, instance, value):
        if instance is None: return self
        if value is None: return 
        if not callable(value):
            bs_dim_val.raise_if_not(value, self.units)
        setattr(instance, self.name, value)


class QuantifiedPintQuantityDescriptor(PintQuantityDescriptor):
    """Descriptor to check correct units of mass-attributes."""
    def __set__(self, instance, value):
        instance._quantified = True
        super().__set__(instance, value)

class ImmutableNameDescriptor:
    """Descriptor to assure that the name attribute is immutable.""" 

    def __get__(self, instance, instance_type):
        return self._name

    def __set__(self, instance, value):
        if hasattr(instance, '_name'):
            raise AttributeError('Can\'t set immutable attribute')
        if not value.isidentifier() or iskeyword(value):
            raise ValueError('Name must be a valid python expression!')
        self._name = value
