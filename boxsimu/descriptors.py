# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 2017 at 11:08UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

Descriptors used in boxsimu in order to facilitate the handling of different
attributes (e.g. immutable attributes, or attributes that must have a 
certain pint-dimensionality).

"""

import pint
import collections
from attrdict import AttrDict

from . import validation as bs_validation
from keyword import iskeyword

from . import condition as bs_condition
from . import errors as bs_errors
from . import function as bs_function


class PintQuantityDescriptor:
    """Check correct pint-dimensionality of attribute.
    
    Raise an exception if either a newly assigned value to the attribute
    under consideration is not a pint.Quantity or if the dimensionality
    of the pint.Quantity is not correct.

    """

    def __init__(self, name, units, default=None):
        self.name = '_' + name
        self.units = units
        self.default = default

    def __get__(self, instance, instance_type):
        return getattr(instance, self.name, self.default)
    
    def __set__(self, instance, value):
        if instance is None: return self
        if value is None: return 
        bs_validation.raise_if_not(value, self.units)
        setattr(instance, self.name, value.to_base_units())


class QuantifiedPintQuantityDescriptor(PintQuantityDescriptor):
    """Check correct pint-dimensionality of attribute.
    
    If the attribute is set correctly the instance is marked as quantified 
    (e.g. the Fluid or Variable instance is then marked as quantified).

    """

    def __set__(self, instance, value):
        super().__set__(instance, value)
        instance._quantified = True


class ImmutableDescriptor:
    """Check that an attribute is immutable.
    
    Raise an exception if an attribute is assigned a value for the second 
    time.
    
    """ 

    def __init__(self, name):
        self.name = '_' + name
        self.name_raw = name

    def __get__(self, instance, instance_type):
        return getattr(instance, self.name, None)

    def __set__(self, instance, value):
        if hasattr(instance, self.name):
            raise AttributeError(
                'Cannot set immutable attribute "{}".'.format(self.name_raw))
        setattr(instance, self.name, value) 


class ImmutableIdentifierDescriptor(ImmutableDescriptor):
    """Check that the name is immutable and a valid identifier.
    
    Raise an exception if the """ 
    def __set__(self, instance, value):
        if not value.isidentifier() or iskeyword(value):
            raise ValueError('Name must be a valid python expression!')
        super().__set__(instance, value)


class BaseDictDescriptor:
    """Check if keys and values are instances of certain classes.
    
    Descriptor that assures that an attribute is a dict with key-value 
    pairs of certain classes. E.g. key-value pairs of an integer as a key
    and an instance of Box as value.

    Args:
        name (str): name of the attribute of the parents class.
        key_classes (list of classes): Classes that are allowed as key
            instances.
        value_class (list of classes): Classes that are allowed as value
            instances.

    """

    dict_class = dict

    def __init__(self, name, key_classes, value_classes):
        self.name = '_' + name
        self.name_raw = name
        if not isinstance(key_classes, list):
            raise bs_errors.NotInstanceOfError('key_classes', 'list')
        self.key_classes = key_classes
        if not isinstance(value_classes, list):
            raise bs_errors.NotInstanceOfError('value_classes', 'list')
        self.value_classes = value_classes
        
    def __get__(self, instance, instance_type):
        return getattr(instance, self.name)
    
    def __set__(self, instance, value):
        if instance is None: 
            return self
        if value is None: 
            return 
        value = self._check_key_value_types(value)
        setattr(instance, self.name, value)
        
    def _check_key_value_types(self, value):
        if not isinstance(value, self.dict_class):
            raise bs_errors.NotInstanceOfError(self.name_raw, self.dict_class)
        for k, v in value.items():
            key_isinstance_list = [isinstance(k, i) 
                    for i in self.key_classes]
            if not any(key_isinstance_list):
                raise bs_errors.DictKeyNotInstanceOfError(
                    self.name_raw, self.key_classes)
            value_isinstance_list = [isinstance(v, i) 
                    for i in self.value_classes]
            if not any(value_isinstance_list):
                raise bs_errors.DictValueNotInstanceOfError(
                    self.name_raw, self.value_classes)
        return value


class AttrDictDescriptor(BaseDictDescriptor):
    dict_class = AttrDict


class PintQuantityValueDictDescriptor(BaseDictDescriptor):
    """Check if keys have the correct type and values are pint quantites.
    
    Attribute must be a dict with instances of one type of {key_classes} 
    and pint.Quantities with dimensionality equal to these of units as 
    values.
    
    """

    def __init__(self, name, key_classes, *units):
        super().__init__(name, key_classes, value_classes=[
            pint.quantity._Quantity])
        self.units = units
    
    def _check_key_value_types(self, value):
        value = super()._check_key_value_types(value)
        for k, v in value.items():
            bs_validation.raise_if_not(v, *self.units)
        return value


class PintQuantityExpValueDictDescriptor(BaseDictDescriptor):
    """Check if keys have the correct type and values are pint quantites.
    
    Attribute must be a dict with instances of one type of {key_classes} 
    and pint.Quantities with dimensionality equal to these of units or 
    callables that return pint.Quantites with dimensionality equal to 
    these of units as values.

    """

    def __init__(self, name, key_classes, *units):
        super().__init__(name, key_classes, value_classes=[
            pint.quantity._Quantity, collections.Callable])
        self.units = units

    def _check_key_value_types(self, value):
        value = super()._check_key_value_types(value)
        for k, v in value.items():
            value[k] = bs_function.UserFunction(v, *self.units)
        return value


class ConditionUserFunctionDescriptor(BaseDictDescriptor):
    dict_class = bs_condition.Condition

    def __init__(self, name, key_classes):
        super().__init__(name, key_classes, value_classes=[
            pint.quantity._Quantity, collections.Callable])

    def _check_key_value_types(self, value):
        value = super()._check_key_value_types(value)
        for k, v in value.items():
            value[k] = bs_function.UserFunction(v, *self.units)
        return value
