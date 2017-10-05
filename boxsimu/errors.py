
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 2017 at 12:51UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

class BoxsimuBaseException(Exception):
    """Base Exception for Boxsimu Project."""
    pass


class BoxsimuDefaultMessageException(BoxsimuBaseException):
    """Exception with a default message."""
    message_default = None
    
    def __init__(self, message=None):
        if not message:
            message = self.message_default
        super().__init__(message)


class VariableNotQuantifiedError(BoxsimuDefaultMessageException):
    """Raise if Variable is not quantified but should be."""
    message_default = 'Variable was not quantified!'


class FluidNotQuantifiedError(BoxsimuDefaultMessageException):
    """Raise if Fluid is not quantified but should be."""
    message_default = 'Fluid was not quantified!'


class WrongUnitsDimensionalityError(BoxsimuBaseException):
	"""Raise units have wrong dimensionalities."""
	pass


class NotInstanceOfError(BoxsimuBaseException):
    """Raise if a parameter is not an instance of the correct class."""
    message_template = '"{}" must be an instance of a class in: [{}]'

    def __init__(self, variable_name, classes):
        message = self.message_template.format(variable_name, classes)
        super().__init__(message)


class DictKeyNotInstanceOfError(NotInstanceOfError):
    """Raise if a dict-key is not an instance of the correct class."""
    message_template = 'The keys of the dict "{}" must be instances of a '\
        'class in: [{}]'


class DictValueNotInstanceOfError(BoxsimuBaseException):
    """Raise if a dict-value is not an instance of the correct class."""
    message_template = 'The values of the dict "{}" must be instances of a '\
        'class in: [{}]'
        

class NoFluidInBoxError(BoxsimuDefaultMessageException):
    """Raise if a source or target box has no fluid."""
    message_default = 'Boxes without a fluid cannot be part of a Flow.'


class FixedVariableConcentrationError(BoxsimuDefaultMessageException):
    """Raise if a flow from a box has a prescribed variable concentration."""
    message_default = 'Flows from a source box must not prescribe Variable '\
        'concentrations.'


class FluidsOfBoxesNotIdenticalError(BoxsimuDefaultMessageException):
    """Raise if the source- and target-box fluid of a flow are not equal."""
    message_default = 'The source and target box of a flow must be filled '\
        'with the same Fluid to allow a flow between them.'





