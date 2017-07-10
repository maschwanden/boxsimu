# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:23:46 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

Box class that represents a box within a sinlge- or multibox-system.

"""

import copy
from keyword import iskeyword
from attrdict import AttrDict

from . import errors as bs_errors
from . import condition as bs_condition


class Box:
    """Perfectly mixed Box in a single- or multibox system.

    A Box contains one Fluid and zero to multiple Variables.
    Processes and Reactions can be defined that alter the .
    
    Args:
        name (str): Valid python expression used to access the box.
        name_long (str): Human readable string describing the box.     
        fluid (Fluid): Fluid that represents the solvent of the box.
        condition (Condition): Conditions of the box. Used for the evaluation
            of user-defined process/reaction/flow/flux rates.
        variables (list of Variable): Variables that are found within the box.
            The Variable instances must be quantified. That means they must
            be generated using the method q() on a Variable instance.
            Default: []
        processes (list of Process): Processes that take place in the box.
            Default: []
        reactions (list of Reaction): Reactions that take place in the box.
            Default: []

    Attributes:
        id (int): ID of the box within a BoxModelSystem. Note: This
            Attribute is set by the BoxModelSystem instance that
            contains a box.
        name (str): Valid python expression used to access the box.
        name_long (str): Human readable string describing the box.     
        fluid (Fluid): Fluid that represents the solvent of the box.
        condition (Condition): Conditions of the box. Used for the evaluation
            of user-defined process/reaction/flow/flux rates.
        variables (list of Variable): Variables that are found within the box.
        processes (list of Process): Processes that take place in the box.
        reactions (list of Reaction): Reactions that take place in the box.

    """

    def __init__(self, name, name_long, fluid, condition=None, variables=[],
                 processes=[], reactions=[]):
        self.id = None

        if not name.isidentifier() or iskeyword(name):
            raise ValueError('Name must be a valid python variable name!')

        self.name = name
        self.name_long = name_long

        if not fluid.quantified:
            raise bs_errors.FluidNotQuantifiedError('Fluid was not quantified!')
        self.fluid = fluid
        self.condition = copy.deepcopy(
            condition) if condition else bs_condition.Condition()
        self.processes = copy.deepcopy(processes)
        self.reactions = copy.deepcopy(reactions)
        self.variables = AttrDict()

        for variable in variables:
            if not variable.quantified:
                raise bs_errors.VariableNotQuantifiedError(
                    'Variable was not quantified!')
            self.variables[variable.name] = variable

        variable_names = [var_name for var_name, var in self.variables.items()]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError('Variable names have to be unique!')

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.name > other.name
        return false

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.name < other.name
        return false

    @property
    def mass(self):
        variable_mass = sum(
            [var.mass for var_name, var in self.variables.items()])
        return self.fluid.mass + variable_mass

    @property
    def cond(self):
        return self.condition

    @property
    def var(self):
        return self.variables

    def get_volume(self, context=None):
        return self.fluid.get_volume(context)

    def get_concentration(self, variable, context=None):
        """ Returns the mass concentration [kg/kg] of variable within the box. """
        if self.mass.magnitude > 0:
            concentration = self.variables[variable.name].mass / self.mass
            concentration = concentration.to_base_units()
            return concentration
        return 0

    def get_vconcentration(self, variable, context=None):
        """ Returns the volumetric concentration [kg/m^3] of variable within the box."""
        box_volume = self.get_volume(context)
        if box_volume.magnitude > 0:
            concentration = self.variables[variable.name].mass / box_volume
            concentration = concentration.to_base_units()
            return concentration
        return 0
