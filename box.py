# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 11:08UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

A Box represents a box within a sinlge- or multibox-system.

A Box instance represents one compartement in a system that is most often
assumed to be well mixed (but doesn't have to). It contains information on
the amount of fluid/solvent (e.g. water) and traced variables, on the 
running processes and reactions, and the environmental conditions in 
this compartement.

In the sequence of defining a system with boxsimu:
  Instances of the class Box are intatiated after all Variables, Fluids, 
  Processes, and Reactions were created.

"""

import copy
import random
from keyword import iskeyword
from attrdict import AttrDict

from . import errors as bs_errors
from . import condition as bs_condition
from . import descriptors as bs_descriptors


class Box:
    """Box in a single- or multibox system.

    A Box contains one Fluid and zero to multiple Variables.
    Boxes can exchange Fluid and Variable mass with each other.
    Additionally, Processes and Reactions can be defined that alter the 
    Variable mass in a Box.
    
    Args:
        name (str): Valid python expression used to identify the box.
        name_long (str): Human readable string describing the box.   
        fluid (Fluid): Fluid that represents the solvent of the box.
        condition (Condition): Condition of the box. Used for the evaluation
            of user-defined Process-/Reaction-/Flow-/Flux-rates.
        variables (list of Variable): Variables that are found within the box.
            The Variable instances must be quantified (that means they must
            be generated using the method q() on a Variable instance).
            Defaults to an empty list.
        processes (list of Process): Processes that take place in the box.
            Defaults to an empty list.
        reactions (list of Reaction): Reactions that take place in the box.
            Defaults to an empty list.

    Attributes:
        id (int): ID of the box within a BoxModelSystem. Note: This
            Attribute is set by the BoxModelSystem instance that
            contains a box (should not be set by the user!).
        name (str): Valid python expression used to identify the box.
        name_long (str): Human readable string describing the box.     
        fluid (Fluid): Fluid that represents the solvent of the box.
        condition (Condition): Condition of the box. Used for the evaluation
            of user-defined Process-/Reaction-/Flow-/Flux-rates.
        variables (list of Variable): Variables that are found within the box.
        processes (list of Process): Processes that take place in the box.
        reactions (list of Reaction): Reactions that take place in the box.

    """

    def __init__(self, name, name_long, fluid, condition=None, variables=None,
                 processes=None, reactions=None):
        self.name = name
        self.name_long = name_long

        if not fluid.quantified:
            raise bs_errors.FluidNotQuantifiedError('Fluid was not quantified!')
        self.fluid = fluid
        self.condition = condition if condition else bs_condition.Condition()
        self.processes = processes or []
        self.processes.sort()
        self.reactions = reactions or []
        self.reactions.sort()
        self.variables = AttrDict()
        
        variables = variables or []
        for variable in variables:
            if not variable.quantified:
                raise bs_errors.VariableNotQuantifiedError(
                    'Variable was not quantified!')
            self.variables[variable.name] = variable

        variable_names = [var_name for var_name, var in self.variables.items()]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError('Variable names have to be unique!')

    def __hash__(self):
        return hash(repr(self))

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
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        # Make id an immutable attribute
        if hasattr(self, '_id'):
            raise AttributeError('Can\'t set immutable attribute')
        self._id = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        # Make name an immutable attribute
        if hasattr(self, '_name'):
            raise AttributeError('Can\'t set immutable attribute')
        if not value.isidentifier() or iskeyword(value):
            raise ValueError('Name must be a valid python expression!')
        self._name = value

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

    @property
    def pint_ur(self):
        return fluid.mass._REGISTRY

    def get_volume(self, context=None):
        return self.fluid.get_volume(context)

    def get_concentration(self, variable):
        """Return the mass concentration [kg/kg] of variable."""
        if self.mass.magnitude > 0:
            concentration = self.variables[variable.name].mass / self.mass
            concentration = concentration.to_base_units()
            return concentration
        return 0 * self.pint_ur.dimensionless

    def get_vconcentration(self, variable, context=None):
        """Return the volumetric concentration [kg/m^3] of variable."""
        volume = self.get_volume(context)
        if volume.magnitude > 0:
            concentration = self.variables[variable.name].mass / volume
            concentration = concentration.to_base_units()
            return concentration
        return 0 * self.pint_ur.kg / self.pint_ur.meter**3

    # REPRESENTATION functions
    
    def save_as_svg(self, filename):
        if '.' not in filename:
            filename += '.svg'
        system_svg_helper = bs_visualize.BoxModelSystemSvgHelper()
        system_svg_helper.save_box_as_svg(box=self, filename=filename)



