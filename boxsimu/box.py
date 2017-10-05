# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 11:08UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

A Box represents a box within a sinlge- or multibox-system.

A Box instance represents one compartement in a system. A box is most often
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

from . import condition as bs_condition
from . import descriptors as bs_descriptors
from . import errors as bs_errors
from . import ur


class Box:
    """Box in a single- or multibox system.

    An instance of Box represents a compartement of a system that is 
    defined by specific environmental conditions, processes, and reactions.
    A specific box contains at least one instance of Variable. A variable 
    is a quantity that is modelled in the system and whose time evolution is
    of interest to the user.
    A box can be "empty space" only containing some mass of its variables or 
    it can contain an instance of Fluid which fills the complete box and 
    which serves then represents a solvent for all variables within the box.
    If a box contains no fluid it can therefore also not exchange fluid mass
    with other boxes. Thus it can not be a source or target of an instance 
    of the class Flow. However it still can exchange variable mass with 
    other boxes via fluxes (instance of the class Flux).
    
    Args:
        name (str): Valid python expression used to identify the box.
        description (str): Human readable string describing the box.   
        condition (Condition): Condition of the box. Used for the evaluation
            of user-defined Process-/Reaction-/Flow-/Flux-rates.
        fluid (Fluid): Fluid that represents the solvent of the box.
            Defaults to None.
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
            contains a box (must not be set by the user!).
        name (str): Valid python expression used to identify the box.
        description (str): Human readable string describing the box.     
        fluid (Fluid): Fluid that represents the solvent of the box.
        condition (Condition): Condition of the box. Used for the evaluation
            of user-defined Process-/Reaction-/Flow-/Flux-rates.
        variables (list of Variable): Variables that are found within the box.
        processes (list of Process): Processes that take place in the box.
        reactions (list of Reaction): Reactions that take place in the box.

    """
    id = bs_descriptors.ImmutableDescriptor('id')
    name = bs_descriptors.ImmutableIdentifierDescriptor('name')

    def __init__(self, name, description, fluid=None, condition=None, 
            variables=None, processes=None, reactions=None):
        self.name = name
        self.description = description
        
        if fluid:
            if not fluid.quantified:
                raise bs_errors.FluidNotQuantifiedError()
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
                raise bs_errors.VariableNotQuantifiedError()
            self.variables[variable.name] = variable

        variable_names = [var_name for var_name, var in self.variables.items()]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError('Variable names have to be unique!')

    def __str__(self):
        return '<Box {}>'.format(self.name)

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
    def context(self):
        context = self.condition
        for variable_name, variable in self.variables.items():
            setattr(context, variable_name, variable.mass)
        return context

    def get_volume(self, time, system):
        return self.fluid.get_volume(time, self.context, system)

    def get_concentration(self, variable):
        """Return the mass concentration [kg/kg] of variable."""
        if self.mass.magnitude > 0:
            concentration = self.variables[variable.name].mass / self.mass
            concentration = concentration.to_base_units()
            return concentration
        return 0 * ur.dimensionless

    def get_vconcentration(self, variable, time, system):
        """Return the volumetric concentration [kg/m^3] of variable."""
        volume = self.get_volume(time, self.context, system)
        if volume.magnitude > 0:
            concentration = self.variables[variable.name].mass / volume
            concentration = concentration.to_base_units()
            return concentration
        return 0 * ur.kg / ur.meter**3

    # REPRESENTATION functions
    
    def save_as_svg(self, filename):
        if '.' not in filename:
            filename += '.svg'
        system_svg_helper = bs_visualize.BoxModelSystemSvgHelper()
        system_svg_helper.save_box_as_svg(box=self, filename=filename)



