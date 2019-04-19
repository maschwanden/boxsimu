# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 11:08UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

Box represents compartements (parts) of a systems.

A Box instance represents one compartement in a system. It contains
information about the mass of fluid (e.g. water) and variables, the
occuring processes and reactions, and the environmental conditions
in this compartement.

In the sequence of defining a system with boxsimu:
    1) Variable, Fluid
    2) Process, Reaction
 -> 3) Box
    4) Flow, Flux
    5) BoxModelSystem

"""

from attrdict import AttrDict
from keyword import iskeyword

from . import condition as bs_condition
from . import descriptors as bs_descriptors
from . import entities as bs_entities
from . import errors as bs_errors
from . import ur


class Box:
    """Represent a compartement (part) of a system.

    An instance of Box represents a compartement of a system that is
    characterized by specific environmental conditions, processes, and
    reactions. A specific box contains at least one variable (instance of
    class Variable).
    A box can contain an instance of the class Fluid which fills out the box
    and dissolves variables. If a box contains no fluid it can
    therefore also not exchange fluid mass with other boxes. However, it still
    can exchange variable mass with other boxes via fluxes (instances of the
    class Flux).

    Args:
        name (str): Valid python expression used to identify the box.
        description (str): Human readable string describing the box.
            Default: self.name
        condition (Condition): Condition of the box. Used for the evaluation
            of user-defined Process-/Reaction-/Flow-/Flux-rates.
            Default: Condition()
        fluid (Fluid): Fluid that represents the solvent of the box.
            Default: None
        variables (list of Variable): Variables that are present within the
            box.  The Variable instances must be quantified (that means they
            must be generated using the method q() on a Variable instance).
            Default: []
        processes (list of Process): Processes that take place in the box.
            Default: []
        reactions (list of Reaction): Reactions that take place in the box.
            Default: []

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
        variable_names (list of str): Names of the variables defined in the box.
            Ordered according to the variables attribute.
        processes (list of Process): Processes that take place in the box.
        reactions (list of Reaction): Reactions that take place in the box.
        mass (pint Quantity [M]): Total mass of the box. Sum of the masses of
            all variables and the fluid (if present).

    Methods:
        get_volume(time, system): Return the volume of the box (pint.Quantity
            [L^3]).
        get_concentration(variable): Return the concentration of a specific
            variable (float [1]; kg/kg)
        get_vconcentration(variable, time, system): Return the volumetric
            concentration of a specific variable (pint.Quantity [M/L^3]).


    """

    id = bs_descriptors.ImmutableDescriptor('id')
    name = bs_descriptors.ImmutableIdentifierDescriptor('name')

    def __init__(self, name, description, fluid=None, condition=None,
            variables=None, processes=None, reactions=None):
        self.name = name
        self.description = description

        if isinstance(fluid, bs_entities.Fluid):
            if not fluid.quantified:
                raise bs_errors.fluid_is_not_quantified_error
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
                raise bs_errors.variable_is_not_quantified_error
            self.variables[variable.name] = variable

        variable_names = [var_name for var_name, var in self.variables.items()]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError('Variable names must be unique!')

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
        """Return the volume of the box [L^3].

        Assumption: The volume of the box is determined by the fluid only;
            variables are dissolved and their volume is neglibible.

        """
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
        """Save a visual representation of the box as a SVG image.

        Args:
            filename (str): Absolute or relative path where the image should
                be saved.

        """
        if '.' not in filename:
            filename += '.svg'
        system_svg_helper = bs_visualize.BoxModelSystemSvgHelper()
        system_svg_helper.save_box_as_svg(box=self, filename=filename)



