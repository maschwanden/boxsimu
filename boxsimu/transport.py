# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:57:31 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy

from . import box as bs_box
from . import condition as bs_condition
from . import entities as bs_entities
from . import errors as bs_errors
from . import descriptors as bs_descriptors
from . import validation as bs_validation
from . import function as bs_function
from . import ur


class BaseTransport:
    """Base class for the transport of fluids and variables.

    Args:
        name (str): Human readable string describing the base transport.
        source_box (Box): Box from where the fluid transport origins.
        target_box (Box): Box where the fluid transport goes.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is transported. Note: Must have dimensions 
            of [M/T].
        condition (Condition): Condition for the transport. Used for the 
            evaluation of user-defined rate and density functions. Defaults 
            to an empty Condition instance.

    Attributes:
        name (str): Human readable string describing the base transport.
        source_box (Box): Box from where the fluid transport origins.
        target_box (Box): Box where the fluid transport goes.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is transported. Note: Must have dimensions 
            of [M/T].
        condition (Condition): Condition for the transport. Used for the 
            evaluation of user-defined rate and density functions. Defaults 
            to an empty Condition instance.

    """

    name = bs_descriptors.ImmutableIdentifierDescriptor('name')

    def __init__(self, name, source_box, target_box, rate, 
            condition=None, description=None):
        self.name = name
        if source_box and not source_box.fluid:
            raise bs_errors.NoFluidInBoxError()
        if target_box and not target_box.fluid:
            raise bs_errors.NoFluidInBoxError()
        if source_box == target_box:
            raise ValueError('target_box and source_box must not be equal!')
        if not (isinstance(source_box, bs_box.Box) or 
                isinstance(target_box, bs_box.Box)):
            raise ValueError('At least one of the two parameters source_box '
                'and target_box must be an instance of the class Box!')
        self.source_box = source_box
        self.target_box = target_box
        self.rate = bs_function.UserFunction(rate, ur.kg/ur.second)
        self.condition = condition if condition else bs_condition.Condition()
        if not description:
            self.description = name

    def __str__(self):
        return '<BaseTransport {}>'.format(self.name)

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

    def __call__(self, time, context, system):
        """Instances can be called like functions."""
        return self.rate(time, context, system)

    @property
    def context(self):
        return self.condition

    @classmethod
    def get_all_from(cls, source_box, transports):
        """Return all instances of transports that origin from source_box.
        
        Args:
            source_box (Box or None): Origin of the transports.
            transports (list of BaseTransports): Transports that will 
                be considered.

        """
        if source_box is None:
            return [t for t in transports if t.source_box is None]
        elif isinstance(source_box, bs_box.Box):
            transports_with_src_box = [t for t in transports if t.source_box]
            return [t for t in transports_with_src_box 
                    if t.source_box == source_box]
        else:
            raise ValueError('{} is not a Box!'.format(source_box))

    @classmethod
    def get_all_to(cls, target_box, transports):
        """Return all instances of transports that go to target_box.
        
        Args:
            target_box (Box or None): Target of the transports.
            transports (list of BaseTransports): Transports that will 
                be considered.

        """
        if target_box is None:
            return [t for t in transports if t.target_box is None]
        elif isinstance(target_box, bs_box.Box):
            transports_with_trg_box = [t for t in transports if t.target_box]
            return [t for t in transports_with_trg_box 
                    if t.target_box == target_box]
        else:
            raise ValueError('{} is not a Box!'.format(target_box))

    @classmethod
    def get_all_source_and_target_boxes(cls, transports):
        """Return all boxes that are defined as a source or target."""
        box_list = []
        for t in transports:
            if t.source_box is not None:
                box_list.append(t.source_box)
            if t.target_box is not None:
                box_list.append(t.target_box)
        return list(set(box_list))


class Flow(BaseTransport):
    """Represent a fluid mass-transport between boxes/the outside.

    Args:
        name (str): Human readable string describing the base transport.
        source_box (Box): Box from where the transport origins.
        target_box (Box): Box where the transport goes.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is transported. Note: Must have dimensions of
            [M/T].
        tracer_transport (Boolean): Specifies if tracers (e.g. Variables) are
            passively transported with this fluid transport from one place
            to an other.
        concentrations (dict {Variable: pint:Quantity}): Concentrations of 
            variables of flows that stem from outside the system in [kg/kg]. 
            Note: Only Flows from outside the system can have fixed 
            concentrations associated with the flow!
    
    Attributes (additionals to BaseTransport):
        name (str): Human readable string describing the base transport.
        source_box (Box): Box from where the transport origins.
        target_box (Box): Box where the transport goes.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is transported. Note: Must have dimensions 
            of [M/T].
        tracer_transport (Boolean): Specifies if tracers (e.g. Variables) are
            passively transported with this fluid transport from one place
            to an other.

    """

    concentrations = bs_descriptors.PintQuantityExpValueDictDescriptor(
            'concentrations', [bs_entities.Variable], ur.kg/ur.kg)

    def __init__(self, name, source_box, target_box, rate, 
            tracer_transport=True, concentrations={}):
        if (isinstance(source_box, bs_box.Box) and 
                isinstance(target_box, bs_box.Box)):
            if source_box.fluid != target_box.fluid:
                raise bs_errors.FluidsOfBoxesNotIdenticalError()

        self.tracer_transport = tracer_transport
        self.variables = []
        self.concentrations = {}
        
        if source_box and len(concentrations) > 0:
            raise bs_errors.FixedVariableConcentrationError()
        
        self.concentrations = concentrations
        self.variables = [key for key, value in concentrations.items()]
        super().__init__(name, source_box, target_box, rate)

    def add_transported_variable(self, variable, concentration):
        """Add constant variable concentration to the flow. 
        
        Note: Only for Flows that have no source box (source_box=None)!

        """
        if self.source_box:
            raise FixedVariableConcentrationError()
        if not isinstance(variable, bs_entities.Variable):
            raise bs_errors.DictKeyNotInstanceOfError(
                    'variable_concentration_dict', 'Variable')
        bs_validation.raise_if_not_dimless(concentration)
        self.concentrations[variable] = concentration
        self.variables.append(variable)


class Flux(BaseTransport):
    """Represent a variable mass-transport between boxes/the outside.

    A Flux can for example be:
        Sedimentation of particles from the photolytic zone into the bottom
        layer of a lake.

    Args:
        name (str): Human readable string describing the base transport.
        source_box (Box): Box from where the transport origins.
        target_box (Box): Box where the transport goes.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is transported. Note: Must have dimensions of
            [M/T].

    """

    def __init__(self, name, source_box, target_box, variable, rate):
        self.variable = variable
        super().__init__(name, source_box, target_box, rate)



