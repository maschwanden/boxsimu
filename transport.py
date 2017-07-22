# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:57:31 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy

from . import box as bs_box
from . import errors as bs_errors
from . import entities as bs_entities
from . import dimensionality_validation as bs_dim_val


class BaseTransport:
    """Base class for transports of fluids and variables.

    Args:
        name (str): Human readable string describing the base transport.
        source_box (Box): Box from where the transport origins.
        target_box (Box): Box where the transport goes.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is transported. Note: Must have dimensions of
            [M/T].

    Attributes:
        name (str): Human readable string describing the base transport.
        source_box (Box): Box from where the transport origins.
        target_box (Box): Box where the transport goes.
        rate (pint.Quantity or callable that returns pint.Quantity): Rate 
            at which the variable is transported. Note: Must have dimensions of
            [M/T].

    """

    def __init__(self, name, source_box, target_box, rate):
        self.name = name
        self.source_box = source_box
        self.target_box = target_box
        self.rate = rate

    def __str__(self):
        return '<BaseTransport {}: {}>'.format(self.name, self.rate)

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

    def __call__(self, *args, **kwargs):
        """Instances can be called like functions."""
        rate = self.rate  # Default of rate
        if callable(self.rate):
            rate = self.rate(*args, **kwargs)
        return rate.to_base_units()

    @classmethod
    def get_all_from(cls, source_box, flows):
        """Return all instances that origin from source_box.
        
        Args:
            source_box (Box or None): Origin of the transports.
            flows (list of Flow): Flows that whould be considered.

        """
        if source_box is None:
            return [f for f in flows if f.source_box is None]
        elif isinstance(source_box, bs_box.Box):
            return [f for f in [fb for fb in flows if fb.source_box]
                    if f.source_box == source_box]
        else:
            raise ValueError('{} is not a Box!'.format(source_box))

    @classmethod
    def get_all_to(cls, target_box, flows):
        """Return all instances that go to target_box.
        
        Args:
            target_box (Box or None): Target of the transports.
            flows (list of Flow): Flows that whould be considered.

        """
        if target_box is None:
            return [f for f in flows if f.target_box is None]
        elif isinstance(target_box, bs_box.Box):
            return [f for f in [fb for fb in flows if fb.target_box]
                    if f.target_box == target_box]
        else:
            raise ValueError('{} is not a Box!'.format(target_box))


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
            variables of flows that stem from outside the system. 
            Note: Only Flows from outside the system can have fixed 
            concentrations associated with the flow!
    
    Attributes (additionals to BaseTransport):
        tracer_transport (Boolean): Specifies if tracers (e.g. Variables) are
            passively transported with this fluid transport from one place
            to an other.
        concentrations (dict {Variable: pint:Quantity}): Concentrations of 
            variables of flows that stem from outside the system. 
            Note: Only Flows from outside the system can have fixed 
            concentrations associated with the flow!

    Notes:
        For the evaluation of the rate at runtime the following rules apply:
        - if the flow has a source box the context of this source box is 
            used as context
        - if the flow has only a target box (therefore it is a flow from the 
            outside of the system into the system) the global context of the 
            system is used as context!

    """

    def __init__(self, name, source_box, target_box, rate, 
            tracer_transport=True, concentrations={}):
        if source_box == target_box:
            raise ValueError('target_box and source_box must not be equal!')

        if not (isinstance(source_box, bs_box.Box) or 
                isinstance(target_box, bs_box.Box)):
            raise ValueError('At least one of the two parameters source_box '
                'and target_box must be an instance of the class Box!')
        if (isinstance(source_box, bs_box.Box) and 
                isinstance(target_box, bs_box.Box)):
            if source_box.fluid != target_box.fluid:
                raise ValueError('target_box.fluid and source_box.fluid '
                    'must be equal!')

        self.tracer_transport = tracer_transport

        self.variables = []
        self.concentrations = {}

        # Check if variable_concentration_dict is valid
        for variable, concentration in concentrations.items():
            if not isinstance(variable, bs_entities.Variable):
                raise ValueError('Keys of the variable_concentration_dict must be '
                    'instances of the class Variable!')
            bs_dim_val.raise_if_not_dimless(concentration)
            var_copy = copy.deepcopy(variable)
            self.variables.append(var_copy)
            self.concentrations[var_copy] = concentration

        super().__init__(name, source_box, target_box, rate)

    def __str__(self):
        return '<BaseTransport {}: {}>'.format(self.name, self.rate)

    def add_transported_variable(self, variable, concentration):
        """Add constant variable concentration to the flow. 
        
        Note: Only for Flows that have no source box (source_box=None)!

        """
        if self.source_box:
            raise ValueError( 'Fixed variable concentrations can only be '
                'set for Flows with no source_box set (source_box=None)!')
        if not isinstance(variable, bs_entities.Variable):
            raise ValueError('Keys of the variable_concentration_dict must be '
                'instances of the class Variable!')
        bs_dim_val.raise_if_not_dimless(concentration)
        self.concentrations[variable] = concentration


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
        self.name = name
        if source_box == target_box:
            raise ValueError('target_box and source_box must not be equal!')

        if not (isinstance(source_box, bs_box.Box) or 
                isinstance(target_box, bs_box.Box)):
            raise ValueError('At least one of the two parameters source_box '
                'and target_box must be an instance of the class Box!')

        self.variable = variable

        super().__init__(name, source_box, target_box, rate)

