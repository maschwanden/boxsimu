# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:37:12 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import pdb
import copy
import time as time_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attrdict import AttrDict
from pint.errors import DimensionalityError

# import all submodules with prefix 'bs' for BoxSimu
from . import box as bs_box
from . import transport as bs_transport
from . import process as bs_process
from . import condition as bs_condition
from . import solution as bs_solution
from . import dimension_validation as bs_dim_val
from . import utils as bs_utils


class BoxModelSystem:
    """Represent a single- or multibox system.

    BoxModelSystems contains boxes, fluid-flows/variable-fluxes
    between them, and global conditions that can affect for example
    process/reaction rates. 
    
    Args:
        name (str): Human readable string describing the system.     
        boxes (list of Box): List of all Boxes that lie within the system.
        global_condition (Condition): Default conditions for all boxes
            of the system. Default: None (-> Condition()).
        flows (list of Flow): Fluid exchange of the Boxes. Default: []
        fluxes (list of Flux): Variable exchange of the Boxes. Default: []

    Attributes:
        name (str): Human readable string describing the system.     
        boxes (AttrDict of Box): AttrDict of all Boxes that lie within the 
            system. Box names are the key, and the box instance the values
            of the AttrDict.
        box_names (list of str): List of all Box names, sorted alphabetically.
        flows (list of Flow): Fluid exchange of the Boxes.
        fluxes (list of Flux): Variable exchange of the Boxes.
        global_condition (Condition): Default conditions for all boxes
            of the system.

    """

    def __init__(self, name, boxes, global_condition=None, fluxes=[], flows=[]):
        if not len(boxes) > 0:
            raise ValueError('At least one box must be given!')
        
        self.name = name
        self.global_condition = global_condition or bs_condition.Condition()
        self.flows = flows
        self.fluxes = fluxes

        box_dict = {}
        for box in boxes:
            if not isinstance(box, bs_box.Box):
                raise ValueError('"boxes" must be a list of Box')
            box_dict[box.name] = box
        self.boxes = AttrDict(box_dict)

        self.init_system()

    def init_system(self): 
        """Define all variables in all boxes and set variable and box ids.
        
        Run by __init__ of BoxModelSystem. 
        This method must be called if (after creation) new boxes or variables
        are added.

        """
        # Check again if all values of self.boxes are Box instances
        for box_name, box in self.boxes.items():
            if not isinstance(box, bs_box.Box):
                raise ValueError('"boxes" must be a list of Box')

        # Check the uniqueness of the box names
        if len(self.boxes.keys()) != len(set(self.boxes.keys())):
            raise ValueError('Box names have to be unique!')

        self.box_names = list(self.boxes.keys())
        self.box_names.sort()
        self.processes = list(set([process 
            for box_name, box in self.boxes.items()
            for process in box.processes]))
        self.processes.sort()
        self.reactions = list(set([reaction 
            for box_name, box in self.boxes.items()
            for reaction in box.reactions]))
        self.reactions.sort()

        # Add every variables mentioned in a process, reaction, flow, flux or
        # box to a temporary list and add unique variables to all boxes even
        # if they are just defined in one.
        self.variables = self._get_variable_attr_dict()

        # Set variable and box ID's for every variable in every box
        self._set_box_and_variable_ids()

    def _get_variable_attr_dict(self):
        """Return a deepcopy of every type of variable found in the system."""
        tmp_variable_list = []
        for box_name, box in self.boxes.items():
            tmp_variable_list += [var for var_name,
                                  var in box.variables.items()]
            tmp_variable_list += [process.variable for process in box.processes]
            tmp_variable_list += [variable for reaction in box.reactions
                                  for variable in reaction.variables]

        for flow in self.flows:
            tmp_variable_list += [var for var in flow.variables]

        tmp_variable_list += [flux.variable for flux in self.fluxes]

        var_attr_dict = AttrDict()
        for var in tmp_variable_list:
            if var.name not in var_attr_dict.keys():
                var_copy = copy.deepcopy(var)
                var_copy.mass = 0 * self.pint_ur.kg
                var_attr_dict[var_copy.name] = var_copy
        return var_attr_dict

    def _set_box_and_variable_ids(self):
        """Set the id attribute of all boxes and variables.
        
        All variables that are not defined within a box are defined.
        The ids of boxes and variables are set alphabetically to the 
        name attribute. Therefore a box called 'lake' will get a 
        smaller id than a box called 'ocean'.

        """
        box_id = 0
        for box_name in self.box_names:
            box = self.boxes[box_name]
            box.id = box_id
            box_id += 1

            var_id = 0
            var_names = list(self.variables.keys())
            var_names.sort()
            for var_name in var_names:
                if var_name not in box.variables.keys():
                    box.variables[var_name] = copy.deepcopy(
                        self.variables[var_name])
                box.variables[var_name].id = var_id
                self.variables[var_name].id = var_id
                var_id += 1

    @property
    def box_list(self):
        return [self.boxes[box_name] for box_name in self.box_names]

    @property
    def N_boxes(self):
        return len(self.boxes)

    @property
    def N_variables(self):
        return len(self.variables)

    @property
    def mm(self):
        df = pd.DataFrame(
            columns=self.variables.keys(),
            index=self.boxes.keys())
        for box_name, box in self.boxes.items():
            for var_name, var in box.variables.items():
                df.loc[box_name, var_name] = var.mass
        return df

    @property
    def cm(self):
        df = pd.DataFrame(
            columns=self.variables.keys(),
            index=self.boxes.keys())
        for box_name, box in self.boxes.items():
            for var_name, var in box.variables.items():
                if box.fluid.mass > 0:
                    df.loc[box_name, var_name] = var.mass / box.fluid.mass
                else:
                    df.loc[box_name, var_name] = 0
        return df

    @property
    def pint_ur(self):
        pint_ur = None

        # Get pint registry from fluid masses (because at least one box
        # must exist and must have a fluid associated with a valid mass)
        for box in self.box_list:
            if not pint_ur:
                pint_ur = box.fluid.mass._REGISTRY
        return pint_ur

    def get_box_context(self, box=None):
        """Return context of box for evaluating user-defined functions.

        Return an AttrDict containing the box condition and variables. 
        In addition all other boxes of the boxmodel-system 
        are also added to the context in order to give the user-defined 
        function access to condition and variables of other boxes too.

        Args:
            box (Box): Box for which the context is desired. If default value (None) is
                given a global context is returned.

        """
        if box:
            condition = box.condition
            condition.set_surrounding_condition(self.global_condition)
            context = condition
            for var_name, var in box.variables.items():
                setattr(context, var_name, var.mass)
        else:
            condition = self.global_condition
            context = condition
        
        for box_name, box in self.boxes.items():
            setattr(context, box.name, AttrDict({
                    'condition': box.condition, 
                    'variables': AttrDict({variable_name: variable.mass 
                        for variable_name, variable in box.variables.items()}),
                    'box': box}))
        setattr(context, 'global_condition', self.global_condition)
        # Return a deep copy of the context in order to prevent user-defined
        # functions to alter the conditions.
        return copy.deepcopy(context)

    def get_global_context(self):
        return self.get_box_context()

    #####################################################
    # Fluid and Variable Mass/Concentration Vectors/Matrices
    #####################################################

    def get_fluid_mass_1Darray(self):
        """Return fluid masses of all boxes."""
        m = np.zeros(self.N_boxes)

        units = []
        for box_name, box in self.boxes.items():
            fluid_mass = box.fluid.mass.to_base_units()
            bs_dim_val.dimensionality_check_mass_err(fluid_mass)
            units.append(fluid_mass.units)
            m[box.id] = fluid_mass.magnitude
        
        units_set = set(units)
        if len(units_set) == 1:
            m_units = units_set.pop()
        elif len(units_set) == 0:
            m_units = self.pint_ur.kg / self.pint_ur.second
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return m * m_units

    def get_variable_mass_1Darray(self, variable):
        """Return masses of variable of all boxes.

        Args:
            variable (Variable): Variable of which the mass vector should 
                be returned.

        """
        m = np.zeros(self.N_boxes)

        units = []
        for box_name, box in self.boxes.items():
            variable_mass = box.variables[variable.name].mass.to_base_units()
            bs_dim_val.dimensionality_check_mass_err(variable_mass)
            units.append(variable_mass.units)
            m[box.id] = variable_mass.magnitude

        units_set = set(units)
        if len(units_set) == 1:
            m_units = units_set.pop()
        elif len(units_set) == 0:
            m_units = self.pint_ur.kg / self.pint_ur.second
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return m * m_units

    def get_variable_concentration_1Darray(self, variable):
        """Return concentration [M/M] of variable of all boxes.

        Args:
            variable (Variable): Variable of which the concentration vector 
                should be returned.

        """
        c = np.zeros(self.N_boxes)

        units = []
        for box_name, box in self.boxes.items():
            variable_mass = box.variables[variable.name].mass
            if box.fluid.mass.magnitude == 0 or variable_mass.magnitude == 0:
                concentration = 0 * self.pint_ur.dimensionless
            else:
                concentration = (variable_mass / box.fluid.mass).to_base_units()
            bs_dim_val.dimensionality_check_dimless_err(concentration)
            units.append(concentration.units)
            c[box.id] = concentration.magnitude

        units_set = set(units)
        if len(units_set) == 1:
            c_units = units_set.pop()
        elif len(units_set) == 0:
            c_units = self.pint_ur.kg / self.pint_ur.second
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return c * c_units

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def get_fluid_mass_internal_flow_2Darray(self, time, flows=None):
        """Return fluid mass exchange rates due to flows between boxes.

        Return a 2D list (Matrix-like) with the fluid mass exchange rates 
        between boxes of the system. Row i of the 2D list represents the 
        flows that go away from box i (sinks). Column j of the 2D list 
        represents the flows that go towards box j (sources).

        Args:
            time (pint.Quantity [T]): Time at which the flows shall be 
                evaluated.
            flows (list of Flow): List of the flows which should be 
                considered. Default value is None. If flows==None, all 
                flows of the system are considered.  

        """
        A = np.zeros([self.N_boxes, self.N_boxes])
        flows = flows or self.flows

        units = []
        for flow in flows:
            if flow.source_box is None or flow.target_box is None:
                continue
            src_box_context = self.get_box_context(flow.source_box)
            fluid_flow_rate = flow(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(fluid_flow_rate)
            units.append(fluid_flow_rate.units)
            A[flow.source_box.id, flow.target_box.id] += \
                    fluid_flow_rate.magnitude

        units_set = set(units)
        if len(units_set) == 1:
            A_units = units_set.pop()
        elif len(units_set) == 0:
            A_units = self.pint_ur.kg / self.pint_ur.second
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return A * A_units

    def get_fluid_mass_flow_sink_1Darray(self, time, flows=None):
        """Return fluid mass sinks due to flows out of the system.

        Return 1D list with fluid mass sinks due to flows out of the system.
        Row i of the 1D list represents the fluid mass sink of box i out
        of the system.

        Args:
            time (pint.Quantity [T]): Time at which the flows shall be 
                evaluated.
            flows (list of Flow): List of the flows which should be 
                considered. Default value is None. If flows==None, all 
                flows of the system are considered.

        """
        s = np.zeros(self.N_boxes)
        flows = flows or self.flows
        flows = bs_transport.Flow.get_all_to(None, flows)

        units = []
        for flow in flows:
            src_box_context = self.get_box_context(flow.source_box)
            fluid_flow_rate = flow(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(fluid_flow_rate)
            units.append(fluid_flow_rate.units)
            s[flow.source_box.id] += fluid_flow_rate.magnitude

        units_set = set(units)
        if len(units_set) == 1:
            s_units = units_set.pop()
        elif len(units_set) == 0:
            s_units = self.pint_ur.kg / self.pint_ur.second
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return s * s_units

    def get_fluid_mass_flow_source_1Darray(self, time, flows=None):
        """Return fluid mass sources due to flows from outside the system.

        Return 1D list with fluid mass sources due to flows from outside the 
        system. Row i of the 1D list represents the fluid mass sources from 
        outside the system into box i.

        Args:
            time (pint.Quantity [T]): Time at which the flows shall be 
                evaluated.
            flows (list of Flow): List of the flows which should be considered.
            Default value
                is None. If flows==None, all flows of the system are considered.

        """
        q = np.zeros(self.N_boxes)
        flows = flows or self.flows
        flows = bs_transport.Flow.get_all_from(None, flows)

        units = []
        for flow in flows:
            trg_box_context = self.get_box_context(flow.target_box)
            fluid_flow_rate = flow(time, trg_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(fluid_flow_rate)
            units.append(fluid_flow_rate.units)
            q[flow.target_box.id] += fluid_flow_rate.magnitude

        units_set = set(units)
        if len(units_set) == 1:
            q_units = units_set.pop()
        elif len(units_set) == 0:
            q_units = self.pint_ur.kg / self.pint_ur.second
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return q * q_units

    #####################################################
    # Variable Sink/Source Vectors/Matrices
    #####################################################

    def _check_mass_rate_units(self, units):
        """Check list of pint units if they are mass rates."""
        units_set = set(units)
        if len(units_set) == 1:
            res_units = units_set.pop()
        elif len(units_set) == 0:
            res_units = self.pint_ur.kg / self.pint_ur.second
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return res_units

    # FLOW

    def get_variable_internal_flow_2Darray(self, variable, time, f_flow, 
            flows=None):
        """Return variable exchange rates between the boxes due to flows.

        Return 2D list of variable exchange rates due to fluid flows between 
        the boxes and the corresponding passive transport of variable.

        Args:
            variable (Variable): Variable of which the sink vector should be 
                returned.
            time (pint.Quantity [T]): Time at which the flows shall be 
                evaluated.
            f_flow (1D array): Reduction of the mass flow coefficients due 
                to mass conservation constraints (if an box is empty no 
                fluid can flow away from this box). Coefficients have 
                values in the range [0,1]. These coefficents are returned 
                from Solver.calculate_mass_flows.
            flows (list of Flow): List of the flows which should be 
                considered. Default value is None. If flows==None, all 
                flows of the system are considered.

        """
        A = np.zeros([self.N_boxes, self.N_boxes])
        flows = flows or self.flows

        units = []
        for flow in flows:
            if flow.source_box is None or flow.target_box is None:
                continue
            src_box_context = self.get_box_context(flow.source_box)
            fluid_flow_rate = flow(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(fluid_flow_rate)
            concentration = flow.source_box.get_concentration(variable)
            bs_dim_val.dimensionality_check_dimless(concentration)

            variable_flow_rate = (fluid_flow_rate *
                    concentration).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time(variable_flow_rate)
            units.append(variable_flow_rate.units)
            A[flow.source_box.id, flow.target_box.id] += \
                    variable_flow_rate.magnitude

        A_units = self._check_mass_rate_units(units)
        return A * A_units

    def get_variable_flow_sink_1Darray( self, variable, time, f_flow, 
            flows=None): 
        """Return variable sinks due to flows out of the system for all boxes.

        The flow of variable from a box out of the system is returned as
        a 1D list.
        For every box all fluid mass flows that transport variable
        (flow.tracer_transport == True) are summed up and multiplied
        by the concentration of variable in the flow's source_box.

        Args:
            variable (Variable): Variable of which the sink vector should 
                be returned.
            time (pint.Quantity [T]): Time at which the flows shall be 
                evaluated.
            f_flow (1D array): Reduction of the mass flow coefficients 
                due to mass conservation constraints (if an box is empty 
                no fluid can flow away from this box). Coefficients have 
                values in the range [0,1]. These coefficents are returend 
                from Solver.calculate_mass_flows.
            flows (list of Flow): List of the flows which should be 
                considered. Default value is None. If flows==None, all 
                flows of the system are considered.

        """
        flows = flows or self.flows
        flows = [flow for flow in bs_transport.Flow.get_all_to(None, flows) 
                    if flow.tracer_transport]
        concentrations = self.get_variable_concentration_1Darray(variable)
        fluid_flow_rates = self.get_fluid_mass_flow_sink_1Darray(time, 
                flows=flows) * f_flow
        s = concentrations * fluid_flow_rates
        return s

    def get_variable_flow_source_1Darray(self, variable, time, flows=None): 
        """Return variable sources due to flows from outside the system.

        The flow of variable from outside the system into system boxes 
        is returned as a 1D list.
        For every box all fluid mass flows that have variable
        concentrations specified are summed up and multiplied with the 
        corresponding variable concentration.

        Args:
            variable (Variable): Variable of which the sink vector should 
                be returned.
            time (pint.Quantity [T]): Time at which the flows shall be 
                evaluated.
            flows (list of Flow): List of the flows which should be 
                considered. Default value is None. If flows==None, all 
                flows of the system are considered.

        """
        q = np.zeros(self.N_boxes)
        flows = flows or self.flows
        variable_flows = [f for f in flows 
                if variable in f.concentrations.keys()]

        units = []
        for flow in bs_transport.Flow.get_all_from(None, variable_flows):
            variable_flow_rate = (flow(time, self.get_global_context()) * 
                    flow.concentrations[variable]).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(
                    variable_flow_rate)
            units.append(variable_flow_rate.units)
            q[flow.target_box.id] += variable_flow_rate.magnitude

        q_units = self._check_mass_rate_units(units)
        return q * q_units

    # FLUX

    def get_variable_internal_flux_2Darray(self, variable, time, fluxes=None): 
        """Return variable flux exchange rates between boxes of the system.

        Returns a 2D list (Matrix-like) with the variable flux exchange
        rates between boxes of the system.
        Row i of the 2D list represents the fluxes that go away from 
        box i (sinks). Column j of the 2D list represents the fluxes 
        that go towards box j (sources).

        Args:
            variable (Variable): Variable of which the sink vector should 
                be returned.
            time (pint.Quantity [T]): Time at which the fluxes shall be 
                evaluated.
            fluxes (list of Flow): List of the fluxes which should be 
                considered. Default value is None. If fluxes==None, all 
                fluxes of the system are considered.

        """
        A = np.zeros([self.N_boxes, self.N_boxes])
        fluxes = fluxes or self.fluxes
        variable_fluxes = [flux for flux in fluxes 
                if variable == flux.variable]

        units = []
        for flux in variable_fluxes:
            if flux.source_box is None or flux.target_box is None:
                continue

            src_box_context = self.get_box_context(flux.source_box)
            flux_rate = flux(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(flux_rate)
            units.append(flux_rate.units)
            A[flux.source_box.id, flux.target_box.id] += flux_rate.magnitude

        A_units = self._check_mass_rate_units(units)
        return A * A_units

    def get_variable_flux_sink_1Darray(self, variable, time, fluxes=None):
        """Return variable sinks due to fluxes out of the system.

        The sinks of variable due to fluxes out of the system is 
        returned as a 1D list.

        Args:
            variable (Variable): Variable of which the sink vector should 
                be returned.
            time (pint.Quantity [T]): Time at which the fluxes shall be 
                evaluated.
            fluxes (list of Flow): List of the fluxes which should be 
                considered. Default value is None. If fluxes==None, all 
                fluxes of the system are considered.

        """
        s = np.zeros(self.N_boxes)
        fluxes = fluxes or self.fluxes
        variable_fluxes = [flux for flux in bs_transport.Flux.get_all_to(
            None, fluxes) if variable == flux.variable]

        units = []
        for flux in variable_fluxes:
            src_box_context = self.get_box_context(flux.source_box)
            flux_rate = flux(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(flux_rate)
            units.append(flux_rate.units)
            s[flux.source_box.id] += flux_rate.magnitude

        s_units = self._check_mass_rate_units(units)
        return s * s_units

    def get_variable_flux_source_1Darray(self, variable, time, fluxes=None):
        """Return variable sources due to fluxes from outside the system.

        The variable sources for every box due to fluxes from outside the 
        system is returned as a 1D list.

        Args:
            variable (Variable): Variable of which the sink vector should 
                be returned.
            time (pint.Quantity [T]): Time at which the fluxes shall be 
                evaluated.
            fluxes (list of Flow): List of the fluxes which should be 
                considered. Default value is None. If fluxes==None, all 
                fluxes of the system are considered.

        """
        q = np.zeros(self.N_boxes)
        fluxes = fluxes or self.fluxes
        variable_fluxes = [flux for flux in bs_transport.Flux.get_all_from(
            None, fluxes) if variable == flux.variable]

        units = []
        for flux in variable_fluxes:
            trg_box_conetxt = self.get_box_context(flux.target_box)
            flux_rate = flux(time, trg_box_conetxt).to_base_units()
            bs_dim_val.dimensionality_check_mass_per_time_err(flux_rate)
            units.append(flux_rate.units)
            q[flux.target_box.id] += flux_rate.magnitude

        q_units = self._check_mass_rate_units(units)
        return q * q_units

    # PROCESS

    def get_variable_process_sink_1Darray(self, variable, time, processes=None):
        """Return variable sinks due to processes.

        The variable sources for every box due to processes is returned 
        as a 1D list.

        Args:
            variable (Variable): Variable of which the sink vector should 
                be returned.
            time (pint.Quantity [T]): Time at which the fluxes shall be 
                evaluated.
            processes (list of Process): List of the processes which should 
                be considered. Default value is None. If processes==None, all 
                processes of the system are considered.

        """
        s = np.zeros(self.N_boxes)
        processes = processes or self.processes
        variable_processes = [p for p in processes if variable == p.variable]
        variable_process_names = [p.name for p in variable_processes]
        
        units = []
        for box_name, box in self.boxes.items():
            box_context = self.get_box_context(box)
            box_processes = [p for p in box.processes 
                    if p.name in variable_process_names]
            box_process_rates = [p(time, box_context).to_base_units() 
                    for p in box_processes]
            for rate in box_process_rates:
                bs_dim_val.dimensionality_check_mass_per_time_err(rate)
                units.append(rate.units)
            sink_rates = [-rate for rate in box_process_rates 
                    if rate.magnitude < 0]
            try:
                s[box.id] += sum(sink_rates).magnitude
            except AttributeError:
                s[box.id] += sum(sink_rates)

        s_units = self._check_mass_rate_units(units)
        return s * s_units

    def get_variable_process_source_1Darray(self, variable, time, 
            processes=None):
        """ Returns the magnitude of the variable sources [kg] due to processes.

        The sources of variable due to processes is returned as a 1D numpy array.

        Args:
            variable (Variable): Variable of which the sink vector should 
                be returned.
            time (pint.Quantity [T]): Time at which the fluxes shall be 
                evaluated.
            processes (list of Process): List of the processes which should 
                be considered. Default value is None. If processes==None, all 
                processes of the system are considered.

        """
        q = np.zeros(self.N_boxes)
        processes = processes or self.processes
        variable_processes = [p for p in processes if variable == p.variable]
        variable_process_names = [p.name for p in variable_processes]
        
        units = []
        for box_name, box in self.boxes.items():
            box_context = self.get_box_context(box)
            box_processes = [p for p in box.processes 
                    if p.name in variable_process_names]
            box_process_rates = [p(time, box_context).to_base_units() 
                    for p in box_processes]
            for rate in box_process_rates:
                bs_dim_val.dimensionality_check_mass_per_time_err(rate)
                units.append(rate.units)
            source_rates = [rate for rate in box_process_rates 
                    if rate.magnitude > 0]
            try:
                q[box.id] += sum(source_rates).magnitude
            except AttributeError:
                q[box.id] += sum(source_rates)

        q_units = self._check_mass_rate_units(units)
        return q * q_units

    # REACTION

    def get_reaction_rate_2Darray(self, time, reaction):
        """Return reaction rates for all variables and boxes."""
        A = np.zeros([self.N_boxes, self.N_variables])
        
        units = []
        for box in self.box_list:
            for variable_name, variable in self.variables.items():
                if not reaction in box.reactions:
                    A[box.id, variable.id] = 0 
                    continue
                rate = reaction(time, self.get_box_context(box), variable)
                rate = rate.to_base_units()
                bs_dim_val.dimensionality_check_mass_per_time_err(rate)
                units.append(rate.units)
                A[box.id, variable.id] = rate.magnitude

        A_units = self._check_mass_rate_units(units)
        return A * A_units


    def get_reaction_rate_3Darray(self, time, reactions=None): 
        """Return all reaction rates for all variables and boxes as a 3D list.

        The primary axis are the boxes. On the secondary axis are the reactions and on the third
        axis are the variables. Therefore, for every Box there exists a 2D numpy array with all
        information of reactions rates for every variable.
        Axis 1: Box
        Axis 2: Variables
        Axis 3: Reactions

        Note: The first and the second axis are constant, that means that 
        the same index of the first and second axis always point to the 
        same box/variable. On the other hand, reactions are just filled in
        in any order. That means that the same index of the third axis doesn't
        show the reaction rates of the same reaction. If for example in one 
        box there are four reactions: R1, R2, R3, R4. Then the third axis of 
        this box will contain: [R1,R2,R3,R4]. For another box that only has
        R3 the third axis will look: [R3,0,0,0]. Thus the third axis is filled
        from 0.

        Args:
            time (pint.Quantity [T]): Time at which the fluxes shall be 
                evaluated.
            reactions (list of Reaction): List of the reactions which should 
                be considered. Default value is None. If reactions==None, all 
                reactions of the system are considered.

        """
        # Initialize cube (minimal lenght of the axis of reactions is one)
        N_reactions = len(self.reactions)
        C = np.zeros([N_reactions, self.N_boxes, self.N_variables])

        units = []
        for i, reaction in enumerate(self.reactions):
            reaction_2Darray = self.get_reaction_rate_2Darray(time, reaction)
            bs_dim_val.dimensionality_check_mass_per_time_err(reaction_2Darray)
            units.append(reaction_2Darray.units)
            C[i,:,:] = reaction_2Darray.magnitude 

        C_units = self._check_mass_rate_units(units)
        return np.moveaxis(C, 0, -1) * C_units



