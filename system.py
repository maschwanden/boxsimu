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

        self._pint_ur = None
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
        self.processes = [process for box_name, box in self.boxes.items()
                          for process in box.processes]
        self.reactions = [reaction for box_name, box in self.boxes.items()
                          for reaction in box.reactions]

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

    def _get_pint_unit_registry(self):
        pint_ur = None

        # Get pint registry from fluid masses (because at least one box
        # must exist and must have a fluid associated with a valid mass)
        for box in self.box_list:
            if not pint_ur:
                pint_ur = box.fluid.mass._REGISTRY
        return pint_ur

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
        if self._pint_ur is None:
            self._pint_ur = self._get_pint_unit_registry()
        return self._pint_ur

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
        """Return current fluid masses of all boxes.

        Args:
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function.
            only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array is returned.
                If set to True only the magnitude (in base units) of the quantities are returned.
                Therefore if numpy_array is set to True, only_magnitude is automatically set to True
                too.

        """
        m = np.zeros(self.N_boxes)

        units = []
        for box_name, box in self.boxes.items():
            box_fluid_mass = box.fluid.mass.to_base_units()  # Convert to BU
            units.append(box_fluid_mass.units)
            bs_dim_val.dimensionality_check_mass_err(box_fluid_mass)
            m[box.id] = box_fluid_mass.magnitude
        
        units_set = set(units)
        if len(units_set) == 1:
            m_units = units_set.pop()
        elif len(units_set) == 0:
            m_units = 1
        else:
            raise DimensionalityError(units_set.pop(), units_set.pop())
        return m * m_units

    def get_variable_mass_1Darray(self, variable, 
            only_magnitude=False, numpy_array=False):
        """Return magnitude of current variable masses of all boxes.

        Args:
            variable (Variable): Variable of which the mass vector should be returned.
                only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function.
            only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array is returned.
                If set to True only the magnitude (in base units) of the quantities are returned.
                Therefore if numpy_array is set to True, only_magnitude is automatically set to True
                too.

        """
        if numpy_array:
            only_magnitude = True
            m = np.zeros(self.N_boxes)
        else:
            m = [0] * self.N_boxes

        for box_name, box in self.boxes.items():
            var_mass = box.variables[variable.name].mass.to_base_units()
            bs_dim_val.dimensionality_check_mass_err(var_mass)

            if only_magnitude:
                m[box.id] = var_mass.magnitude
            else:
                m[box.id] = var_mass
        return m

    def get_variable_concentration_1Darray(self, variable, 
            only_magnitude=False, numpy_array=False):
        """Return current variable concentration [M/M] of all boxes.

        Args:
            variable (Variable): Variable of which the concentration vector should be returned.
            only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array is returned.
                If set to True only the magnitude (in base units) of the quantities are returned.
                Therefore if numpy_array is set to True, only_magnitude is automatically set to True
                too.

        """
        if numpy_array:
            only_magnitude = True
            c = np.zeros(self.N_boxes)
        else:
            c = [0] * self.N_boxes

        for box_name, box in self.boxes.items():
            var = box.variables[variable.name]
            if box.fluid.mass.magnitude == 0 or var.mass.magnitude == 0:
                c[box.id] = 0
            else:
                concentration = (var.mass / box.fluid.mass).to_base_units()
                if only_magnitude:
                    c[box.id] = concentration.magnitude
                else:
                    c[box.id] = concentration
        return c

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def get_fluid_mass_internal_flow_2Darray(self, time, flows=None, 
            only_magnitude=False, numpy_array=False):
        """Return fluid mass exchange rates due to flows between boxes of the system.

        Return a 2D list (Matrix-like) with the fluid mass exchange rates between 
        boxes of the system. Row i of the 2D list represents the flows that go 
        away from box i (sinks). Column j of the 2D list represents the flows 
        that go towards box j (sources).

        Args:
            time (pint.Quantity [T]): Time at which the flows shall be evaluated.
            flows (list of Flow): List of the flows which should be considered. Default value
                is None. If flows==None, all flows of the system are considered.
            only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array is returned.
                If set to True only the magnitude (in base units) of the quantities are returned.
                Therefore if numpy_array is set to True, only_magnitude is automatically set to True
                too.

        """
        if numpy_array:
            only_magnitude = True
            A = np.zeros([self.N_boxes, self.N_boxes])
        else:
            A = [[0 for col in range(self.N_boxes)]
                 for row in range(self.N_boxes)]

        flows = flows or self.flows

        for flow in flows:
            src_box = flow.source_box
            trg_box = flow.target_box

            if src_box is None or trg_box is None:
                continue

            src_box_context = self.get_box_context(src_box)
            mass_flow_rate = flow(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(mass_flow_rate)

            if only_magnitude:
                if numpy_array:
                    A[src_box.id, trg_box.id] += mass_flow_rate.magnitude
                else:
                    A[src_box.id][trg_box.id] += mass_flow_rate.magnitude
            else:
                A[src_box.id][trg_box.id] += mass_flow_rate
        return A

    def get_fluid_mass_flow_sink_1Darray(self, time, flows=None, 
            only_magnitude=False, numpy_array=False):
        """Return fluid mass sinks due to flows out of the system.

        Return 1D list with fluid mass sinks due to flows out of the system.
        Row i of the 1D list represents the fluid mass sink of box i out
        of the system.

        Args:
            time (pint.Quantity [T]): Time at which the flows shall be evaluated.
            flows (list of Flow): List of the flows which should be considered. Default value
                is None. If flows==None, all flows of the system are considered.
            only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array is returned.
                If set to True only the magnitude (in base units) of the quantities are returned.
                Therefore if numpy_array is set to True, only_magnitude is automatically set to True
                too.

        """
        print('numpy_array', numpy_array)
        if numpy_array:
            only_magnitude = True
            s = np.zeros(self.N_boxes)
        else:
            s = [0] * self.N_boxes

        for val in s:
            print('val', val)
            print('type(val)', type(val))

        flows = flows or self.flows
        sink_flows = bs_transport.Flow.get_all_to(None, flows)

        for flow in sink_flows:
            src_box = flow.source_box

            src_box_context = self.get_box_context(src_box)
            mass_flow_rate = flow(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(mass_flow_rate)
            
            print('mass_flow_rate', mass_flow_rate)
            print('type(mass_flow_rate)', type(mass_flow_rate))
            if only_magnitude:
                s[src_box.id] += mass_flow_rate.magnitude
            else:
                s[src_box.id] += mass_flow_rate
        
        print('###################################################################################')
        return s

    def get_fluid_mass_flow_source_1Darray(self, time, flows=None, 
            only_magnitude=False, numpy_array=False):
        """Return fluid mass sources due to flows from outside the system.

        Return 1D list with fluid mass sources due to flows from outside the 
        system. Row i of the 1D list represents the fluid mass sources from 
        outside the system into box i.

        Args:
            time (pint.Quantity [T]): Time at which the flows shall be evaluated.
            flows (list of Flow): List of the flows which should be considered. Default value
                is None. If flows==None, all flows of the system are considered.
            only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array is returned.
                If set to True only the magnitude (in base units) of the quantities are returned.
                Therefore if numpy_array is set to True, only_magnitude is automatically set to True
                too.

        """
        if numpy_array:
            only_magnitude = True
            q = np.zeros(self.N_boxes)
        else:
            q = [0] * self.N_boxes

        flows = flows or self.flows
        source_flows = bs_transport.Flow.get_all_from(None, flows)

        for flow in source_flows:
            src_box = flow.source_box
            trg_box = flow.target_box

            trg_box_context = self.get_box_context(trg_box)
            mass_flow_rate = flow(time, trg_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(mass_flow_rate)

            if only_magnitude:
                q[trg_box.id] += mass_flow_rate.magnitude
            else:
                q[trg_box.id] += mass_flow_rate
        return q

    #####################################################
    # Variable Sink/Source Vectors/Matrices
    #####################################################

    # FLOW

    def get_variable_internal_flow_2Darray(self, variable, time, f_flow, 
            flows=None, only_magnitude=False, numpy_array=False):
        """Return variable exchange rates between the boxes due to flows.

        Return 2D list of variable exchange rates due to fluid flows between 
        the boxes and the corresponding passive transport of variable.

        Args:
            variable (Variable): Variable of which the sink vector should be returned.
            time (pint.Quantity [T]): Time at which the flows shall be evaluated.
            f_flow (1D array): Reduction of the mass flow coefficients due to mass
                conservation constraints (if an box is empty no fluid can flow away from this
                box). Coefficients have values in the range [0,1]. These coefficents are
                returned from Solver.calculate_mass_flows.
            flows (list of Flow): List of the flows which should be considered. Default value
                is None. If flows==None, all flows of the system are considered.
            only_magnitude (Boolean): Defines whether Quantities or floats are returned
                from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array is returned.
                If set to True only the magnitude (in base units) of the quantities are returned.
                Therefore if numpy_array is set to True, only_magnitude is automatically set to True
                too.

        """
        if numpy_array:
            only_magnitude = True
            A = np.zeros([self.N_boxes, self.N_boxes])
        else:
            A = [[0 for col in range(self.N_boxes)]
                 for row in range(self.N_boxes)]

        flows = flows or self.flows

        for flow in flows:
            src_box = flow.source_box
            trg_box = flow.target_box

            if src_box is None or trg_box is None:
                continue

            src_box_context = self.get_box_context(src_box)
            mass_flow_rate = flow(time, src_box_context)
            bs_dim_val.dimensionality_check_mass_transport_err(mass_flow_rate)

            var_concentration = src_box.get_concentration(
                variable, src_box_context)
            bs_dim_val.dimensionality_check_dimless(var_concentration)

            var_transported = mass_flow_rate * var_concentration
            var_transported = var_transported.to_base_units()
            bs_dim_val.dimensionality_check_mass_transport(var_transported)
            
            if only_magnitude:
                if numpy_array:
                    A[src_box.id, trg_box.id] += var_transported.magnitude
                else:
                    A[src_box.id][trg_box.id] += var_transported.magnitude
            else:
                A[src_box.id][trg_box.id] += var_transported
        return A

    def get_variable_flow_sink_1Darray( self, variable, time, f_flow, 
            flows=None, only_magnitude=False, numpy_array=False):
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
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        flows = flows or self.flows
        tt_sink_flows = [flow for flow in bs_transport.Flow.get_all_to(None, flows) 
                if flow.tracer_transport]
        variable_concentration = self.get_variable_concentration_1Darray(
                variable, only_magnitude, numpy_array)
        mass_flow_sink_tmp = self.get_fluid_mass_flow_sink_1Darray(
            time, flows=tt_sink_flows, only_magnitude=only_magnitude, 
            numpy_array=numpy_array)

        if numpy_array:
            f_flow = np.array(f_flow)
            # if numpy_arrays==True both vectors, mass_flow_sink and 
            # variable_concentration, will have values in base units
            # and only their magnitude is contained
            mass_flow_sink = mass_flow_sink_tmp * f_flow
            s = variable_concentration * mass_flow_sink
        else:
            f_flow = list(f_flow)
            mass_flow_sink_tmp = list(mass_flow_sink_tmp)
            mass_flow_sink = [a*b for a,b in zip(mass_flow_sink_tmp, f_flow)]
            print()
            print('mass_flow_sink {}'.format(mass_flow_sink))
            print('variable_concentration {}'.format(variable_concentration))
            for a in variable_concentration:
                print('a', a)
                print('type(a)', type(a))
            for b in mass_flow_sink:
                print('b', b)
                print('type(b)', type(b))
            s = [a*b for a,b in zip(variable_concentration, mass_flow_sink)]

            print(s)
            print()
        return s

    def get_variable_flow_source_1Darray(self, variable, time, 
            flows=None, only_magnitude=False, numpy_array=False):
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
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        if numpy_array:
            only_magnitude = True
            q = np.zeros(self.N_boxes)
        else:
            q = [0] * self.N_boxes

        flows = flows or self.flows

        for flow in bs_transport.Flow.get_all_from(None, flows):
            trg_box = flow.target_box
            variable_source = (flow(time, self.get_global_context()) * 
                    flow.concentrations[variable]).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(variable_source)
            if only_magnitude:
                q[trg_box.id] += variable_source.magnitude
            else:
                q[trg_box.id] += variable_source
        return q

    # FLUX

    def get_variable_internal_flux_2Darray(self, variable, time, 
            fluxes=None, only_magnitude=False, numpy_array=False):
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
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        if numpy_array:
            only_magnitude = True
            A = np.zeros([self.N_boxes, self.N_boxes])
        else:
            A = [[0 for col in range(self.N_boxes)]
                 for row in range(self.N_boxes)]
        
        fluxes = fluxes or self.fluxes
        variable_fluxes = [flux for flux in fluxes 
                if variable in flux.variables]

        for flux in variable_fluxes:
            src_box = flux.source_box
            trg_box = flux.target_box

            if src_box is None or trg_box is None:
                continue

            src_box_context = self.get_box_context(src_box)
            variable_mass_flux_rate = flux(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(variable_mass_flux_rate)

            if only_magnitude:
                if numpy_array:
                    A[src_box.id, trg_box.id] += variable_mass_flux_rate.magnitude
                else:
                    A[src_box.id][trg_box.id] += variable_mass_flux_rate.magnitude
            else:
                A[src_box.id][trg_box.id] += variable_mass_flux_rate
        return A

    def get_variable_flux_sink_1Darray(self, variable, time, 
            fluxes=None, only_magnitude=False, numpy_array=False):
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
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        if numpy_array:
            only_magnitude = True
            s = np.zeros(self.N_boxes)
        else:
            s = [0] * self.N_boxes

        fluxes = fluxes or self.fluxes
        variable_fluxes = [flux for flux in bs_transport.Flux.get_all_to(
            None, fluxes) if variable in flux.variables]

        for flux in variable_fluxes:
            src_box = flux.source_box
            src_box_conetxt = self.get_box_context(src_box)
            variable_sink = flux(time, src_box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(variable_sink)

            if only_magnitude:
                s[src_box.id] += variable_sink.magnitude
            else:
                s[src_box.id] += variable_sink
        return s

    def get_variable_flux_source_1Darray(self, variable, time, 
            fluxes=None, only_magnitude=False, numpy_array=False):
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
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        if numpy_array:
            only_magnitude = True
            q = np.zeros(self.N_boxes)
        else:
            q = [0] * self.N_boxes

        fluxes = fluxes or self.fluxes
        variable_fluxes = [flux for flux in bs_transport.Flux.get_all_from(
            None, fluxes) if variable in flux.variables]

        for flux in variable_fluxes:
            trg_box = flux.target_box
            trg_box_conetxt = self.get_box_context(trg_box)
            variable_source = flux(time, trg_box_conetxt).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(variable_source)

            if only_magnitude:
                q[src_box.id] += variable_source.magnitude
            else:
                q[src_box.id] += variable_source
        return q

    # PROCESS

    def get_variable_process_sink_1Darray(self, variable, time, 
            processes=None, only_magnitude=False, numpy_array=False):
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
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        if numpy_array:
            only_magnitude = True
            s = np.zeros(self.N_boxes)
        else:
            s = [0] * self.N_boxes
        
        processes = processes or self.processes
        variable_processes = [p for p in processes if variable in p.variables]
        
        for process in variable_processes:
            box_context = self.get_box_context(process.box)
            process_rate = process(time, box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(process_rate)

            # If process rate is greater than zero it is not a sink!
            if process_rate.magnitude >= 0:
                continue 

            if only_magnitude:
                s[process.box.id] += process_rate.magnitude
            else:
                s[process.box.id] += process_rate
        return s

    def get_variable_process_source_1Darray(self, variable, time, 
            processes=None, only_magnitude=False, numpy_array=False):
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
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        if numpy_array:
            only_magnitude = True
            q = np.zeros(self.N_boxes)
        else:
            q = [0] * self.N_boxes
        
        processes = processes or self.processes
        variable_processes = [p for p in processes if variable in p.variables]
        
        for process in variable_processes:
            box_context = self.get_box_context(process.box)
            process_rate = process(time, box_context).to_base_units()
            bs_dim_val.dimensionality_check_mass_transport_err(process_rate)

            # If process rate is greater than zero it is not a sink!
            if process_rate.magnitude <= 0:
                continue 

            if only_magnitude:
                q[process.box.id] += process_rate.magnitude
            else:
                q[process.box.id] += process_rate
        return q

    # REACTION

    def get_reaction_rate_3Darray(self, time, 
            reactions=None, only_magnitude=False, numpy_array=False):
        """Return all reaction rates for all variables and boxes as a 3D list.

        The primary axis are the boxes. On the secondary axis are the reactions and on the third
        axis are the variables. Therefore, for every Box there exists a 2D numpy array with all
        information of reactions rates for every variable.
        Axis 1: Box
        Axis 2: Reactions
        Axis 3: Variables

        Args:
            time (pint.Quantity [T]): Time at which the fluxes shall be 
                evaluated.
            reactions (list of Reaction): List of the reactions which should 
                be considered. Default value is None. If reactions==None, all 
                reactions of the system are considered.
            only_magnitude (Boolean): Defines whether Quantities or floats 
                are returned from this function. 
            numpy_array (Boolean): Defines whether a list or a numpy array 
                is returned. If set to True only the magnitude (in base units) 
                of the quantities are returned. Therefore if numpy_array is 
                set to True, only_magnitude is automatically set to True too.

        """
        # Initialize cube (minimal lenght of the axis of reactions is one)
        N_reactions = max(1, max([len(box.reactions)
                                  for box in self.box_list]))
        if numpy_array:
            only_magnitude = True
            C = np.zeros([self.N_boxes, N_reactions, self.N_variables])
        else:
            C = [[[0 for col in range(self.N_variables)]
                 for row in range(N_reactions)] 
                 for depth in range(self.N_boxes)]

        for box_name, box in self.boxes.items():
            for i_reaction, reaction in enumerate(box.reactions):
                for variable, coeff in \
                        reaction.variable_reaction_coefficients.items():
                    rate = reaction(time, self.get_box_context(box), variable) 

                    if only_magnitude:
                        if numpy_array:
                            C[box.id, i_reaction, variable.id] = rate.magnitude
                        else:
                            C[box.id][i_reaction][variable.id] = rate.magnitude
                    else:
                            C[box.id][i_reaction][variable.id] = rate
        return C

    # 2Darray methods

    get_all_variable_flow_sink_2Darray = bs_utils._1Darray_to_2Darray_method(
        get_variable_flow_sink_1Darray)
    get_all_variable_flow_source_2Darray = bs_utils._1Darray_to_2Darray_method(
        get_variable_flow_source_1Darray)
    get_all_variable_flux_sink_2Darray = bs_utils._1Darray_to_2Darray_method(
        get_variable_flux_sink_1Darray)
    get_all_variable_flux_source_2Darray = bs_utils._1Darray_to_2Darray_method(
        get_variable_flux_source_1Darray)
    get_all_variable_process_sink_2Darray = bs_utils._1Darray_to_2Darray_method(
        get_variable_process_sink_1Darray)
    get_all_variable_process_source_2Darray = bs_utils._1Darray_to_2Darray_method(
        get_variable_process_source_1Darray)
