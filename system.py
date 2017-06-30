# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:37:12 2016

@author: aschi
"""

import pdb
import copy
import time as time_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attrdict import AttrDict

from transport import Flow, Flux
from process import Process, Reaction
from condition import Condition
from solution import Solution
import utils


class BoxModelSystem:
    """ A BoxModelSystem represents the entierty of the simulated system. 
    
    A BoxModelSystem instance contains boxes and the fluid-flows/variable-fluxes 
    between them. Additionally global conditions that can affect affect e.g. 
    process rates can be specified.
    
    Attributes:
    - name (str): Name of the sustance. Should be a short descriptive text.
    - boxes (list of Box): Instances of the class Box. Represent sinlge boxes in the 
    entirety of the multiboxmodel.
    - flows (list of Flow): Fluid mass flows between the boxes or the boxes and the outside 
                            of the system. 
    - fluxes (list of Flux): Variable mass fluxes between the boxes or the boxes and the 
                             outside of the system.
    """
   
    def __init__(self, name, boxes, global_condition=None, fluxes=[], flows=[],):
        self.name = name
        self.boxes = AttrDict({box.name: box for box in boxes})
        self.global_condition = global_condition or Condition()
        
        self.flows = flows
        self.fluxes = fluxes
        self.processes = [process for box_name, box in self.boxes.items() for process in box.processes]
        self.reactions = [reaction for box_name, box in self.boxes.items() for reaction in box.reactions]

        # Add every variables mentioned in a process, reaction, flow, flux or 
        # box to a temporary list and add unique variables to all boxes even 
        # if they are just defined in one.
        self.variables = self._get_all_variables()
        
        # Check the uniqueness of the box names
        if len(self.boxes.keys()) != len(set(self.boxes.keys())):
            raise ValueError('Box names have to be unique!')
        
        # Set variable and box ID's for every variable in every box
        self._set_box_and_variable_ids()
        
        # Setup Mass and Concentration Matrix
        # Mass Matrix: information of how much of every 
        # variable is in the system boxes
        # Concentration Matrix: Mass of variable per mass of fluid 
        # in the system boxes
        self.mm = pd.DataFrame(columns=self.variables.keys(), index=self.boxes.keys())
        self.cm = pd.DataFrame(columns=self.variables.keys(), index=self.boxes.keys())
        for box_name, box in self.boxes.items():
            for var_name, var in box.variables.items():
                self.mm.loc[box_name, var_name] = var.mass
                self.cm.loc[box_name, var_name] = var.mass / box.fluid.mass

                
    @property            
    def box_list(self):
        return [box for box_name, box in self.boxes.items()]
    
    @property
    def N_boxes(self):
        return len(self.boxes)
    
    @property
    def N_variables(self):
        return len(self.variables)

    def _get_all_variables(self):
        """ Returns one deepcopy of every type of variable found in the system. """
        tmp_variable_list = []
        for box_name, box in self.boxes.items():
            tmp_variable_list += [var for var_name, var in box.variables.items()]
            tmp_variable_list += [process.variable for process in box.processes]
            tmp_variable_list += [reaction.variable for reaction in box.reactions]

        for flow in self.flows:
            tmp_variable_list += [var for var in flow.variables]

        tmp_variable_list += [flux.variable for flux in self.fluxes]

        var_attr_dict = AttrDict()
        for var in tmp_variable_list:
            if not var.name in var_attr_dict.keys():
                var_copy = var.get_empty_copy()
                var_attr_dict[var_copy.name] = var_copy
        return var_attr_dict

    def _set_box_and_variable_ids(self):
        box_id = 0
        for box_name, box in self.boxes.items():
            box.ID = box_id
            box_id += 1

            var_id = 0
            for var_name, var in self.variables.items():
                if not var_name in box.variables.keys():
                    box.variables[var_name] = copy.deepcopy(var)
                box.variables[var_name].ID = var_id
                self.variables[var_name].ID = var_id
                var_id += 1 

    def get_context_of_box(self, box=None):
        """
        Returns a context (AttrDict) of box for evaluating user-defined functions.
        
        context returns an AttrDict containing the box condition and 
        variables. In addition all other boxes of the whole boxmodel-system are also 
        added to the context in order to give the user-defined function access 
        to condition and variables of other boxes too.

        Attributes:
        - box (Box): Box for which the context is desired. If default value (None) is
        given a global context is returned.
        """
        if box:
            condition = box.condition
            condition.set_surrounding_condition(self.global_condition)
            context = condition
            for var_name, var in box.variables.items():
                setattr(context, var_name, var)
        else:
            condition = self.global_condition
            context = condition

        for box_name, box_i in self.boxes.items():
            setattr(context, 'box_'+box_i.name, box_i)
        # Return a deep copy of the context in order to prevent user-defined 
        # functions to alter the conditions.
        return copy.deepcopy(context) 
    
    def get_global_context(self):
        return self.get_context_of_box()

    #####################################################
    # Fluid and Variable Mass/Concentration Vectors/Matrices
    #####################################################

    def get_fluid_mass_vector(self):
        """ Returns the magnitude of the current fluid masses [kg] of all boxes. """

        m = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            box_fluid_mass = box.fluid.mass.to_base_units()
            utils.dimensionality_check_mass_err(box_fluid_mass) 
            m[box.ID] = box_fluid_mass.magnitude
        return m

    def get_variable_mass_vector(self, variable):
        """ Returns the magnitude of the current variable masses [kg] of all boxes. 
        
        Attributes:
        - variable (Variable): Variable of which the mass vector should be returned.
        """

        m = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            var_mass = box.variables[variable.name].mass.to_base_units()
            utils.dimensionality_check_mass_err(var_mass) 
            m[box.ID] = var_mass.magnitude
        return m

    def get_variable_concentration_vector(self, variable):
        """ Returns the magnitude of the current variable concentration [kg/kg] of all boxes. 
        
        Attributes:
        - variable (Variable): Variable of which the concentration vector should be returned.
        """

        c = np.zeros(self.N_boxes)

        for box_name, box in self.boxes.items():
            var = box.variables[variable.name]
            if box.fluid.mass.magnitude == 0 or var.mass.magnitude == 0:
                c[box.ID] = 0
            else:
                conc = (var.mass / box.fluid.mass).to_base_units()
                c[box.ID] = conc.magnitude
        return c

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def get_fluid_mass_flow_matrix(self, time, flows=None):
        """ Returns the magnitude of the fluid mass exchange rates [kg/s] 
        due to flows between boxes of the system. 
        
        Returns a 2D numpy array (Matrix-like) with the fluid mass exchange 
        rates between boxes of the system.
        Row i of the 2D numpy array represents the flows that go away from box i (sinks).
        Column j of the 2D numpy array represents the flows that go towards box j (sources).
        
        Attributes:
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        - flows (list of Flow): List of the flows which should be considered. Default value
        is None. If flows==None, all flows of the system are considered.
        """

        flows = flows or self.flows
        A = np.zeros([self.N_boxes, self.N_boxes])
        
        for flow in flows:
            src_box = flow.source_box
            trg_box = flow.target_box
            if src_box == None or trg_box == None:
                continue

            src_box_context = self.get_context_of_box(src_box)
            mass_flow_rate = flow(time, src_box_context)
            utils.dimensionality_check_mass_flux_err(mass_flow_rate)
            A[src_box.ID, trg_box.ID] += mass_flow_rate.magnitude
        return A

    def get_fluid_mass_flow_sink_vector(self, time, flows=None):
        """ Returns the magnitude of the fluid mass sinks [kg] due to flows out of the system. 
        
        Returns a 1D numpy array with sinks of fluid mass due to 
        flows out of the system. 
        Row i of the 1D numpy array represents the sink of box i out 
        of the system.

        Attributes:
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        - flows (list of Flow): List of the flows which should be considered. Default value
        is None. If flows==None, all flows of the system are considered.
        """

        flows = flows or self.flows
        sink_flows = Flow.get_all_to(None, flows)
        s = np.zeros(self.N_boxes)

        for flow in sink_flows:
            src_box = flow.source_box
            trg_box = flow.target_box

            src_box_context = self.get_context_of_box(src_box)
            mass_flow_rate = flow(time, src_box_context)
            utils.dimensionality_check_mass_flux_err(mass_flow_rate)
            s[src_box.ID] += mass_flow_rate.magnitude
        return s

    def get_fluid_mass_flow_source_vector(self, time, flows):
        """ Returns fluid mass sources [kg] due to flows from outside the system. 
        
        Returns a 1D numpy array with sources of fluid mass due to 
        flows from outside the system.
        Row i of the 1D numpy array represents the sources from outside 
        the system into box i.

        Attributes:
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        - flows (list of Flow): List of the flows which should be considered. Default value
        is None. If flows==None, all flows of the system are considered.
        """

        flows = flows or self.flows
        source_flows = Flow.get_all_from(None, flows)
        q = np.zeros(self.N_boxes)
        
        for flow in source_flows:
            src_box = flow.source_box
            trg_box = flow.target_box

            trg_box_context = self.get_context_of_box(trg_box)
            mass_flow_rate = flow(time, trg_box_context)
            utils.dimensionality_check_mass_flux_err(mass_flow_rate)
            q[trg_box.ID] += mass_flow_rate.magnitude
        return q

    #####################################################
    # Variable Sink/Source Vectors/Matrices
    #####################################################

    def _vector_as_matrix_method(vector_method):
        """ Returns method which instead of a vector for one variable, returns
        a matrix for all variables of the system with the columns being the outputs
        from the corresponding _get..._vector() method for all variables.
        """

        def _get_matrix_method(self, *args):
            tmp_var_flow_sink_vector_list = [0] * self.N_variables
            for variable_name, variable in self.variables.items():
                var_flow_sink_vector = vector_method(self, variable, *args)
                tmp_var_flow_sink_vector_list[variable.ID] = var_flow_sink_vector
            return np.array(tmp_var_flow_sink_vector_list).T

        return _get_matrix_method

    # FLOW

    def get_variable_flow_sink_vector(self, variable, time, f_flow, flows=None):
        """ Returns the magnitude of the variable sinks [kg] due to flows out 
        of the system for all boxes.
        
        The flow of variable from a box out of the system is returned as 
        a 1D numpy array.
        For every box all fluid mass flows that transport variable 
        (flow.tracer_transport == True) are summed up and multiplied 
        by the concentration of variable in the flow's source_box. 

        Attributes:
        - variable (Variable): Variable of which the sink vector should be returned.
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        - f_flow (1D array): Reduction of the mass flow coefficients due to mass 
        conservation constraints (if an box is empty no fluid can flow away from this
        box). Coefficients have values in the range [0,1]. These coefficents are 
        returend from Solver.calculate_mass_flows.
        - flows (list of Flow): List of the flows which should be considered. Default value
        is None. If flows==None, all flows of the system are considered.
        """

        flows = flows or self.flows
        tt_sink_flows = [flow for flow in Flow.get_all_to(None, flows) if flow.tracer_transport]
        var_conc = self.get_variable_concentration_vector(variable)
        mass_flow_sink = self.get_fluid_mass_flow_sink_vector(time, flows=tt_sink_flows) * f_flow  
        s = mass_flow_sink * var_conc  # Element wise multiplication
        return s 

    get_variable_flow_sink_matrix = _vector_as_matrix_method(get_variable_flow_sink_vector)

    def get_variable_flow_source_vector(self, variable, time, flows=None):
        """ Returns the magnitude of the variable sources [kg] due to flows from 
        outside the system for all boxes.
        
        The flow of variable from outside the system into system boxes is returned as 
        a 1D numpy array.
        For every box all fluid mass flows that have variable concentrations specified are
        summed up and multiplied with the corresponding variable concentration.

        Attributes:
        - variable (Variable): Variable of which the source vector should be returned.
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        - flows (list of Flow): List of the flows which should be considered. Default value
        is None. If flows==None, all flows of the system are considered.
        """

        flows = flows or self.flows
        q = np.zeros(self.N_boxes)

        for box_name, box in self.boxes.items():
            # For all source flows calculate their source of variable (if they even transport any tracer
            # (tracer_transport==True) and in special the given variable (variable in flow.variables)).
            # The sources are equal to the transported fluid mass * the concentration of variable.
            variable_sources = [flow(time, self.get_global_context()) * flow.concentrations[variable.name] 
                                for flow in Flow.get_all_from(None, flows) 
                                if (flow.tracer_transport==True and variable in flow.variables 
                                    and flow.target_box==box)]
            variable_source_sum = sum(variable_sources)
            try:
                q[box.ID] = variable_source_sum.magnitude
            except AttributeError:
                q[box.ID] = variable_source_sum
        return q

    get_variable_flow_source_matrix = _vector_as_matrix_method(get_variable_flow_source_vector)

    # FLUX

    def get_variable_flux_matrix(self, time, flows=None):
        """ Returns the magnitude of the variable flux exchange rates [kg/s] 
        due to variable fluxes between boxes of the system. 
        
        Returns a 2D numpy array (Matrix-like) with the variable flux exchange 
        rates between boxes of the system.
        Row i of the 2D numpy array represents the fluxes that go away from box i (sinks).
        Column j of the 2D numpy array represents the fluxes that go towards box j (sources).
        
        Attributes:
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        - fluxes (list of Flux): List of the fluxes which should be considered. Default value
        is None. If fluxes==None, all fluxes of the system are considered.
        """

        # TODO

        pass

        # flows = flows or self.flows
        # A = np.zeros([self.N_boxes, self.N_boxes])
        # 
        # for flow in flows:
        #     src_box = flow.source_box
        #     trg_box = flow.target_box
        #     if src_box == None or trg_box == None:
        #         continue

        #     src_box_context = self.get_context_of_box(src_box)
        #     mass_flow_rate = flow(time, src_box_context)
        #     utils.dimensionality_check_mass_flux_err(mass_flow_rate)
        #     A[src_box.ID, trg_box.ID] += mass_flow_rate.magnitude
        # return A

    def get_variable_flux_sink_vector(self, variable, time):
        """ Returns the magnitude of the variable sinks [kg] due to fluxes. 
        
        The sinks of variable due to processes is returned as a 1D numpy array.

        Attributes:
        - variable (Variable): Variable of which the sink vector should be returned.
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        """

        s = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            box_rates = [f(time, self.get_context_of_box(box)) for f in 
                         Flux.get_all_from(box, self.fluxes) if f.variable==variable] 
            variable_sinks = [r for r in box_rates if r < 0]
            s[box.ID] = sum(variable_sinks)
        return s

    get_variable_flux_sink_matrix = _vector_as_matrix_method(get_variable_flux_sink_vector)

    def get_variable_flux_source_vector(self, variable, time):
        """ Returns the magnitude of the variable sources [kg] due to fluxes. 
        
        The sources of variable due to processes is returned as a 1D numpy array.

        Attributes:
        - variable (Variable): Variable of which the sink vector should be returned.
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        """

        q = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            box_rates = [f(time, self.get_context_of_box(box)) for f in 
                    Flux.get_all_from(box, self.fluxes) if f.variable==variable] 
            variable_sources = [r for r in box_rates if r > 0]
            q[box.ID] = sum(variable_sources)
        return q

    get_variable_flux_source_matrix = _vector_as_matrix_method(get_variable_flux_source_vector)

    # PROCESS

    def get_variable_process_sink_vector(self, variable, time):
        """ Returns the magnitude of the variable sinks [kg] due to processes. 
        
        The sinks of variable due to processes is returned as a 1D numpy array.

        Attributes:
        - variable (Variable): Variable of which the sink vector should be returned.
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        """

        s = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            box_rates = [p(time, self.get_context_of_box(box)) for p in box.processes 
                                 if p.variable==variable] 
            variable_sinks = [r for r in box_rates if r < 0]
            s[box.ID] = sum(variable_sinks)
        return s

    get_variable_process_sink_matrix = _vector_as_matrix_method(get_variable_process_sink_vector)

    def get_variable_process_source_vector(self, variable, time):
        """ Returns the magnitude of the variable sources [kg] due to processes. 
        
        The sources of variable due to processes is returned as a 1D numpy array.

        Attributes:
        - variable (Variable): Variable of which the source vector should be returned.
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        """
        
        q = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            box_rates = [p(time, self.get_context_of_box(box)) for p in box.processes 
                                 if p.variable==variable] 
            variable_sources = [r for r in box_rates if r > 0]
            q[box.ID] = sum(variable_sources)
        return q


    get_variable_process_source_matrix = _vector_as_matrix_method(get_variable_process_source_vector)

    # REACTION

    def get_reaction_rate_cube(self, time):
        """ Returns a 3D numpy array with containing all reaction rates for all variables and boxes. 
        
        The primary axis are the boxes. On the secondary axis are the reactions and on the third 
        axis are the variables. Therefore, for every Box there exists a 2D numpy array with all 
        information of reactions rates for every variable.
        Axis 1: Box
        Axis 2: Reactions
        Axis 3: Variables

        Attributes:
        - time (pint Quantity [T]): Time at which the flows shall be evaluated.
        """

        # Initialize cube (minimal lenght of the axis of reactions is one)
        N_reactions = max(1, max([len(box.reactions) for box in self.box_list]))
        rr_cube = np.zeros([self.N_boxes, N_reactions, self.N_variables])
        
        for box_name, box in self.boxes.items():
            for i_reaction, reaction in enumerate(box.reactions):
                for variable, coeff in reaction.variable_reaction_coeff_dict.items():
                    rate = reaction(time, self.get_context_of_box(box), variable)
                    rr_cube[box.ID, i_reaction, variable.ID] = rate
        return rr_cube


    # def get_variable_reaction_sink_vector(self, time, variable):
    #     """ Returns the magnitude of the variable sinks [kg] due to reactions. 
    #     
    #     The sinks of variable due to reactions is returned as a 1D numpy array.

    #     Attributes:
    #     - time (pint Quantity [T]): Time at which the flows shall be evaluated.
    #     - variable (Variable): Variable of which the sink vector should be returned.
    #     """
    #     
    #     return np.zeros(self.N_boxes)

    # def get_variable_reaction_source_vector(self, time, variable):
    #     """ Returns the magnitude of the variable sources [kg] due to reactions. 
    #     
    #     The sources of variable due to reactions is returned as a 1D numpy array.

    #     Attributes:
    #     - time (pint Quantity [T]): Time at which the flows shall be evaluated.
    #     - variable (Variable): Variable of which the source vector should be returned.
    #     """

    #     return np.zeros(self.N_boxes)


