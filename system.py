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

from transport import Flow
from process import Process, Reaction
from condition import Condition
from solution import Solution

from utils import (dimensionality_check, dimensionality_check_err, 
        dimensionality_check_mass_flux_err, dimensionality_check_mass_err)

from pint import UnitRegistry
ur = UnitRegistry(autoconvert_offset_to_baseunit = True)

DEBUG = False

def dprint(*args):
    if DEBUG: print(*args)

def dinput(*args):
    if DEBUG:
        return input(*args)


class BoxModelSystem:
    """ The BoxModelSystem containts all boxes and the flows/fluxes between them.
    
    Attributes:
    - name: Name of the sustance. Should be a short describtive text.
    - boxes: Instances of the class Box. Represent sinlge boxes in the 
    entirety of the multiboxmodel.
    - flows: Volume flows between the boxes. This is always a volume flow of
             the fluid within the box.
    - fluxes: Mass fluxes between the boxes. This can for example be a flux
              of phosphate per hour.
    """
   
    def __init__(self, name, boxes, global_condition, fluxes=[], flows=[],):
        self.name = name
        self.boxes = AttrDict({box.name: box for box in boxes})
        self.global_condition = global_condition
        
        self.variables = AttrDict({})
        self.flows = flows
        self.fluxes = fluxes

        # Mass and Concentration Matrix: information of how much of every 
        # variable is in every box
        self.mm = None
        self.cm = None
        
        # Check the uniqueness of the box names
        box_names = [box_name for box_name, box in self.boxes.items()]
        if len(box_names) != len(set(box_names)):
            raise ValueError('Box names have to be unique!')
        
        # Add the variables to all boxes even if they are just defined in one
        self.variable_names = []
        
        box_id = 0
        for box_name, box in self.boxes.items():
            box.ID = box_id
            box_id += 1
            box.system = self
            box.set_global_condition(global_condition)
        
            for var_name, var in box.variables.items():
                if not var_name in self.variables.keys():
                    var_copy = copy.deepcopy(var)
                    var_copy.mass *= 0
                    self.variables[var_name] = var_copy
                    self.variable_names.append(var_name)

        for flow in Flow.get_all_from(None, self.flows):
            for var in flow.variables:
                if not var.name in self.variables.keys():
                    var_copy = copy.deepcopy(var)
                    var_copy.mass *= 0
                    self.variables[var_name] = var_copy
                    self.variable_names.append(var_name)
                    
        # Add the variables that are not defined in a box
        # And: Set variable ID's for every variable in every box
        for box_name, box in self.boxes.items():
            var_id = 0
            for var_name, var in self.variables.items():
                if not var_name in box.variables.keys():
                    box.variables[var_name] = copy.deepcopy(var)
                box.variables[var_name].ID = var_id
                var_id += 1    

        for var_name, var in self.variables.items():
            self.variables[var_name] = var

        system_variable_names = [var_name for var_name, var in 
                                 self.variables.items()]
        
        # initialize mass and concentration matrix: 
        self.mm = pd.DataFrame(columns=system_variable_names, index=box_names)
        self.cm = pd.DataFrame(columns=system_variable_names, index=box_names)
        for box_name, box in self.boxes.items():
            for var_name, var in box.variables.items():
                self.mm.loc[box_name, var_name] = var.mass
                self.cm.loc[box_name, var_name] = var.mass / box.fluid.mass
                
        self.processes = [process for box_name, box in self.boxes.items() for process in box.processes]
        self.reactions = [reaction for box_name, box in self.boxes.items() for reaction in box.reactions]



    @property            
    def box_list(self):
        return [box for box_name, box in self.boxes.items()]
    
    @property
    def N_boxes(self):
        return len(self.boxes)
    
    @property
    def N_variables(self):
        return len(self.variables)
    
    def get_fluid_mass_flow_matrix(self, time, flows):
        """ Returns fluid mass exchange rates due to flows between boxes of the system. 
        
        Returns a 2D numpy array (Matrix-like) with the sum of fluid mass exchange rates between boxes of the system.
        Row i of the 2D numpy array represents the flows that go away from box i (sinks).
        Column j of the 2D numpy array represents the flows that go towards box j (sources).
        """
        N_boxes = self.N_boxes
        A = np.zeros([N_boxes, N_boxes])
        
        for flow in flows:
            src_box = flow.source_box
            trg_box = flow.target_box
            if src_box == None or trg_box == None:
                continue
            mass_flow_rate = flow.mass_flow_rate(time, src_box.context).to_base_units()
            dimensionality_check_mass_flux_err(mass_flow_rate)
            A[src_box.ID, trg_box.ID] += mass_flow_rate.magnitude
        return A
            
    def get_fluid_mass_flow_source_vector(self, time, flows):
        """ Returns fluid mass sources due to flows from the outside of the system. 
        
        Returns a 1D numpy array with the sum of sources of fluid mass due to 
        flows from the outside of the system within a box of the system for every box.
        """
        q = np.zeros(self.N_boxes)
        
        sources = Flow.get_all_from(None, flows)
        for flow in sources:
            src_box = flow.source_box
            trg_box = flow.target_box
            if src_box == None and trg_box == None:
                continue
            mass_flow_rate = flow.mass_flow_rate(time, trg_box.context).to_base_units()
            dimensionality_check_mass_flux_err(mass_flow_rate)
            q[trg_box.ID] += mass_flow_rate.magnitude
        return q
    
    def get_fluid_mass_flow_sink_vector(self, time, flows):
        """ Returns fluid mass sinks due to flows out of the system. 
        
        Returns a 1D numpy array with the sum of sinks of fluid mass due to 
        flows out of the system for every box.
        """
        s = np.zeros(self.N_boxes)
        
        sinks = Flow.get_all_to(None, flows)
        for flow in sinks:
            src_box = flow.source_box
            trg_box = flow.target_box
            if src_box == None and trg_box == None:
                continue
            mass_flow_rate = flow.mass_flow_rate(time, src_box.context).to_base_units()
            dimensionality_check_mass_flux_err(mass_flow_rate)
            s[src_box.ID] += mass_flow_rate.magnitude
        return s
    
    def get_fluid_mass_vector(self):
        m = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            box_fluid_mass = box.fluid.mass.to_base_units()
            dimensionality_check_mass_flux_err(box_fluid_mass) 
            m[box.ID] = box_fluid_mass.magnitude
        return m

    def get_variable_mass_vector(self, variable):
        m = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            var_mass = box.variables[variable.name].mass.to_base_units()
            dimensionality_check_mass_flux_err(var_mass) 
            m[box.ID] = var_mass
        return m

    def get_variable_concentration_vector(self, variable):
        # TODO
        c = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            var = box.variables[variable.name]
            #print('varmass: ', var.mass)
            #print('fluidmass', box.fluid.mass)
            if box.fluid.mass.magnitude == 0 or var.mass.magnitude == 0:
                c[box.ID] = 0
                c_units = 1
            elif var.mass > 0.5 * box.fluid.mass:
                print('VARMASS > 0.5 * FLUIDMASS!!!')                
                conc = (var.mass / box.fluid.mass).to_base_units()
                c[box.ID] = conc.magnitude
                c_units = conc.units
            else:
                conc = (var.mass / box.fluid.mass).to_base_units()
                c[box.ID] = conc.magnitude
                c_units = conc.units
        return c, c_units

    def get_variable_flow_sink_vector(self, time, variable, f_flow):
        # TODO: critically check if this function is correct
        var_conc, var_conc_units = self.get_box_variable_concentration_vector(variable)
        flow_s = mass_flow_sink_vector * var_conc
        units = 1
        if len(flow_s) > 0:
            units = flow_s[0].units

        return flow_s, units 

    def get_variable_flow_source_vector(self, time, variable, f_flow):
        # TODO: critically check if this function is correct
        q = np.zeros(self.N_boxes)
        mass_flow_source_vector = self.get_fluid_mass_flow_source_vector(
        tt_source_flows = [flow for flow in Flow.get_all_from(None, self.flows) 
                           if (flow.tracer_transport==True and variable in flow.variables)]
        for box_name, box in self.boxes.items():
            box_flow_sources = [flow(time, box.context) * mass_flow_source_vector[box.ID] for flow in
                             tt_source_flows if flow.target_box==box]) 
            q[box.ID] = sum([source.magnitude for source in box_flow_sources])
        return q, box_flow_sources.units 
            
    def get_variable_process_sink_vector(self, time, variable):
        p_s = np.zeros(self.N_boxes)
        units = 1
        for box_name, box in self.boxes.items():
            box_process_sink_rates = [p(time, box.context) for p in box.processes 
                                      if p.variable==variable and p(time, box.context).magnitude < 0 ]
            p_s[box.ID] = sum(box_process_sink_rates)
            if len(box_process_sink_rates) > 0:
                units = box_process_sink_rates[0].units
        return p_s, units

    def get_variable_process_source_vector(self, time, variable):
        p_q = np.zeros(self.N_boxes)
        units = 1
        for box_name, box in self.boxes.items():
            box_process_source_rates = [p(time, box.context) for p in box.processes 
                                      if p.variable==variable and p(time, box.context).magnitude >= 0 ]
            p_q[box.ID] = sum(box_process_source_rates)
            if len(box_process_source_rates) > 0:
                units = box_process_source_rates[0].units
        return p_q, units

    def get_variable_reaction_sink_vector(self, time, variable):
        # TODO: not correct right now!! since the function _get_variable_x_sink_vector doesnt distinguish between the different boxes!
        reaction_rates =  [r(time, box.context)*r.get_coeff_of(variable) for box_name, box in self.boxes.items() 
                           for r in box.reactions if variable in r.variables]
        return self._get_variable_x_sink_vector(time, reaction_rates)

    def get_variable_reaction_source_vector(self, time, variable):
        # TODO: not correct right now!! since the function _get_variable_x_sink_vector doesnt distinguish between the different boxes!
        reaction_rates =  [r(time, box.context)*r.get_coeff_of(variable) for box_name, box in self.boxes.items() 
                           for r in box.reactions if variable in r.variables]
        return self._get_variable_x_source_vector(time, reaction_rates)

    def get_variable_flux_sink_vector(self, time, variable):
        # TODO: not correct right now!! since the function _get_variable_x_sink_vector doesnt distinguish between the different boxes!
        flux_rates =  [f(time, f.source_box.context) for f in self.fluxes if f.variable==variable]
        return self._get_variable_x_sink_vector(time, flux_rates)

    def get_variable_flux_source_vector(self, time, variable):
        # TODO: not correct right now!! since the function _get_variable_x_sink_vector doesnt distinguish between the different boxes!
        flux_rates =  [f(time, f.source_box.context) for f in self.fluxes if f.variable==variable]
        return self._get_variable_x_source_vector(time, flux_rates)

    
    
    
    def calculate_mass_flows(self, time, dt):
        """ Calculates the mass changes in every box.

        Attributes:
        - time (pint quantity): time at which the mass flows should be
                                calculated
        - dt (pint quantity): timestep used

        Returns:
        - mass_changes (numpy array): Array with the magnitude of mass 
                                      changes for every box
        - mass_units (numpy array): Units of mass_changes
        - A (numpy 2D array): Flow matrix.
        """
        # f is the reduction coefficent of the "sink-flows" of each box
        f = np.ones(self.N_boxes) # scaling factor for sinks of each box
        m, m_units = self.get_box_fluid_mass_vector()
        m_ini = copy.deepcopy(m)
        # A, q, and s all have units of mass flux [kg/s]
        A = self.get_flow_matrix(time, self.flows)
        q = self.get_flow_source_vector(time, self.flows)
        s = self.get_flow_sink_vector(time, self.flows)
        
        dimless_dt = ((self.flows[0].units/m_units)*dt).to_base_units().magnitude
        
        m_change = (q - s - np.dot(A,np.ones(self.N_boxes)) + 
                     np.dot((A).T, np.ones(self.N_boxes)))*dimless_dt
        m = m_ini + m_change
        
        while np.any(m<0):
            argmin = np.argmin(m)
            flow_source_arr = A[:,np.argmin(m)]
            flow_sink_arr = A[np.argmin(m),:]
            source = q[np.argmin(m)]
            sink = s[np.argmin(m)]

            available_mass = m_ini[argmin]
            net_source =  sum(flow_source_arr) + source
            net_sink = sum(flow_sink_arr) + sink
            
            if (net_source+available_mass) > 0:
                f[argmin] = min(
                        float(net_source+available_mass)/float(net_sink),
                        f[argmin] * 0.95
                )
            else:
                f[argmin] =  0
            
            # Apply reduction of sinks of the box
            #dprint('A before correction: ', A)
            #dprint('f for correction: ', f)
            A = (A.transpose()*f).transpose()
            #dprint('A after correction: ', A)
            s = f*s
            m_change = (q - s - np.dot(A,np.ones(self.N_boxes)) + 
                     np.dot((A).T, np.ones(self.N_boxes)))*dimless_dt
            m = m_ini + m_change
        return m_change, m_units, f

    def calculate_changes_of_variable(self, time, dt, variable, f_flow):
        """ Calculates the changes of ONE variable in every box.

        Attributes:
        - time (pint quantity): time at which the mass flows should be
                                calculated
        - dt (pint quantity): timestep used
        - f_flow (numpy 1D array): Reduction coefficient of the mass flows due to 
                                   empty boxes.

        Returns:
        - var_changes (numpy array): Array with the magnitude of variable 
                                      changes for every box
        - var_units (numpy array): Units of var_changes
        """
        # reduction coefficent of the "variable-sinks" of each box for the treated variable
        f = np.ones(self.N_boxes) # scaling factor for sinks of each box
        one_vec = np.ones(self.N_boxes)
        var, var_units = self.get_box_variable_mass_vector(variable)
        var_conc, var_conc_units = self.get_box_variable_concentration_vector(variable)

        var_conc_matrix = np.array([var_conc, ] * self.N_boxes).T

        var_ini = copy.deepcopy(var)

        # mass flows relevant for variable-advection
        tt_flows = [flow for flow in self.flows if flow.tracer_transport==True]  # Tracer-Transport Flows
        tt_flows_with_variables = [flow for flow in tt_flows if len(flow.variables) > 0] 
        mass_flow_A = self.get_flow_matrix(time, tt_flows)
        mass_flow_A = (mass_flow_A.transpose()*f_flow).transpose()
        mass_flow_s = self.get_flow_sink_vector(time, tt_flows_with_variables)*f_flow
        mass_flow_q = self.get_flow_source_vector(time, tt_flows)*f_flow

        # all of these vectors have units of kg/s
        flow_s = mass_flow_s * var_conc
        flow_q = mass_flow_q
        p_s, p_s_units = self.get_process_sink_vector_of_variable(time, variable)
        p_q, p_q_units = self.get_process_source_vector_of_variable(time, variable)
        r_s, r_s_units = self.get_reaction_sink_vector_of_variable(time, variable)
        r_q, r_q_units = self.get_reaction_source_vector_of_variable(time, variable)
        f_s, f_s_units = self.get_flux_sink_vector_of_variable(time, variable)
        f_q, f_q_units = self.get_flux_source_vector_of_variable(time, variable)

        dimless_dt = ((f_q_units/var_units)*dt).to_base_units().magnitude
         
        sources = ((p_q + r_q + f_q + np.dot(mass_flow_A.T, var_conc))*dimless_dt)
        sinks = ((p_s + r_s + f_s + np.dot(mass_flow_A * var_conc_matrix, one_vec))*dimless_dt)
        var_changes = sources - sinks

        dprint('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
        dprint('var: {}'.format(variable.name))
        dprint('mass_flow_A: {}'.format(mass_flow_A))
        dprint('var_conc: ', var_conc)
        dprint('tt_flows: ', tt_flows)
        dprint('tt_flows_with_variables: ', tt_flows_with_variables)
        dprint('var_conc_matrix: ', var_conc_matrix)
        dprint('p_s: {}'.format(p_s))
        dprint('p_q: {}'.format(p_q))
        dprint('r_s: {}'.format(r_s))
        dprint('r_q: {}'.format(r_q))
        dprint('f_s: {}'.format(f_s))
        dprint('f_q: {}'.format(f_q))
        dprint('np.dot(B.T, var_conc) : {}'.format(np.dot(B.T, var_conc)))
        dprint('np.dot(mass_flow_A * var_conc_matrix, one_vec): {}'.format(np.dot(mass_flow_A * var_conc_matrix, one_vec)))
        dprint('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

        dprint('------------------------------------')
        dprint('var_changes: {}'.format(var_changes))
        dprint('------------------------------------')


        dprint('------------------------------------')
        dprint('source due to Processes per timestep (np.dot(one_vec, p_q)): {}'.format(np.dot(one_vec, p_q)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Processes per timestep (- np.dot(one_vec, p_s)): {}'.format(- np.dot(one_vec, p_s)))
        dprint('------------------------------------')


        dprint('------------------------------------')
        dprint('source due to Reactions per timestep (np.dot(one_vec, r_q)): {}'.format(np.dot(one_vec, r_q)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Reactions per timestep (- np.dot(one_vec, r_s)): {}'.format(- np.dot(one_vec, r_s)))
        dprint('------------------------------------')


        dprint('------------------------------------')
        dprint('source due to Fluxes per timestep (np.dot(one_vec, f_q)): {}'.format(np.dot(one_vec, f_q)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Fluxes per timestep (- np.dot(one_vec, f_s)): {}'.format(- np.dot(one_vec, f_s)))
        dprint('------------------------------------')


        dprint('------------------------------------')
        dprint('source due to Flows per timestep (np.dot(mass_flow_A.T, var_conc)): {}'.format(np.dot(mass_flow_A.T, var_conc)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Flows per timestep (- np.dot(mass_flow_A * var_conc_matrix, one_vec)): {}'.format(- np.dot(mass_flow_A * var_conc_matrix, one_vec)))
        dprint('------------------------------------')
        
        var = var_ini + var_changes
       
        while_counter = 0
        
        #print('VAR: ', var)
        #print('var_conc', var_conc)
        while np.any(var<0):
            while_counter += 1
            dprint('in while.... {}'.format(while_counter))
            dprint('f before corrections', f)

            argmin = np.argmin(var)
            dprint('argmin: {}'.format(argmin))
            
            available_var = var_ini[argmin]

            dprint('available_var: ', available_var)

            sources = ((p_q + r_q + f_q + np.dot(mass_flow_A.T, var_conc))*dimless_dt)
            sinks = ((p_s + r_s + f_s + np.dot(mass_flow_A * var_conc_matrix, one_vec))*dimless_dt)
            net_source = sources[argmin]
            net_sink = sinks[argmin]
            
            if (net_source+available_var) > 0: 
                dprint('net_source+available_var > 0!')
                f[argmin] = min(
                        float(net_source+available_var)/float(net_sink),
                        f[argmin] * 0.95
                )
            else:
                dprint('net_source+available_var <= 0! -> f[argmin] = 0')
                f[argmin] = 0


            dprint('f after corrections: ', f)
            
            # Reduce Sinks and Sources accordingly to calculated f values
            mass_flow_A = (mass_flow_A.transpose()*f).transpose()
            flow_s = mass_flow_s * f
            mass_flow_q
            p_s = p_s * f
            r_s = r_s * f
            f_s = f_s * f

            sources = ((p_q + r_q + f_q + np.dot(mass_flow_A.T, var_conc))*dimless_dt)
            sinks = ((p_s + r_s + f_s + np.dot(mass_flow_A * var_conc_matrix, one_vec))*dimless_dt)

            dprint('np.dot(mass_flow_A.T, var_conc) * f)', np.dot(mass_flow_A.T, var_conc) * f)
            dprint('np.dot(mass_flow_A * var_conc_matrix, one_vec)*f', np.dot(mass_flow_A * var_conc_matrix, one_vec)*f)

            # Calculated corrected var_changes
            var_changes = (sources - sinks)

            var = var_ini + var_changes
            
            dprint('var: ' + variable.name)
            dprint('variable mass per box at the begining of this timestep: {}'.format(var_ini))
            dprint('variable mass per box at the end of this timestep: {}'.format(var_ini + var_changes))
            dprint('var_changes: ', var_changes)
            dprint('sources', sources)
            dprint('sinks', sinks)


            sdlkfahdl = dinput('AT THE END OF WHILE...')

        dprint('>>>>>>>>>>>')
        dprint('FINAL {} change: {}'.format(variable.name, var_changes))
        dprint('FINAL {}: {}'.format(variable.name, var))
        dprint('>>>>>>>>>>>')

        sdlfjsdll = dinput('AT THE END OF CALCULATE VARIABLE MASS CHANGE FUNCTION')

        return var_changes, var_units
             
    def solve_flows(self, total_integration_time, dt, debug=True):
        global DEBUG 
        DEBUG = debug
        pdb.set_trace()


        print('Start solving the flows of the box model...')
        print('- integration time: {}'.format(total_integration_time))
        print('- time step: {}'.format(dt))
        
        # Get number of time steps - round up if there is a remainder
        N_time_steps = int(total_integration_time/dt)
        time = total_integration_time * 0

        # Start time of function
        start_time = time_module.time() 
        
        sol = Solution(total_integration_time, dt, self.box_list)
        
        progress = 0
        for i in range(N_time_steps):
            progress_old = progress
            progress = round(float(i)/float(N_time_steps), 1) * 100.0
            if progress != progress_old:
                dprint("{}%".format(progress))
            #dprint('Time: {}'.format(time.to(dt.units).round()))
            sol.time.append(time)
            
            m_changes, m_units, A = self.calculate_mass_flows(time, dt)
            
            for box_name, box in self.boxes.items():
                # Write changes to box objects
                box.fluid.mass += m_changes[box.ID] * m_units
                
                # Save volumes to Solution instance
                sol.ts[box_name]['mass'].append(box.fluid.mass)
            
            time += dt

            a = dinput('Wait for it...')
        
        # End Time of Function
        end_time = time_module.time()
        print('Function "solve_flows(...)" used {:3.3f}s'.format(end_time - start_time))

        return sol

    def solve(self, total_integration_time, dt, debug=True):
        global DEBUG 
        DEBUG = debug

        print('Start solving the box model...')
        print('- integration time: {}'.format(total_integration_time))
        print('- time step: {}'.format(dt))
        
        dprint('----> mm: ', self.mm)
        dprint('----> sum(mm): ', self.mm.sum())
        

        # Start time of function
        start_time = time_module.time() 

        # Get number of time steps - round up if there is a remainder
        N_time_steps = int(total_integration_time/dt)
        time = total_integration_time * 0
        
        sol = Solution(total_integration_time, dt, self.box_list)
        
        progress = 0
        for i in range(N_time_steps):
            progress_old = progress
            progress = round(float(i)/float(N_time_steps), 1) * 100.0
            if progress != progress_old:
                print("{}%".format(progress))
            dprint('Time: {}'.format(time))
            sol.time.append(time)
            if time.magnitude > 140:
                #DEBUG = True
                #pdb.set_trace()
                pass

            ##################################################
            # Calculate Mass fluxes
            ##################################################
            
            m_change, m_units, A = self.calculate_mass_flows(time, dt)
            

            ##################################################
            # Calculate Variable changes due to PROCESSES, 
            # REACTIONS, FUXES and finally due to the FLOWS
            ##################################################

            var_changes = {}
            var_units = {} 

            for var_name, var in self.variables.items():
                dprint('\n\n\n\n\nSOLVE for variable {}'.format(var_name))
                var_changes[var_name], var_units[var_name] = self.calculate_changes_of_variable(time, dt, var, A)

            dprint('var_changes: {}'.format(var_changes))
            dprint('var_units: {}'.format(var_units))
            

            ##################################################
            # Apply changes to Boxes and save values to
            # Solution instance
            ##################################################
         

            dprint('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            for box_name, box in self.boxes.items():
                for var_name, var in box.variables.items():
                    dprint('1111self.boxes[{}].variables[{}].mass: '.format(box_name, var_name), self.boxes[box_name].variables[var_name].mass)
            dprint('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')



            for box_name, box in self.boxes.items():
                # Write changes to box objects
                box.fluid.mass += m_change[box.ID] * m_units

                # Save mass to Solution instance
                sol.ts[box_name]['mass'].append(box.fluid.mass)

                for var_name, var in box.variables.items():

                    dprint('__________________________________')
                    dprint('$$$$$$1111self.boxes[{}].variables[{}].mass: '.format(box_name, var_name), self.boxes[box_name].variables[var_name].mass)
                    dprint('box_name: ', box_name)
                    dprint(var_name + ':')
                    dprint(var_changes[var_name][box.ID])
                    dprint(var_units[var_name])

                    dprint('@@@@ : ', var_changes[var_name][box.ID])
                    dprint('@@@@ : ', var_units[var_name])
                    dprint('@@@@ : ', self.boxes[box_name].variables[var_name].mass)


                    new_var_mass = var_changes[var_name][box.ID] * var_units[var_name] + self.boxes[box_name].variables[var_name].mass
                    

                    dprint('@@@@ : ', new_var_mass)
                    dprint(box_name, var_name)
                    self.boxes[box_name].variables[var_name].mass = new_var_mass


                    sol.ts[box_name][var_name].append(self.boxes[box_name].variables[var_name].mass)


                    self.mm.loc[box_name, var_name] = new_var_mass
                    if box.fluid.mass.magnitude > 0:
                        self.cm.loc[box_name, var_name] = new_var_mass / box.fluid.mass
                    else: 
                        self.cm.loc[box_name, var_name] = 0

                    dprint('$$$$$$2222self.boxes[{}].variables[{}].mass: '.format(box_name, var_name), self.boxes[box_name].variables[var_name].mass)
            time += dt

            dprint('----> mm: ', self.mm)
            dprint('----> sum(mm): ', self.mm.sum())
            a = dinput('(End of timestep loop) Wait for it...')

        # End Time of Function
        end_time = time_module.time()
        print('Function "solve(...)" used {:3.3f}s'.format(end_time - start_time))
        
        return sol
