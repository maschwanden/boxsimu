# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:37:12 2016

@author: aschi
"""

import copy
import time as time_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attrdict import AttrDict

from transport import Flow
from process import Process, Reaction
from condition import Condition


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
        system_variables = {}
        
        box_id = 0
        for box_name, box in self.boxes.items():
            box.ID = box_id
            box_id += 1
            box.system = self
            box.set_global_condition(global_condition)
        
            for var_name, var in box.variables.items():
                if not var_name in system_variables.keys():
                    var_copy = copy.deepcopy(var)
                    var_copy.mass *= 0
                    system_variables[var_name] = var_copy
                    
        # Add the variables that are not defined in a box
        # And: Set variable ID's for every variable in every box
        for box_name, box in self.boxes.items():
            var_id = 0
            for var_name, var in system_variables.items():
                if not var_name in box.variables.keys():
                    box.variables[var_name] = var
                box.variables[var_name].ID = var_id
                var_id += 1    

        system_variable_names = [var_name for var_name, var in 
                                 system_variables.items()]
        
        # initialize mass and concentration matrix: 
        self.mm = pd.DataFrame(columns=system_variable_names, index=box_names)
        self.cm = pd.DataFrame(columns=system_variable_names, index=box_names)
        for box_name, box in self.boxes.items():
            for var_name, var in box.variables.items():
                self.mm.loc[box_name, var_name] = var.mass
                self.cm.loc[box_name, var_name] = var.mass / box.volume
                
        self.processes = [process for box_name, box in self.boxes.items() for process in box.processes]

    @property            
    def box_list(self):
        return [box for box_name, box in self.boxes.items()]
    
    @property
    def N_boxes(self):
        return len(self.boxes)
    
    @property
    def N_variables(self):
        return len(self.variables)
    
    def get_flow_matrix(self, time):
        N_boxes = self.N_boxes
        A = np.zeros([N_boxes, N_boxes])
        
        for flow in self.flows:
            src_box = flow.source_box
            trg_box = flow.target_box
            if src_box == None or trg_box == None:
                continue
            mass_flow_rate = flow.mass_flow_rate(time, src_box.context)
            A[src_box.ID, trg_box.ID] += mass_flow_rate.magnitude
        return A
            
    
    def get_flow_source_vector(self, time):
        s = np.zeros(self.N_boxes)
        
        sources = Flow.get_all_from(None, self.flows)
        for flow in sources:
            src_box = flow.source_box
            trg_box = flow.target_box
            if src_box == None and trg_box == None:
                continue
            mass_flow_rate = flow.mass_flow_rate(time, trg_box.context)
            s[trg_box.ID] += mass_flow_rate.magnitude
        return s
    
    def get_flow_sink_vector(self, time):
        q = np.zeros(self.N_boxes)
        
        sinks = Flow.get_all_to(None, self.flows)
        for flow in sinks:
            src_box = flow.source_box
            trg_box = flow.target_box
            if src_box == None and trg_box == None:
                continue
            mass_flow_rate = flow.mass_flow_rate(time, src_box.context)
            q[src_box.ID] += mass_flow_rate.magnitude
        return q
    
    def get_box_fluid_mass_vector(self, time):
        m = np.zeros(self.N_boxes)
        for box_name, box in self.boxes.items():
            m[box.ID] = box.fluid.mass.magnitude
        return m, box.fluid.mass.units
        
           
    def solve(self, total_integration_time, dt):
        print('Start solving the box model...')
        print('- integration time: {}'.format(total_integration_time))
        print('- time step: {}'.format(dt))
        
        # Start time of function
        start_time = time_module.time() 

        # Get number of time steps - round up if there is a remainder
        N_time_steps = int(total_integration_time/dt)
        time = total_integration_time * 0
        
        sol = Solution(total_integration_time, dt, self.box_list)
        
        progress = 0
        for i in range(N_time_steps):
#            print('No. timestep: {}'.format(i))
            progress_old = progress
            progress = round(float(i)/float(N_time_steps), 1) * 100.0
            if progress != progress_old:
                print("{}%".format(progress))
            #print('Time: {}'.format(time.to(dt.units).round()))
            sol.time.append(time)
                              
            ##################################################
            # Calculate Mass fluxes
            ##################################################
             
            m, m_units = self.get_box_fluid_mass_vector(time)
            m_change, m_units = self.calculate_mass_flows(time, dt)
            

            ##################################################
            # Calculate Variable changes due to PROCESSES and 
            # FUXES and finally due to the FLOWS
            ##################################################
            
            var_changes, var_units = self.calculate_variable_changes(time)
                        
            
            ##################################################
            # Apply changes to Boxes 
            ##################################################
         
            for box_name, box in self.boxes.items():
                # Write changes to box objects
                box.fluid.mass += m_change[box.ID] * m_units
                
                # Save volumes to Solution instance
                sol.box_sol[box_name]['mass'].append(m[box.ID])
            
            time += dt
            #a = input('Wait for it...')

        # End Time of Function
        end_time = time_module.time()
        print('Function "solve(...)" used {:3.3f}s'.format(end_time - start_time))
        
        return sol
    
    
    def solve_flows(self, total_integration_time, dt):
        print('Start solving the flows of the box model...')
        print('- integration time: {}'.format(total_integration_time))
        print('- time step: {}'.format(dt))
        
        # Get number of time steps - round up if there is a remainder
        N_time_steps = int(total_integration_time/dt)
        time = total_integration_time * 0
        
        sol = Solution(total_integration_time, dt, self.box_list)
        
        progress = 0
        for i in range(N_time_steps):
#            print('No. timestep: {}'.format(i))
            progress_old = progress
            progress = round(float(i)/float(N_time_steps), 1) * 100.0
            if progress != progress_old:
                print("{}%".format(progress))
            #print('Time: {}'.format(time.to(dt.units).round()))
            sol.time.append(time)
            
            m, m_units = self.get_box_fluid_mass_vector(time)
            m_change, m_units = self.calculate_mass_flows(time, dt)
                             
            m = (m + m_change) * m_units
            
            for box_name, box in self.boxes.items():
                # Write changes to box objects
                box.fluid.mass = m[box.ID]
                
                # Save volumes to Solution instance
                sol.box_sol[box_name]['mass'].append(m[box.ID])
            
            time += dt
            #a = input('Wait for it...')
        
        return sol
    
    def calculate_mass_flows(self, time, dt):
        f = np.ones(self.N_boxes) # scaling factor for sinks of each box
        m, m_units = self.get_box_fluid_mass_vector(time)
        m_ini = copy.deepcopy(m)
        # A, q, and s all have units of mass flux [kg/s]
        A = self.get_flow_matrix(time)
        q = self.get_flow_source_vector(time)
        s = self.get_flow_sink_vector(time)
        
#        print('m  : {}'.format(m))
#        print('A  : {}'.format(A))
#        print('s  : {}'.format(s))
#        print('q  : {}'.format(q))
        
        dimless_dt = ((self.flows[0].units/m_units)*dt).to_base_units().magnitude
        
#        print('f = {}'.format(f))
#        print('f*s = {}'.format(f*s))
#        print('q - f*s = {}'.format(q - f*s))
#        
#        print('mass of the boxes before this timestep:\n{}'.format(m))
        
        m_change = (q - s - np.dot(A,np.ones(self.N_boxes)) + 
                     np.dot((A).T, np.ones(self.N_boxes)))*dimless_dt
        m = m_ini + m_change
        
        while np.any(m<0):
            flow_source_arr = A[:,np.argmin(m)]
            flow_sink_arr = A[np.argmin(m),:]
            source = q[np.argmin(m)]
            sink = s[np.argmin(m)]
            net_source =  sum(flow_source_arr) + source
#            print('net_source: {}'.format(net_source))
            net_sink = sum(flow_sink_arr) + sink
#            print('net_sink: {}'.format(net_sink))
            
            f[np.argmin(m)] = min(
                    float(net_source)/float(net_sink),
                    f[np.argmin(m)] * 0.95
            )
#            print('float(net_source)/float(net_sink): {}'.format(float(net_source)/float(net_sink)))
            
            # Apply reduction of sinks of the box
            A = (A.transpose()*f).transpose()
            s = f*s
            m_change = (q - s - np.dot(A,np.ones(self.N_boxes)) + 
                     np.dot((A).T, np.ones(self.N_boxes)))*dimless_dt
            m = m_ini + m_change
#            
#        print('mass of the boxes after this timestep:\n{}'.format(m))
#            
#        print('net change of mass of the boxes in this:\n{}'.format((q - f*s - f*np.dot(A,np.ones(self.N_boxes)) + 
#                     np.dot((f*A).T, np.ones(self.N_boxes)))*dimless_dt))
#        print('------> sources: q :\n{}'.format(q))
#        print('------> sinks: f*s :\n{}'.format(f*s))
#        print('------> sinks through internal flows: f*np.dot(A,np.ones(self.N_boxes)) :\n{}'.format(f*np.dot(A,np.ones(self.N_boxes))))
#        print('------> source through internal flows: np.dot((f*A).T, np.ones(self.N_boxes)) :\n{}'.format(np.dot((f*A).T, np.ones(self.N_boxes))))
#        print('f : {}'.format(f))
#        print('--------------------------------------')
#        
        return m_change, m_units

    def calculate_variable_changes(self, time):
        #print('calculate_variable_changes')
        # Process vector: contains the net change of the mass of variables
        p = np.zeros(self.N_variables)
        
        var_changes = np.zeros(self.N_variables)
        var_units = []

        for box_name, box in self.boxes.items():
            #print(box_name)
            var_process_change_rate = 0

            for var_name, var in box.variables.items():
                #print(var_name)
                var_reaction_processes = [p for r in box.reactions for p in r.get_all_processes_of_variable(var)]
                var_processes = Process.get_all_of_variable(var, box.processes) + var_reaction_processes
                for process in var_processes:
                    var_process_change_rate += process(time, process.box)
                    #print('process of variable {}: {} with a rate of {}'.format(var_name, process.name, process(time, process.box))) 
                
                
    
        return p, var_units 
            
            

class Solution:
    def __init__(self, total_integration_time, dt, boxes):
        self.total_integration_time = total_integration_time
        self.dt = dt
        self.time = []
        self.time_units = None
        self.time_magnitude = None
        
        self.box_sol = AttrDict({box.name: AttrDict({}) for box in boxes})
        
        for box in boxes:
            self.box_sol[box.name] = AttrDict(
                    {'box': box, 'mass': [], 'volume': []})
            for var_name, var in box.variables.items():
                self.box_sol[box.name][var_name] = []
                
    def plot_box_masses(self):
        fig, ax = plt.subplots()
        
        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]
        
        for box_name, box_sol in self.box_sol.items():
            masses = self.box_sol[box_name]['mass']
            mass_magnitude = [mass.magnitude for mass in masses]
            ax.plot(self.time_magnitude, mass_magnitude, 
                    label='Box {}'.format(box_sol.box.ID))
            
        ax.set_ylabel('kg')
        ax.set_xlabel(self.time_units)
        ax.set_title('Box masses')
        ax.legend()


            
