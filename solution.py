# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 2017 at 12:07UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
import time as time_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attrdict import AttrDict

from . import dimensionality_validation as bs_dim_val
from . import system as bs_system


class Solution:
    """Storage of a simulation solution created by a Solver instance.
    
    Stores a simulation and additional meta-information of the simulation.
    Offers various plotting functions to visualize the simulation solution.
    
    Args:
        total_integration_time (pint.Quantity [T]): Total length of the simulation.
        dt (pint.Quantity [T]): Integration timestep. 
        initial_system (BoxModelSystem): Initial state of the system that is 
            simulated.
    
    Attributes:
        total_integration_time (pint.Quantity): Total length of the simulation.
        dt (pint.Quantity): Integration timestep. 
        time (list of pint.Quantity): List of all times at which the system
            was solved (at which a result is available).
        system (BoxModelSystem): System which is simulated. 
        time_units (pint.Units): Units of Quantities within the time attribute.
        time_magnitude (float): Magnitudes of Quantities within the time 
            attribute.
        timeseries (AttrDict of AttrDict): For every box, there 
            exists one AttrDict which contains time series of all its 
            quantities (Fluid mass, Variable mass...) and the box instance.
            
    """

    def __init__(self, total_integration_time, dt, initial_system):
        bs_dim_val.dimensionality_check_time_err(total_integration_time)
        bs_dim_val.dimensionality_check_time_err(dt)

        self.total_integration_time = total_integration_time
        self.dt = dt
        
        if not isinstance(system, bs_system.BoxModelSystem):
            raise ValueError('Parameter system must be instance of '
                'BoxModelSystem!')
        self.system = copy.deepcopy(system)

        self.time = []
        self.time_units = None
        self.time_magnitude = None
        
        self.timeseries = AttrDict()
        for box_name, box in self.system.boxes.items():
            tmp_dict = {'box': box, 'mass': [], 'volume': []}
            for variable_name, variable in self.system.variables.items():
                tmp_dict[variable_name] = []
            self.timeseries[box_name] = AttrDict(tmp_dict)

    def plot_fluid_masses(self, boxes=None, ):
        fig, ax = plt.subplots()

        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        for box_name, ts in self.ts.items():
            masses = self.ts[box_name]['mass']
            mass_magnitude = [mass.magnitude for mass in masses]
            ax.plot(self.time_magnitude, mass_magnitude,
                    label='Box {}'.format(ts.box.ID))

        ax.set_ylabel('kg')
        ax.set_xlabel(self.time_units)
        ax.set_title('Box masses')
        ax.legend()

    def plot_variables_masses(self, variables, figsize=[10, 10]):
        N_boxes = len(self.ts)
        Nx = int(np.ceil(N_boxes**0.5))  # int((N_boxes+1)**0.5)
        Ny = int(round(N_boxes**0.5))
        print(Nx, Ny)
        fig, axarr = plt.subplots(Ny, Nx, sharex=True, figsize=figsize)

        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        for i, (box_name, ts) in enumerate(self.ts.items()):
            print('i: {}'.format(i))
            x = i % Nx
            y = int(i / Nx)
            print('x={}  y={}'.format(x, y))

            handles = []

            for j, var_name in enumerate(variable_names):
                masses = self.ts[box_name][var_name]
                mass_magnitude = [mass.magnitude for mass in masses]
                axarr[y, x].set_title('Box {}'.format(box_name))
                line, = axarr[y, x].plot(
                    self.time_magnitude, mass_magnitude, label=var_name)
                handles.append(line)

        fig.text(
            0.5,
            0.04,
            'Time [{}]'.format(
                self.time_units),
            ha='center',
            va='center')
        fig.text(
            0.06,
            0.5,
            'Mass [kg]',
            ha='center',
            va='center',
            rotation='vertical')
        fig.legend(handles, variable_names)
