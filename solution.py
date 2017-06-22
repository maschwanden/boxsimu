# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:07:34 2017

@author: aschi
"""

import copy
import time as time_module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attrdict import AttrDict


class Solution:
    def __init__(self, total_integration_time, dt, boxes):
        self.total_integration_time = total_integration_time
        self.dt = dt
        self.time = []
        self.time_units = None
        self.time_magnitude = None
        
        self.ts = AttrDict({box.name: AttrDict({}) for box in boxes})
        
        for box in boxes:
            self.ts[box.name] = AttrDict(
                    {'box': box, 'mass': [], 'volume': []})
            for var_name, var in box.variables.items():
                self.ts[box.name][var_name] = []
                
    def plot_box_masses(self):
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

    def plot_variables_masses(self, variable_names, figsize=[10,10]):
        N_boxes = len(self.ts)
        Nx = int(np.ceil(N_boxes**0.5)) #int((N_boxes+1)**0.5)
        Ny  = int(round(N_boxes**0.5))
        print(Nx, Ny)
        fig, axarr = plt.subplots(Ny, Nx, sharex=True, figsize=figsize)

        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]
        
        print(axarr)

        for i, (box_name, ts) in enumerate(self.ts.items()):
            print('i: {}'.format(i))
            x = i%Nx
            y = int(i/Nx)
            print('x={}  y={}'.format(x, y))

            handles = []

            for j, var_name in enumerate(variable_names):
                masses = self.ts[box_name][var_name]
                mass_magnitude = [mass.magnitude for mass in masses]
                axarr[y, x].set_title('Box {}'.format(box_name))
                line, = axarr[y, x].plot(self.time_magnitude, mass_magnitude, label=var_name)
                handles.append(line) 
            
        fig.text(0.5, 0.04, 'Time [{}]'.format(self.time_units), ha='center', va='center')
        fig.text(0.06, 0.5, 'Mass [kg]', ha='center', va='center', rotation='vertical')
        fig.legend(handles, variable_names)
