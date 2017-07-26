# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 2017 at 12:07UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
import time as time_module
import numpy as np
import matplotlib.pyplot as plt
from attrdict import AttrDict

from . import dimensionality_validation as bs_dim_val
from . import system as bs_system
from . import utils as bs_utils


class Solution:
    """Storage of a simulation solution created by a Solver instance.
    
    Stores a simulation and additional meta-information of the simulation.
    Offers various plotting functions to visualize the simulation solution.
    
    Args:
        total_integration_time (pint.Quantity [T]): Total length of the simulation.
        dt (pint.Quantity [T]): Integration timestep. 
        system (BoxModelSystem): System that is simulated.
    
    Attributes:
        total_integration_time (pint.Quantity): Total length of the simulation.
        dt (pint.Quantity): Integration timestep. 
        time (list of pint.Quantity): List of all times at which the system
            was solved (at which a result is available).
        system (BoxModelSystem): System which is simulated. 
        time_units (pint.Units): Units of Quantities within the time attribute.
        time_magnitude (float): Magnitudes of Quantities within the time 
            attribute.
        ts (AttrDict of AttrDict): For every box, there 
            exists one AttrDict which contains time series of all its 
            quantities (Fluid mass, Variable mass...) and the box instance.
            
    """

    def __init__(self, total_integration_time, dt, system):
        bs_dim_val.raise_if_not_time(total_integration_time)
        bs_dim_val.raise_if_not_time(dt)

        self.total_integration_time = total_integration_time
        self.dt = dt
        
        self.system = copy.deepcopy(system)

        self.time = []
        self.time_units = None
        self.time_magnitude = None
        
        self.ts = AttrDict()
        for box_name, box in self.system.boxes.items():
            tmp_dict = {'box': box, 'mass': [], 'volume': []}
            for variable_name, variable in self.system.variables.items():
                tmp_dict[variable_name] = []
            self.ts[box_name] = AttrDict(tmp_dict)

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
                    label='Box {}'.format(ts.box.id))

        ax.set_ylabel('kg')
        ax.set_xlabel(self.time_units)
        ax.set_title('Box masses')
        ax.legend()
        return fig, ax

    def plot_variable_mass_of_all_boxes(self, variable, figsize=[10, 10]):
        fig, ax = plt.subplots()

        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        for box_name, ts in self.ts.items():
            masses = self.ts[box_name][variable.name]
            mass_magnitude = [mass.magnitude for mass in masses]
            ax.plot(self.time_magnitude, mass_magnitude,
                    label='Box {}'.format(ts.box.name))

        ax.set_ylabel('kg')
        ax.set_xlabel(self.time_units)
        ax.set_title(variable.name)
        ax.legend()
        return fig, ax

    def plot_all_variable_mass_of_box(self, box, figsize=[10, 10]):
        fig, ax = plt.subplots()

        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        for variable in self.system.variable_list:
            var_mass += self.ts[box.name][variable.name]
            mass_magnitude = [mass.magnitude for mass in var_mass]
            ax.plot(self.time_magnitude, mass_magnitude,
                    label='Variable {}'.format(variable.name))

        ax.set_ylabel('kg')
        ax.set_xlabel(self.time_units)
        ax.set_title(box.name)
        ax.legend()
        return fig, ax

    def plot_total_variable_masses(self, figsize=[10, 10]):
        fig, ax = plt.subplots()

        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        for variable in self.system.variable_list:
            var_masses = np.zeros(len(self.time_magnitude))
            i = 0
            for box_name, ts in self.ts.items():
                vm = bs_utils.get_array_quantity_from_array_of_quantities(
                        self.ts[box_name][variable.name])
                var_masses += vm
                i += 1
            mass_magnitude = [mass.magnitude for mass in var_masses]
            ax.plot(self.time_magnitude, mass_magnitude,
                    label='Variable {}'.format(variable.name))

        ax.set_ylabel('kg')
        ax.set_xlabel(self.time_units)
        ax.set_title('Total Variable Concentrations')
        ax.legend()
        return fig, ax


