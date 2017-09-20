# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 2017 at 12:07UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import copy
import time as time_module
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from attrdict import AttrDict

from . import dimensionality_validation as bs_dim_val
from . import errors as bs_errors # import WrongUnitsDimensionalityError
from . import system as bs_system
from . import utils as bs_utils
from . import ur


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

        self.default_figsize = [7,4]
        self.yaxis_log = False

    def plot_fluid_masses(self, boxes=None, figsize=None, yaxis_log=False):
        if not yaxis_log:
            yaxis_log = self.yaxis_log
        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]
        if not figsize:
            figsize = self.default_figsize

        fig, ax = self._get_subplots(
                title='Fluid Masses',
                xlabel=self.time_units,
                ylabel='kg',
                figsize=figsize,
                yaxis_log=yaxis_log)

        for box_name, ts in self.ts.items():
            masses = self.ts[box_name]['mass']
            mass_magnitude = [mass.magnitude for mass in masses]
            ax.plot(self.time_magnitude, mass_magnitude,
                    label='Box {}'.format(ts.box.id))

        ax.legend()
        return fig, ax

    def plot_variable_mass_of_all_boxes(self, variable, figsize=None, 
            yaxis_log=False):
        if not yaxis_log:
            yaxis_log = self.yaxis_log
        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        if not figsize:
            figsize = self.default_figsize

        fig, ax = self._get_subplots(
                title='Variable Mass of {}'.format(variable.name),
                xlabel=self.time_units,
                ylabel='kg',
                figsize=figsize,
                yaxis_log=yaxis_log)

        for box_name, ts in self.ts.items():
            masses = self.ts[box_name][variable.name]
            mass_magnitude = [mass.magnitude for mass in masses]
            ax.plot(self.time_magnitude, mass_magnitude,
                    label='Box {}'.format(ts.box.name))
        ax.legend()
        return fig, ax

    def plot_variable_concentration_in_boxes(self, variable, boxes=None,
            figsize=None, yaxis_log=False, volumetric=False, units=None):
        """Plot the timeseries of the variable concentration [kg/kg] in Boxes.

        Plot the time series of the variable concentration [kg/kg] in the 
        specified Boxes. If no Boxes are specified (default) all boxes of the
        system are used.

        """
        if not boxes:
            boxes = self.system.box_list
        box_names = [box.name for box in boxes]
        if not yaxis_log:
            yaxis_log = self.yaxis_log
        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]
        if not figsize:
            figsize = self.default_figsize
        if not units:
            if volumetric:
                y_label_text = 'kg/m^3'
                units = ur.kg/ur.meter**3
            else:
                y_label_text = 'kg/kg'
                units = ur.dimensionless
        else:
            if bs_dim_val.is_density(units):
                volumetric = True
            elif bs_dim_val.is_dimless(units):
                volumetric = False
            else:
                raise bs_errors.WrongUnitsDimensionalityError('Parameter units '
                        'has incorrect dimensionality!')
            y_label_text = units.format_babel()

        fig, ax = self._get_subplots(
                title='Variable Concentration of {}'.format(variable.name),
                xlabel=self.time_units,
                ylabel=y_label_text,
                figsize=figsize,
                yaxis_log=yaxis_log)

        for box_name, ts in self.ts.items():
            if not box_name in box_names:
                continue # This Box was not intended to be plotted
            var_mass_units = self.ts[box_name][variable.name][0].units
            var_mass_ts = np.array([m.magnitude 
                for m in self.ts[box_name][variable.name]]) * var_mass_units
            if volumetric:
                box_volume_units = self.ts[box_name]['volume'][0].units
                box_volume_ts = np.array([v.magnitude for v 
                    in self.ts[box_name]['volume']]) * box_volume_units
                concentrations = var_mass_ts / box_volume_ts
            else:
                box = self.system.boxes[box_name]
                concentrations = var_mass_ts / box.mass
            concentrations.ito(units)
            ax.plot(self.time_magnitude, concentrations,
                    label='Box {}'.format(ts.box.name))

        ax.legend()
        return fig, ax

    def plot_all_variable_mass_of_box(self, box, figsize=None, 
            yaxis_log=None):
        if not yaxis_log:
            yaxis_log = self.yaxis_log
        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        if not figsize:
            figsize = self.default_figsize

        if yaxis_log:
            yaxis_log = 'log'
        else:
            yaxis_log = None

        fig, ax = self._get_subplots(
                title='Total Variable Masses',
                xlabel=self.time_units,
                ylabel='kg',
                figsize=figsize,
                yaxis_scale=yaxis_log)

        var_mass = []
        for variable in self.system.variable_list:
            var_mass += self.ts[box.name][variable.name]
            mass_magnitude = [mass.magnitude for mass in var_mass]
            ax.plot(self.time_magnitude, mass_magnitude,
                    label='Variable {}'.format(variable.name))
        ax.legend()
        return fig, ax

    def plot_total_variable_masses(self, figsize=None, yaxis_log=None):
        if not yaxis_log:
            yaxis_log = self.yaxis_log
        if not self.time_units:
            self.time_units = self.time[0].units
        if not self.time_magnitude:
            self.time_magnitude = [t.magnitude for t in self.time]

        if not figsize:
            figsize = self.default_figsize

        if yaxis_log:
            yaxis_log = 'log'
        else:
            yaxis_log = None

        fig, ax = self._get_subplots(
                title='Total Variable Mass',
                xlabel=self.time_units,
                ylabel='kg',
                figsize=figsize,
                yaxis_scale=yaxis_log)

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

        ax.legend()
        return fig, ax

    def _get_subplots(self, title, xlabel, ylabel, figsize=[10,10], 
            yaxis_log=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        if yaxis_log: 
            ax.set_yscale('log')
        return fig, ax



