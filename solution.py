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
import matplotlib.ticker as mtick
from attrdict import AttrDict

from . import box as bs_box
from . import dimensionality_validation as bs_dim_val
from . import descriptors as bs_descriptors
from . import entities as bs_entities
from . import errors as bs_errors 
from . import process as bs_process
from . import system as bs_system
from . import transport as bs_transport
from . import utils as bs_utils
from . import ur


class TimeStep:
    """Information about one timestep of a simulation.

    A Timestep instance contains information about the start and end time
    of a timestep in a simulation, quantities at both times (start and end
    time) and the rates of processes, reactions, flows, and fluxes.

    Args:
        start_time (pint.Quantity [T]): Start time of the timestep.
        end_time (pint.Quantity [T]): End time of the timestep.

    """
    start_time = bs_descriptors.PintQuantityDescriptor(
            'start_time', ur.second)
    end_time = bs_descriptors.PintQuantityDescriptor(
            'end_time', ur.second)

    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def add_box_mass(self, box, mass):
        """Add masses of Boxes, Fluids, and Variables at t=end_time$

        Args:
            box (Box): Instance of class Box for which the mass is given.
            mass (pint.Quantity [M]): Total mass of the box at the end 
                of the timestep.

        """
        if not isinstance(box, bs_box.Box):
            raise 
            
        for key, value in masses.items():
            pass

    def add_box_volume(self, box, *volumes):
        """Add volumes of Boxes at t=end_time$

        Args:
            volumes (dict): Key value pairs of Boxes as keys and pint 
            Quantities with the dimensionality of [M] as values.

        """
        pass

    def add_variable_mass(self, box, variable, mass):
        pass

    def add_box_rates(self, box, *rates):
        """Add rates of Processes and Reactions in the timespan dt.

        Args:
            rates (dict): Key value pairs of Processes or Reactions as keys 
            and pint Quantities with the dimensionality of [M] as values.

        """
        pass


class Solution:
    """Storage of a simulation's solution.
    
    An instance of Solution stores the outcome and additional 
    meta-information of the simulation.
    Additionaly, the Solution class offers various plotting functions to 
    visualize the result of the simulation
    
    Args:
        system (BoxModelSystem): System that is simulated.
        total_integration_time (pint.Quantity [T]): Total length of the simulation.
        dt (pint.Quantity [T]): Integration timestep. 
    
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

    total_integration_time = bs_descriptors.PintQuantityDescriptor(
            'total_integration_time', ur.second)
    dt = bs_descriptors.PintQuantityDescriptor('dt', ur.second)

    def __init__(self, system, N_timesteps, dt):
        self.system = system
        self.N_timesteps = N_timesteps
        self.dt = 1 * dt
        self.dt = self.dt.to_base_units()
        self.total_integration_time = N_timesteps * dt

        self.setup_solution_dataframe()
        
        self.default_figsize = [7,4]
        self.yaxis_log = False

    def setup_solution_dataframe(self):
        time_array = np.linspace(0, self.total_integration_time.magnitude,
                num=self.N_timesteps)
        time_units = self.dt.units
        
        # Setup Dataframe for timeseries of quantities (masses, volumes..)
        quantities = ['mass', 'volume'] + self.system.variable_names 
        col_tuples = [(box, quant) for box in self.system.box_names
                                   for quant in quantities]
        index = pd.MultiIndex.from_tuples(col_tuples, 
                names=['Box', 'Quantity'])
        self.df_ts = pd.DataFrame(index=index).T
        self.df_ts.units = ur.kg

        # Setup Dataframe for timeseries of rates (proecesses, flows..)
        col_tuples = []
        for box_name, box in self.system.boxes.items():
            for variable_name, variable in self.system.variables.items():
                col_tuples.append((box_name, variable_name, 'flows'))
                col_tuples.append((box_name, variable_name, 'fluxes'))
                col_tuples.append((box_name, variable_name, 'processes'))
                col_tuples.append((box_name, variable_name, 'reactions'))
                flows = self.system.flows
                fluxes = self.system.fluxes
                for flow in bs_transport.Flow.get_all_from(box, flows):
                    col_tuples.append((box_name, variable_name, 
                        'flow_{}'.format(flow.name)))
                for flow in bs_transport.Flow.get_all_to(box, flows):
                    col_tuples.append((box_name, variable_name, 
                        'flow_{}'.format(flow.name)))
                for flux in bs_transport.Flux.get_all_from(box, fluxes):
                    col_tuples.append((box_name, variable_name, 
                        'flux_{}'.format(flux.name)))
                for flux in bs_transport.Flux.get_all_to(box, fluxes):
                    col_tuples.append((box_name, variable_name, 
                        'flux_{}'.format(flux.name)))
                for process in box.processes:
                    col_tuples.append((box_name, variable_name, 
                        'process_{}'.format(process.name)))
                for reaction in box.reactions:
                    col_tuples.append((box_name, variable_name, 
                        'reaction_{}'.format(reaction.name)))

        index = pd.MultiIndex.from_tuples(col_tuples, 
                names=['Box', 'Variable', 'Mechanism'])
        self.df_ts_rates = pd.DataFrame(index=index).T
        self.df_ts_rates.units = ur.kg/ur.second

    def add_timestep(self, timestep):
        """Add a timestep to the solution.

        Args:
            timestep (Timestep): Instance of class Timestep.

        """
        pass

    # VISUALIZATION

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



