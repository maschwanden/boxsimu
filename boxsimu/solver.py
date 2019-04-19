# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 10:37UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import os
import pdb
import copy
import time as time_module
import datetime
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from attrdict import AttrDict
import math

from . import solution as bs_solution
from . import utils as bs_utils
from . import ur


def save_simulation_state(system):
    filename = '{:%Y%m%d}_{}_TS{}.pickle'.format(
            datetime.date.today(), system.name, timestep)
    with open(filename, 'wb') as f:
        pickle.dump(system, f)


def load_simulation_state(system):
    pass


def solve(system, total_integration_time, dt, save_frequency=100, debug=False):
    """Simulate the time evolution of all variables within the system.

    Collect all information about the system, create differential 
    equations from this information and integrate them (numercially)
    into the future.

    Args:
        system (System): The system that is simulated.
        total_integration_time (pint.Quantity [T]): The time span which
            the system should be solved into the "future". The system will
            be simulated for the time period zero to approximately 
            total_integration_time (depending whether 
            total_integration_time is a multiple of dt; if not the real
            integration horizon will be be bigger than 
            [0, total_integration_time]).
        dt (pint.Quantity [T]): Size of the timestep for the simulation.
            The bigger the timestep the faster the simulation will be
            calculated, however, if the timestep is chosen too high
            there can arise numerical instabilites!
        save_frequency (int): Number of timesteps after which the solve 
            progress is saved to a pickle file. If the solver
            is interupted after the state was saved to a pickle file,
            the solve function will automatically progress at the latest
            saved state.
        debug (bool): Activates debugging mode (pdb.set_trace()).
            Defaults to False.

    """
    # Start time of function
    func_start_time = time_module.time()

    # Saves the time since the start of the simulate at which the last 
    # save (pickling) was conducted
    last_save_timedelta = 0  

    if debug:
        pdb.set_trace()
            
    # Get number of time steps - round up if there is a remainder
    N_timesteps = math.ceil(total_integration_time / dt)
    # Recalculate total integration time based on the number of timesteps
    total_integration_time = N_timesteps * dt
    print('DDATTEE')
    print('Start solving the BoxModelSystem...')
    print('- total integration time: {}'.format(total_integration_time))
    print('- dt (time step): {}'.format(dt))
    print('- number of time steps: {}'.format(N_timesteps))

    time = total_integration_time * 0
    sol = bs_solution.Solution(system, N_timesteps, dt)

    # Save initial state to solution
    for box in system.box_list:
        # sol.df.loc[0] = np.nan
        sol.df.loc[0, (box.name,'mass')] = box.fluid.mass.magnitude
        sol.df.loc[0, (box.name,'volume')] = \
                system.get_box_volume(box).magnitude
        for variable in system.variable_list:
            var_name = variable.name
            sol.df.loc[0, (box.name,var_name)] = \
                    box.variables[var_name].mass.magnitude

    timetesteps_since_last_save = 0
    progress = 0
    for timestep in range(N_timesteps):
        # Calculate progress in percentage of processed timesteps
        progress_old = progress
        progress = int(float(timestep) / float(N_timesteps)*10) * 10.0
        if progress != progress_old:
            print("{}%".format(progress))
        
        #print(timetesteps_since_last_save)
        # Check if simulation is running long enough to save the state
        if timetesteps_since_last_save >= save_frequency:
            timetesteps_since_last_save = 1
        else:
            timetesteps_since_last_save += 1

        time += dt

        ##################################################
        # Calculate Mass fluxes
        ##################################################

        dm, f_flow = _calculate_mass_flows(system, time, dt)

        ##################################################
        # Calculate Variable changes due to PROCESSES,
        # REACTIONS, FUXES and FLOWS
        ##################################################

        dvar = _calculate_changes_of_all_variables(
                system, time, dt, f_flow)

        ##################################################
        # Apply changes to Boxes and save values to
        # Solution instance
        ##################################################

        for box in system.box_list:
            # Write changes to box objects
            box.fluid.mass += dm[box.id]

            # Save mass to Solution instance
            sol.df.loc[timestep, (box.name, 'mass')] = \
                    box.fluid.mass.magnitude
            sol.df.loc[timestep, (box.name, 'volume')] = \
                    system.get_box_volume(box).magnitude

            for variable in system.variable_list:
                var_name = variable.name
                system.boxes[box.name].variables[var_name].mass += \
                        dvar[box.id, variable.id]
                sol.df.loc[timestep, (box.name,variable.name)] = \
                        box.variables[variable.name].mass.magnitude

    # End Time of Function
    func_end_time = time_module.time()
    print(
        'Function "solve(...)" used {:3.3f}s'.format(
            func_end_time - func_start_time))
    return sol


def _calculate_mass_flows(system, time, dt):
    """Calculate mass changes of every box.

    Args:
        time (pint.Quantity [T]): Current time (age) of the system.
        dt (pint.Quantity [T]): Timestep used.

    Returns:
        dm (numpy 1D array of pint.Quantities): Mass changes of every box.
        f_flow (numpy 1D array): Reduction coefficient of the mass 
            flows (due to becoming-empty boxes -> box mass cannot 
            decrase below 0kg).

    """
    # f_flow is the reduction coefficent of the "sink-flows" of each box
    # scaling factor for sinks of each box
    f_flow = np.ones(system.N_boxes)
    v1 = np.ones(system.N_boxes)

    m_ini = system.get_fluid_mass_1Darray()

    # get internal flow matrix and calculate the internal souce and sink 
    # vectors. Also get the external sink and source vector
    A = system.get_fluid_mass_internal_flow_2Darray(time)
    # internal 
    s_i = bs_utils.dot(A, v1)
    q_i = bs_utils.dot(A.T, v1)
    s_e = system.get_fluid_mass_flow_sink_1Darray(time)
    q_e = system.get_fluid_mass_flow_source_1Darray(time)

    # calculate first estimate of mass change vector
    dm = (q_e + q_i - s_e - s_i) * dt
    # calculate first estimate of mass after timestep
    m = m_ini + dm

    while np.any(m.magnitude < 0):
        argmin = np.argmin(m)
        # Calculate net sink and source and mass of the 'empty' box.
        net_source = (q_e[argmin] + q_i[argmin])*dt
        net_sink = (s_e[argmin] + s_i[argmin])*dt
        available_mass = m_ini[argmin]
        total_mass = (net_source + available_mass).to_base_units()

        if total_mass.magnitude > 0: 
            f_new = (total_mass / net_sink).to_base_units().magnitude 
            f_flow[argmin] = min(f_new, f_flow[argmin] * 0.98)
        else:
            f_flow[argmin] = 0

        # Apply reduction of sinks of the box
        A = (A.T * f_flow).T
        s_i = bs_utils.dot(A, v1)
        q_i = bs_utils.dot(A.T, v1)
        s_e = f_flow * s_e
        dm = (q_e + q_i - s_e - s_i) * dt
        m = m_ini + dm
    return dm, f_flow


def _calculate_changes_of_all_variables(system, time, dt, f_flow):
    """ Calculates the changes of all variable in every box.

    Args:
        time (pint.Quantity [T]): Current time (age) of the system.
        dt (pint.Quantity [T]): Timestep used.
        f_flow (numpy 1D array): Reduction coefficient of the mass flows 
            due to empty boxes.

    Returns:
        dvar (numpy 2D array of pint.Quantities): Variables changes of 
            every box. First dimension are the boxes, second dimension
            are the variables.

    """
    # reduction coefficent of the "variable-sinks" of each box for the
    # treated variable
    # scaling factor for sinks of each box
    f_var = np.ones([system.N_boxes, system.N_variables])
    var_ini = bs_utils.stack([system.get_variable_mass_1Darray(
        variable) for variable in system.variable_list], axis=-1)

    while True:
        dvar_list, net_sink_list, net_source_list = zip(*[_get_dvar(
            system, variable, time, dt, f_var, f_flow) 
            for variable in system.variable_list])
        dvar = bs_utils.stack(dvar_list, axis=-1)
        net_sink = bs_utils.stack(net_sink_list, axis=-1)
        net_source = bs_utils.stack(net_source_list, axis=-1)

        var = (var_ini + dvar).to_base_units()
        
        net_sink[net_sink.magnitude == 0] = np.nan  # to evade division by zero

        f_var_tmp = ((var_ini + net_source) / net_sink).magnitude 
        f_var_tmp[np.isnan(f_var_tmp)] = 1
        f_var_tmp[f_var_tmp > 1] = 1

        # If any element of f_var_tmp is smaller than one this means that
        # for at least one variable in one box the sinks are bigger than
        # the sum of the source and the already present variable mass.
        # Thus: The mass of this variable would fall below zero!
        # Reduce the sinks proportional to the ratio of the sources and 
        # the already present variable mass to the sinks.
        if np.any(f_var_tmp < 1):
            # To be sure that the sinks are reduced enough and to 
            # evade any rouding errors the reduction ratio of the sinks
            # (f_var_tmp) is further decreased by a very small number.
            f_var_tmp[f_var_tmp < 1] -= 1e-15 # np.nextafter(0, 1)
            f_var *= f_var_tmp
        else:
            break
    return dvar


def _get_sink_source_flow(system, variable, time, dt, f_var, f_flow):
    v1 = np.ones(system.N_boxes)
    flows = system.flows
    A_flow = system.get_variable_internal_flow_2Darray(variable, 
            time, f_flow, flows)
    A_flow = (A_flow.T * f_var[:, variable.id]).T
    s_flow_i = bs_utils.dot(A_flow, v1)
    q_flow_i = bs_utils.dot(A_flow.T, v1)
    s_flow_e = system.get_variable_flow_sink_1Darray(variable, 
            time, f_flow, flows)
    s_flow_e = system.get_variable_flow_sink_1Darray(variable, 
            time, f_flow, flows) * f_var[:, variable.id]

    q_flow_e = system.get_variable_flow_source_1Darray(variable,
            time, flows)
    sink_flow = ((s_flow_i + s_flow_e) * dt).to_base_units()
    source_flow = ((q_flow_i + q_flow_e) * dt).to_base_units()
    return sink_flow, source_flow


def _get_sink_source_flux(system, variable, time, dt, f_var):
    v1 = np.ones(system.N_boxes)
    fluxes = system.fluxes
    A_flux = system.get_variable_internal_flux_2Darray(variable,
            time, fluxes)
    A_flux = (A_flux.T * f_var[:, variable.id]).T
    s_flux_i = bs_utils.dot(A_flux, v1)
    q_flux_i = bs_utils.dot(A_flux.T, v1)
    s_flux_e = system.get_variable_flux_sink_1Darray(variable, 
            time, fluxes)
    s_flux_e = system.get_variable_flux_sink_1Darray(variable, 
            time, fluxes) * f_var[:, variable.id]

    q_flux_e = system.get_variable_flux_source_1Darray(variable,
            time, fluxes)
    sink_flux = ((s_flux_i + s_flux_e) * dt).to_base_units()
    source_flux = ((q_flux_i + q_flux_e) * dt).to_base_units()
    dvar_flux = source_flux - sink_flux
    return sink_flux, source_flux


def _get_sink_source_process(system, variable, time, dt, f_var):
    processes = system.processes
    s_process = system.get_variable_process_sink_1Darray(variable, 
            time, processes)
    s_process = system.get_variable_process_sink_1Darray(variable, 
            time, processes) * f_var[:, variable.id]
    q_process = system.get_variable_process_source_1Darray(variable,
            time, processes)
    sink_process = (s_process * dt).to_base_units()
    source_process = (q_process * dt).to_base_units()
    return sink_process, source_process


def _get_sink_source_reaction(system, variable, time, dt, f_var):
    reactions = system.reactions
    rr_cube = system.get_reaction_rate_3Darray(time, reactions)

    ## APPLY CORRECTIONS HERE!
    if np.any(f_var < 1):
        f_rr_cube = np.ones_like(rr_cube)
        for index in np.argwhere(f_var < 1):
            reduction_factor = f_var[tuple(index)]
            box = system.box_list[index[0]]
            box_name = box.name
            variable_name = system.variable_list[index[1]].name
            sink_reaction_indecies = np.argwhere(rr_cube[index[0], index[1], :].magnitude < 0)
            sink_reaction_indecies = list(sink_reaction_indecies.flatten())

            for sink_reaction_index in sink_reaction_indecies:
                if f_rr_cube[index[0], index[1], sink_reaction_index] > reduction_factor:
                    f_rr_cube[index[0], :, sink_reaction_index] = reduction_factor
        rr_cube *= f_rr_cube

    # Set all positive values to 0
    sink_rr_cube = np.absolute(rr_cube.magnitude.clip(max=0)) * rr_cube.units
    # Set all negative values to 0
    source_rr_cube = rr_cube.magnitude.clip(min=0) * rr_cube.units
    s_reaction = sink_rr_cube.sum(axis=2)[:, variable.id]
    q_reaction = source_rr_cube.sum(axis=2)[:, variable.id]
    
    sink_reaction = (s_reaction * dt).to_base_units()
    source_reaction = (q_reaction * dt).to_base_units()
    return sink_reaction, source_reaction


def _get_dvar(system, variable, time, dt, f_var, f_flow):
    # Get variables sources (q) and sinks (s)
    # i=internal, e=external

    sink_flow, source_flow = _get_sink_source_flow(
            system, variable, time, dt, f_var, f_flow)
    sink_flux, source_flux = _get_sink_source_flux(
            system, variable, time, dt, f_var)
    sink_process, source_process = _get_sink_source_process(
            system, variable, time, dt, f_var)
    sink_reaction, source_reaction = _get_sink_source_reaction(
            system, variable, time, dt, f_var)

    net_sink = sink_flow + sink_flux + sink_process + sink_reaction
    net_source = (source_flow + source_flux + source_process + 
            source_reaction)
    
    net_sink = net_sink.to_base_units()
    net_source = net_source.to_base_units()
    dvar = (net_source - net_sink).to_base_units()
    return dvar, net_sink, net_source
























































class Solver:
    """Class that simulates the evolution of a BoxModelSystem in time.

    Functions:
        solve: Solve the complete system. That means all Fluid mass flows
            are calculated together with processes/reactions/fluxes/flows 
            of variables that are traced within the system. Returns a 
            Solution instance which contains the time series of all system 
            quantities and can also plot them.

    Attributes:
        system (System): System which is simulated.

    """

    def __init__(self, system):
        self.system_initial = system

    def solve(self, total_integration_time, dt, debug=False):
        """Simulate the time evolution of all variables within the system.

        Collect all information about the system, create differential 
        equations from this information and integrate them (numercially)
        into the future.

        Args:
            system (System): The system that is simulated.
            total_integration_time (pint.Quantity [T]): The time span which
                the system should be solved into the "future". The system will
                be simulated for the time period zero to approximately 
                total_integration_time (depending whether 
                total_integration_time is a multiple of dt; if not the real
                integration horizon will be be bigger than 
                [0, total_integration_time]).
            dt (pint.Quantity [T]): Size of the timestep for the simulation.
                The bigger the timestep the faster the simulation will be
                calculated, however, if the timestep is chosen too high
                there can arise numerical instabilites!
            debug (bool): Activates debugging mode (pdb.set_trace()).
                Defaults to False.

        """
        # Start time of function
        func_start_time = time_module.time()

        # Saves the time since the start of the simulate at which the last 
        # save (pickling) was conducted
        last_save_timedelta = 0  

        if debug:
            pdb.set_trace()
                
        # Get number of time steps - round up if there is a remainder
        N_timesteps = math.ceil(total_integration_time / dt)
        # Recalculate total integration time based on the number of timesteps
        total_integration_time = N_timesteps * dt

        print('Start solving the BoxModelSystem...')
        print('- total integration time: {}'.format(total_integration_time))
        print('- dt (time step): {}'.format(dt))
        print('- number of time steps: {}'.format(N_timesteps))

        time = total_integration_time * 0
        self.system = copy.deepcopy(self.system_initial)
        sol = bs_solution.Solution(self.system, N_timesteps, dt)

        # Save initial state to solution
        for box in self.system.box_list:
            # sol.df.loc[0] = np.nan
            sol.df.loc[0, (box.name,'mass')] = box.fluid.mass.magnitude
            sol.df.loc[0, (box.name,'volume')] = \
                    self.system.get_box_volume(box).magnitude
            for variable in self.system.variable_list:
                var_name = variable.name
                sol.df.loc[0, (box.name,var_name)] = \
                        box.variables[var_name].mass.magnitude

        progress = 0
        for timestep in range(N_timesteps):
            # Calculate progress in percentage of processed timesteps
            progress_old = progress
            progress = int(float(timestep) / float(N_timesteps)*10) * 10.0
            if progress != progress_old:
                print("{}%".format(progress))
                
                # Check if simulation is running since more than a minute
                # since the last save was conducted.
                time_since_last_save = (time_module.time() - func_start_time 
                        - last_save_timedelta)
                if time_since_last_save > 6:
                    last_save_timedelta = time_module.time() - func_start_time 
                    self.save('{}_TS{}.pickle'.format(self.system.name, 
                        timestep))

            time += dt

            ##################################################
            # Calculate Mass fluxes
            ##################################################

            dm, f_flow = self._calculate_mass_flows(time, dt)

            ##################################################
            # Calculate Variable changes due to PROCESSES,
            # REACTIONS, FUXES and FLOWS
            ##################################################

            dvar = self._calculate_changes_of_all_variables(
                    time, dt, f_flow)

            ##################################################
            # Apply changes to Boxes and save values to
            # Solution instance
            ##################################################

            for box in self.system.box_list:
                # Write changes to box objects
                box.fluid.mass += dm[box.id]

                # Save mass to Solution instance
                sol.df.loc[timestep, (box.name, 'mass')] = \
                        box.fluid.mass.magnitude
                sol.df.loc[timestep, (box.name, 'volume')] = \
                        self.system.get_box_volume(box).magnitude

                for variable in self.system.variable_list:
                    var_name = variable.name
                    self.system.boxes[box.name].variables[var_name].mass += \
                            dvar[box.id, variable.id]
                    sol.df.loc[timestep, (box.name,variable.name)] = \
                            box.variables[variable.name].mass.magnitude

        # End Time of Function
        func_end_time = time_module.time()
        print(
            'Function "solve(...)" used {:3.3f}s'.format(
                func_end_time - func_start_time))
        return sol


    # PICKLING

    def save(self, file_name):
        """Pickle instance and save to file_name."""
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, file_name):
        """Load pickled instance from file_name."""
        with open(file_name, 'rb') as f:
            solution = pickle.load(f)
            if not isinstance(solution, Solution):
                raise ValueError(
                        'Loaded pickle object is not a Solution instance!')
        return solution


    # HELPER functions

    def _calculate_mass_flows(self, time, dt):
        """Calculate mass changes of every box.

        Args:
            time (pint.Quantity [T]): Current time (age) of the system.
            dt (pint.Quantity [T]): Timestep used.

        Returns:
            dm (numpy 1D array of pint.Quantities): Mass changes of every box.
            f_flow (numpy 1D array): Reduction coefficient of the mass 
                flows (due to becoming-empty boxes -> box mass cannot 
                decrase below 0kg).

        """
        # f_flow is the reduction coefficent of the "sink-flows" of each box
        # scaling factor for sinks of each box
        f_flow = np.ones(self.system.N_boxes)
        v1 = np.ones(self.system.N_boxes)

        m_ini = self.system.get_fluid_mass_1Darray()

        # get internal flow matrix and calculate the internal souce and sink 
        # vectors. Also get the external sink and source vector
        A = self.system.get_fluid_mass_internal_flow_2Darray(time)
        # internal 
        s_i = bs_utils.dot(A, v1)
        q_i = bs_utils.dot(A.T, v1)
        s_e = self.system.get_fluid_mass_flow_sink_1Darray(time)
        q_e = self.system.get_fluid_mass_flow_source_1Darray(time)

        # calculate first estimate of mass change vector
        dm = (q_e + q_i - s_e - s_i) * dt
        # calculate first estimate of mass after timestep
        m = m_ini + dm

        while np.any(m.magnitude < 0):
            argmin = np.argmin(m)
            # Calculate net sink and source and mass of the 'empty' box.
            net_source = (q_e[argmin] + q_i[argmin])*dt
            net_sink = (s_e[argmin] + s_i[argmin])*dt
            available_mass = m_ini[argmin]
            total_mass = (net_source + available_mass).to_base_units()

            if total_mass.magnitude > 0: 
                f_new = (total_mass / net_sink).to_base_units().magnitude 
                f_flow[argmin] = min(f_new, f_flow[argmin] * 0.98)
            else:
                f_flow[argmin] = 0

            # Apply reduction of sinks of the box
            A = (A.T * f_flow).T
            s_i = bs_utils.dot(A, v1)
            q_i = bs_utils.dot(A.T, v1)
            s_e = f_flow * s_e
            dm = (q_e + q_i - s_e - s_i) * dt
            m = m_ini + dm
        return dm, f_flow

    def _calculate_changes_of_all_variables(self, time, dt, f_flow):
        """ Calculates the changes of all variable in every box.

        Args:
            time (pint.Quantity [T]): Current time (age) of the system.
            dt (pint.Quantity [T]): Timestep used.
            f_flow (numpy 1D array): Reduction coefficient of the mass flows 
                due to empty boxes.

        Returns:
            dvar (numpy 2D array of pint.Quantities): Variables changes of 
                every box. First dimension are the boxes, second dimension
                are the variables.

        """
        # reduction coefficent of the "variable-sinks" of each box for the
        # treated variable
        # scaling factor for sinks of each box
        f_var = np.ones([self.system.N_boxes, self.system.N_variables])
        var_ini = bs_utils.stack([self.system.get_variable_mass_1Darray(
            variable) for variable in self.system.variable_list], axis=-1)

        while True:
            dvar_list, net_sink_list, net_source_list = zip(*[self._get_dvar(
                variable, time, dt, f_var, f_flow) 
                for variable in self.system.variable_list])
            dvar = bs_utils.stack(dvar_list, axis=-1)
            net_sink = bs_utils.stack(net_sink_list, axis=-1)
            net_source = bs_utils.stack(net_source_list, axis=-1)

            var = (var_ini + dvar).to_base_units()
            
            net_sink[net_sink.magnitude == 0] = np.nan  # to evade division by zero

            f_var_tmp = ((var_ini + net_source) / net_sink).magnitude 
            f_var_tmp[np.isnan(f_var_tmp)] = 1
            f_var_tmp[f_var_tmp > 1] = 1

            # If any element of f_var_tmp is smaller than one this means that
            # for at least one variable in one box the sinks are bigger than
            # the sum of the source and the already present variable mass.
            # Thus: The mass of this variable would fall below zero!
            # Reduce the sinks proportional to the ratio of the sources and 
            # the already present variable mass to the sinks.
            if np.any(f_var_tmp < 1):
                # To be sure that the sinks are reduced enough and to 
                # evade any rouding errors the reduction ratio of the sinks
                # (f_var_tmp) is further decreased by a very small number.
                f_var_tmp[f_var_tmp < 1] -= 1e-15 # np.nextafter(0, 1)
                f_var *= f_var_tmp
            else:
                break
        return dvar

    def _get_sink_source_flow(self, variable, time, dt, f_var, f_flow):
        v1 = np.ones(self.system.N_boxes)
        flows = self.system.flows
        A_flow = self.system.get_variable_internal_flow_2Darray(variable, 
                time, f_flow, flows)
        A_flow = (A_flow.T * f_var[:, variable.id]).T
        s_flow_i = bs_utils.dot(A_flow, v1)
        q_flow_i = bs_utils.dot(A_flow.T, v1)
        s_flow_e = self.system.get_variable_flow_sink_1Darray(variable, 
                time, f_flow, flows)
        s_flow_e = self.system.get_variable_flow_sink_1Darray(variable, 
                time, f_flow, flows) * f_var[:, variable.id]

        q_flow_e = self.system.get_variable_flow_source_1Darray(variable,
                time, flows)
        sink_flow = ((s_flow_i + s_flow_e) * dt).to_base_units()
        source_flow = ((q_flow_i + q_flow_e) * dt).to_base_units()
        return sink_flow, source_flow

    def _get_sink_source_flux(self, variable, time, dt, f_var):
        v1 = np.ones(self.system.N_boxes)
        fluxes = self.system.fluxes
        A_flux = self.system.get_variable_internal_flux_2Darray(variable,
                time, fluxes)
        A_flux = (A_flux.T * f_var[:, variable.id]).T
        s_flux_i = bs_utils.dot(A_flux, v1)
        q_flux_i = bs_utils.dot(A_flux.T, v1)
        s_flux_e = self.system.get_variable_flux_sink_1Darray(variable, 
                time, fluxes)
        s_flux_e = self.system.get_variable_flux_sink_1Darray(variable, 
                time, fluxes) * f_var[:, variable.id]

        q_flux_e = self.system.get_variable_flux_source_1Darray(variable,
                time, fluxes)
        sink_flux = ((s_flux_i + s_flux_e) * dt).to_base_units()
        source_flux = ((q_flux_i + q_flux_e) * dt).to_base_units()
        dvar_flux = source_flux - sink_flux
        return sink_flux, source_flux

    def _get_sink_source_process(self, variable, time, dt, f_var):
        processes = self.system.processes
        s_process = self.system.get_variable_process_sink_1Darray(variable, 
                time, processes)
        s_process = self.system.get_variable_process_sink_1Darray(variable, 
                time, processes) * f_var[:, variable.id]
        q_process = self.system.get_variable_process_source_1Darray(variable,
                time, processes)
        sink_process = (s_process * dt).to_base_units()
        source_process = (q_process * dt).to_base_units()
        return sink_process, source_process

    def _get_sink_source_reaction(self, variable, time, dt, f_var):
        reactions = self.system.reactions
        rr_cube = self.system.get_reaction_rate_3Darray(time, reactions)

        ## APPLY CORRECTIONS HERE!
        if np.any(f_var < 1):
            f_rr_cube = np.ones_like(rr_cube)
            for index in np.argwhere(f_var < 1):
                reduction_factor = f_var[tuple(index)]
                box = self.system.box_list[index[0]]
                box_name = box.name
                variable_name = self.system.variable_list[index[1]].name
                sink_reaction_indecies = np.argwhere(rr_cube[index[0], index[1], :].magnitude < 0)
                sink_reaction_indecies = list(sink_reaction_indecies.flatten())

                for sink_reaction_index in sink_reaction_indecies:
                    if f_rr_cube[index[0], index[1], sink_reaction_index] > reduction_factor:
                        f_rr_cube[index[0], :, sink_reaction_index] = reduction_factor
            rr_cube *= f_rr_cube


        # Set all positive values to 0
        sink_rr_cube = np.absolute(rr_cube.magnitude.clip(max=0)) * rr_cube.units
        # Set all negative values to 0
        source_rr_cube = rr_cube.magnitude.clip(min=0) * rr_cube.units
        s_reaction = sink_rr_cube.sum(axis=2)[:, variable.id]
        q_reaction = source_rr_cube.sum(axis=2)[:, variable.id]
        
        sink_reaction = (s_reaction * dt).to_base_units()
        source_reaction = (q_reaction * dt).to_base_units()
        return sink_reaction, source_reaction

    def _get_dvar(self, variable, time, dt, f_var, f_flow):
        # Get variables sources (q) and sinks (s)
        # i=internal, e=external

        sink_flow, source_flow = self._get_sink_source_flow(variable, 
                time, dt, f_var, f_flow)
        sink_flux, source_flux = self._get_sink_source_flux(variable, 
                time, dt, f_var)
        sink_process, source_process = self._get_sink_source_process(
                variable, time, dt, f_var)
        sink_reaction, source_reaction = self._get_sink_source_reaction(
                variable, time, dt, f_var)

        net_sink = sink_flow + sink_flux + sink_process + sink_reaction
        net_source = (source_flow + source_flux + source_process + 
                source_reaction)
        
        net_sink = net_sink.to_base_units()
        net_source = net_source.to_base_units()
        dvar = (net_source - net_sink).to_base_units()
        return dvar, net_sink, net_source

