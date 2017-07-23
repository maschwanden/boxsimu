# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 10:37UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import pdb
import copy
import time as time_module
import numpy as np
import matplotlib.pyplot as plt
from attrdict import AttrDict
import math

from . import solution as bs_solution
from . import utils as bs_utils
from . import dimensionality_validation as bs_dim_val

DEBUG = False


def dprint(*args):
    if DEBUG:
        print(*args)


def dinput(*args):
    if DEBUG:
        return input(*args)


class Solver:
    """Class that simulates the evolution of a BoxModelSystem in time.

    Functions:
        solve: Solve the complete system. That means all Fluid mass flows
            are calculated together with processes/reactions/fluxes/flows of variables
            that are traced within the system. Returns a Solution instance which contains
            the time series of all system quantities and can also plot them.
        solve_flows: Solve only the Fluid mass flows of the system. That means
            no variable is traced. Fast function to check if Fluid masses of the boxes
            and associated flows thereof are set correctly. Returns a Solution instance
            which contains the time series of the fluid masses of all boxes of the system
            and can also plot them.

    Attributes:
        system (System): System which is simulated.

    """

    def __init__(self, system):
        self.system = system
        self._system_initial = copy.deepcopy(system)

    def solve_flows(self, total_integration_time, dt):
        print('Start solving the flows of the box model...')
        print('- total integration time: {}'.format(total_integration_time))
        print('- dt (time step): {}'.format(dt))

        bs_dim_val.raise_if_not_time(total_integration_time)
        bs_dim_val.raise_if_not_time(dt)

        # Get number of time steps - round up if there is a remainder
        N_timesteps = math.ceil(total_integration_time / dt)
        time = total_integration_time * 0

        # Start time of function
        start_time = time_module.time()

        sol = bs_solution.Solution(total_integration_time, dt, 
                self.system)

        progress = 0
        for i in range(N_timesteps):
            # Calculate progress in percentage of processed timesteps
            progress_old = progress
            progress = round(float(i) / float(N_timesteps), 1) * 100.0
            if progress != progress_old:
                print("{}%".format(progress))
            sol.time.append(time)

            dm, f = self._calculate_mass_flows(time, dt)

            for box_name, box in self.system.boxes.items():
                # Write changes to box objects
                box.fluid.mass += dm[box.id]

                # Save to Solution instance
                sol.ts[box_name]['mass'].append(box.fluid.mass)

            time += dt

        # End Time of Function
        end_time = time_module.time()
        print('Function "solve_flows(...)" used {:3.3f}s'.format(
                end_time - start_time))
        return sol

    def solve(self, total_integration_time, dt, debug=True):
        global DEBUG
        DEBUG = debug

        print('Start solving the box model...')
        print('- integration time: {}'.format(total_integration_time))
        print('- time step: {}'.format(dt))

        dprint('----> mm: ', self.system.mm)
        dprint('----> sum(mm): ', self.system.mm.sum())

        # Start time of function
        start_time = time_module.time()

        # Get number of time steps - round up if there is a remainder
        N_timesteps = int(total_integration_time / dt)
        time = total_integration_time * 0

        sol = solution.Solution(
            total_integration_time, dt, self.system.box_list)

        progress = 0
        for i in range(N_timesteps):
            progress_old = progress
            progress = round(float(i) / float(N_timesteps), 1) * 100.0
            if progress != progress_old:
                print("{}%".format(progress))
            dprint('Time: {}'.format(time))
            sol.time.append(time)
            if time.magnitude > 140:
                #DEBUG = True
                # pdb.set_trace()
                pass

            ##################################################
            # Calculate Mass fluxes
            ##################################################

            m_change, m_units, A = self._calculate_mass_flows(time, dt)

            ##################################################
            # Calculate Variable changes due to PROCESSES,
            # REACTIONS, FUXES and finally due to the FLOWS
            ##################################################

            var_changes = {}
            var_units = {}

            for var_name, var in self.system.variables.items():
                dprint('\n\n\n\n\nSOLVE for variable {}'.format(var_name))
                var_changes[var_name] = self._calculate_changes_of_variable(
                    time, dt, var, A)

            dprint('var_changes: {}'.format(var_changes))
            dprint('var_units: {}'.format(var_units))

            ##################################################
            # Apply changes to Boxes and save values to
            # Solution instance
            ##################################################

            dprint('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            for box_name, box in self.system.boxes.items():
                for var_name, var in box.variables.items():
                    dprint(
                        '1111self.boxes[{}].variables[{}].mass: '.format(
                            box_name,
                            var_name),
                        self.system.boxes[box_name].variables[var_name].mass)
            dprint('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

            for box_name, box in self.system.boxes.items():
                # Write changes to box objects
                box.fluid.mass += m_change[box.id] * m_units

                # Save mass to Solution instance
                sol.ts[box_name]['mass'].append(box.fluid.mass)

                for var_name, var in box.variables.items():

                    dprint('__________________________________')
                    dprint('$$$$$$1111self.boxes[{}].variables[{}].mass: '.format(
                        box_name, var_name), self.system.boxes[box_name].variables[var_name].mass)
                    dprint('box_name: ', box_name)
                    dprint(var_name + ':')
                    dprint(var_changes[var_name][box.id])
                    dprint(var_units[var_name])

                    dprint('@@@@ : ', var_changes[var_name][box.id])
                    dprint('@@@@ : ', var_units[var_name])
                    dprint(
                        '@@@@ : ',
                        self.system.boxes[box_name].variables[var_name].mass)

                    new_var_mass = var_changes[var_name][box.id] * var_units[var_name] + \
                        self.system.boxes[box_name].variables[var_name].mass

                    dprint('@@@@ : ', new_var_mass)
                    dprint(box_name, var_name)
                    self.system.boxes[box_name].variables[var_name].mass = new_var_mass

                    sol.ts[box_name][var_name].append(
                        self.system.boxes[box_name].variables[var_name].mass)

                    self.system.mm.loc[box_name, var_name] = new_var_mass
                    if box.fluid.mass.magnitude > 0:
                        self.system.cm.loc[box_name,
                                           var_name] = new_var_mass / box.fluid.mass
                    else:
                        self.system.cm.loc[box_name, var_name] = 0

                    dprint('$$$$$$2222self.boxes[{}].variables[{}].mass: '.format(
                        box_name, var_name), self.system.boxes[box_name].variables[var_name].mass)
            time += dt

            dprint('----> mm: ', self.system.mm)
            dprint('----> sum(mm): ', self.system.mm.sum())
            a = dinput('(End of timestep loop) Wait for it...')

        # End Time of Function
        end_time = time_module.time()
        print(
            'Function "solve(...)" used {:3.3f}s'.format(
                end_time - start_time))

        return sol

    # HELPER functions

    def _calculate_mass_flows(self, time, dt):
        """Calculate mass changes of every box.

        Args:
            time (pint.Quantity [T]): Current time (age) of the system.
            dt (pint.Quantity [T]): Timestep used.

        Returns:
            dm (numpy 1D array of pint.Quantities): Mass changes of every box.
            f (numpy 1D array): Reduction coefficient of the mass 
                flows (due to becoming-empty boxes -> box mass cannot 
                decrase below 0kg).

        """
        # f is the reduction coefficent of the "sink-flows" of each box
        # scaling factor for sinks of each box
        f = np.ones(self.system.N_boxes)
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
            net_source = q_e[argmin] + q_i[argmin]
            net_sink = s_e[argmin] + s_i[argmin]
            available_mass = m_ini[argmin]

            if (net_source + available_mass) > 0 and net_sink > 0:
                f[argmin] = min(float(net_source + available_mass) / 
                        float(net_sink), f[argmin] * 0.95)
            else:
                f[argmin] = 0

            # Apply reduction of sinks of the box
            A = (A.T * f).T
            s_i = bs_utils.dot(A, v1)
            q_i = bs_utils.dot(A.T, v1)
            s_e = f * s_e
            dm = (q_e + q_i - s_e - s_i) * dt
            m = m_ini + dm
        return dm, f

    def _calculate_changes_of_variable(self, time, dt, variable, f_flow):
        """ Calculates the changes of ONE variable in every box.

        Args:
            time (pint.Quantity [T]): Current time (age) of the system.
            dt (pint.Quantity [T]): Timestep used.
            f_flow (numpy 1D array): Reduction coefficient of the mass flows 
                due to empty boxes.

        Returns:
            dvar (numpy 1D array of pint.Quantities): Variables changes of 
                every box.

        """
        # reduction coefficent of the "variable-sinks" of each box for the
        # treated variable
        # scaling factor for sinks of each box
        f = np.ones(self.system.N_boxes)
        one_vec = np.ones(self.system.N_boxes)
        var, var_units = self.system.get_box_variable_mass_vector(variable)
        var_conc, var_conc_units = self.system.get_box_variable_concentration_vector(
            variable)

        var_conc_matrix = np.array([var_conc, ] * self.system.N_boxes).T

        var_ini = copy.deepcopy(var)

        # mass flows relevant for variable-advection
        # Tracer-Transport Flows
        tt_flows = [flow for flow in self.system.flows if flow.tracer_transport]
        tt_flows_with_variables = [
            flow for flow in tt_flows if len(
                flow.variables) > 0]
        mass_flow_A = self.system.get_flow_matrix(time, tt_flows)
        mass_flow_A = (mass_flow_A.transpose() * f_flow).transpose()
        mass_flow_s = self.system.get_flow_sink_vector(
            time, tt_flows_with_variables) * f_flow
        mass_flow_q = self.system.get_flow_source_vector(
            time, tt_flows) * f_flow

        # all of these vectors have units of kg/s
        flow_s = mass_flow_s * var_conc
        flow_q = mass_flow_q
        p_s, p_s_units = self.system.get_process_sink_vector_of_variable(
            time, variable)
        p_q, p_q_units = self.system.get_process_source_vector_of_variable(
            time, variable)
        # r_s, r_s_units = self.system.get_reaction_sink_vector_of_variable(time, variable)
        # r_q, r_q_units = self.system.get_reaction_source_vector_of_variable(time, variable)
        f_s, f_s_units = self.system.get_flux_sink_vector_of_variable(
            time, variable)
        f_q, f_q_units = self.system.get_flux_source_vector_of_variable(
            time, variable)

        dimless_dt = ((f_q_units / var_units) * dt).to_base_units().magnitude

        sources = (
            (p_q +
             r_q +
             f_q +
             np.dot(
                 mass_flow_A.T,
                 var_conc)) *
            dimless_dt)
        sinks = (
            (p_s +
             r_s +
             f_s +
             np.dot(
                 mass_flow_A *
                 var_conc_matrix,
                 one_vec)) *
            dimless_dt)
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
        dprint('np.dot(mass_flow_A * var_conc_matrix, one_vec): {}'.format(
            np.dot(mass_flow_A * var_conc_matrix, one_vec)))
        dprint('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

        dprint('------------------------------------')
        dprint('var_changes: {}'.format(var_changes))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint(
            'source due to Processes per timestep (np.dot(one_vec, p_q)): {}'.format(
                np.dot(
                    one_vec,
                    p_q)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Processes per timestep (- np.dot(one_vec, p_s)): {}'.format(- np.dot(one_vec, p_s)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint(
            'source due to Reactions per timestep (np.dot(one_vec, r_q)): {}'.format(
                np.dot(
                    one_vec,
                    r_q)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Reactions per timestep (- np.dot(one_vec, r_s)): {}'.format(- np.dot(one_vec, r_s)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint(
            'source due to Fluxes per timestep (np.dot(one_vec, f_q)): {}'.format(
                np.dot(
                    one_vec,
                    f_q)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Fluxes per timestep (- np.dot(one_vec, f_s)): {}'.format(- np.dot(one_vec, f_s)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint(
            'source due to Flows per timestep (np.dot(mass_flow_A.T, var_conc)): {}'.format(
                np.dot(
                    mass_flow_A.T,
                    var_conc)))
        dprint('------------------------------------')

        dprint('------------------------------------')
        dprint('sink due to Flows per timestep (- np.dot(mass_flow_A * var_conc_matrix, one_vec)): {}'.format(- np.dot(
            mass_flow_A * var_conc_matrix, one_vec)))
        dprint('------------------------------------')

        var = var_ini + var_changes

        while_counter = 0

        #print('VAR: ', var)
        #print('var_conc', var_conc)
        while np.any(var < 0):
            while_counter += 1
            dprint('in while.... {}'.format(while_counter))
            dprint('f before corrections', f)

            argmin = np.argmin(var)
            dprint('argmin: {}'.format(argmin))

            available_var = var_ini[argmin]

            dprint('available_var: ', available_var)

            sources = (
                (p_q +
                 r_q +
                 f_q +
                 np.dot(
                     mass_flow_A.T,
                     var_conc)) *
                dimless_dt)
            sinks = (
                (p_s +
                 r_s +
                 f_s +
                 np.dot(
                     mass_flow_A *
                     var_conc_matrix,
                     one_vec)) *
                dimless_dt)
            net_source = sources[argmin]
            net_sink = sinks[argmin]

            if (net_source + available_var) > 0:
                dprint('net_source+available_var > 0!')
                f[argmin] = min(
                    float(net_source + available_var) / float(net_sink),
                    f[argmin] * 0.95
                )
            else:
                dprint('net_source+available_var <= 0! -> f[argmin] = 0')
                f[argmin] = 0

            dprint('f after corrections: ', f)

            # Reduce Sinks and Sources accordingly to calculated f values
            mass_flow_A = (mass_flow_A.transpose() * f).transpose()
            flow_s = mass_flow_s * f
            mass_flow_q
            p_s = p_s * f
            r_s = r_s * f
            f_s = f_s * f

            sources = (
                (p_q +
                 r_q +
                 f_q +
                 np.dot(
                     mass_flow_A.T,
                     var_conc)) *
                dimless_dt)
            sinks = (
                (p_s +
                 r_s +
                 f_s +
                 np.dot(
                     mass_flow_A *
                     var_conc_matrix,
                     one_vec)) *
                dimless_dt)

            dprint('np.dot(mass_flow_A.T, var_conc) * f)',
                   np.dot(mass_flow_A.T, var_conc) * f)
            dprint('np.dot(mass_flow_A * var_conc_matrix, one_vec)*f',
                   np.dot(mass_flow_A * var_conc_matrix, one_vec) * f)

            # Calculated corrected var_changes
            var_changes = (sources - sinks)

            var = var_ini + var_changes

            dprint('var: ' + variable.name)
            dprint(
                'variable mass per box at the begining of this timestep: {}'.format(var_ini))
            dprint(
                'variable mass per box at the end of this timestep: {}'.format(
                    var_ini + var_changes))
            dprint('var_changes: ', var_changes)
            dprint('sources', sources)
            dprint('sinks', sinks)

            sdlkfahdl = dinput('AT THE END OF WHILE...')

        dprint('>>>>>>>>>>>')
        dprint('FINAL {} change: {}'.format(variable.name, var_changes))
        dprint('FINAL {}: {}'.format(variable.name, var))
        dprint('>>>>>>>>>>>')

        sdlfjsdll = dinput(
            'AT THE END OF CALCULATE VARIABLE MASS CHANGE FUNCTION')

        return var_changes, var_units

    def _calculate_reaction_variable_changes(self):
        pass

