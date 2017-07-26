# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import os
import unittest
from unittest import TestCase

import sys
import copy
import numpy as np
import datetime

from matplotlib import pyplot as plt

if not os.path.abspath(__file__ + "/../../../") in sys.path:
    sys.path.append(os.path.abspath(__file__ + "/../../../"))

from boxsimu.entities import Fluid, Variable
from boxsimu.box import Box
from boxsimu.transport import  Flow, Flux
from boxsimu.condition import Condition
from boxsimu.system import BoxModelSystem 
from boxsimu.process import Process, Reaction
from boxsimu.solver import Solver
from boxsimu import utils
from boxsimu.simulations import boxmodelsystem1
from boxsimu import ur


class BoxModelSystem1Test(TestCase):
    """Test boxsimu framework using a simple box model.
    
    The model tested herein is described in the 
    book "Modelling Methods for Marine Science".
    """

    def setUp(self, *args, **kwargs):
        self.system = boxmodelsystem1.get_system()
        self.solver = Solver(self.system)
        self.uo = self.system.boxes.upper_ocean
        self.do = self.system.boxes.deep_ocean

    def tearDown(self, *args, **kwargs):
       del(self.system)
       del(self.solver)
       del(self.uo)
       del(self.do)

    def assertPintQuantityAlmostEqual(self, q1, q2, **kwargs):
        q1 = q1.to_base_units()
        q2 = q2.to_base_units()
        self.assertAlmostEqual(q1.magnitude, q2.magnitude, **kwargs)
        self.assertEqual(q1.units, q2.units)

    #####################################################
    # Box Functions 
    #####################################################
    
    def test_mass(self):
        self.assertEqual(self.system.boxes.upper_ocean.mass, 
                3e16*1020*ur.kg)
        self.assertEqual(self.system.boxes.deep_ocean.mass, 
                1e18*1030*ur.kg)

    def test_volume(self):
        upper_ocean_context = self.system.get_box_context(
                self.system.boxes.upper_ocean)
        deep_ocean_context = self.system.get_box_context(
                self.system.boxes.deep_ocean)

        self.assertEqual(self.system.boxes.upper_ocean.get_volume(
            upper_ocean_context), 3e16*1020/1000.0 * ur.meter**3)
        self.assertEqual(self.system.boxes.deep_ocean.get_volume(
            upper_ocean_context), 1e18*1030/1000.0 * ur.meter**3)

    def test_concentration(self):
        pass


    #####################################################
    # System Base Functions 
    #####################################################
    
    def test_box_id(self):
        self.assertEqual(self.system.boxes.upper_ocean.id, 1)
        self.assertEqual(self.system.boxes.deep_ocean.id, 0)

    def test_variable_id(self):
        self.assertEqual(self.system.variables.po4.id, 0)

    def test_N_boxes(self):
        self.assertEqual(self.system.N_boxes, 2)
    
    def test_N_variables(self):
        self.assertEqual(self.system.N_variables, 1)

    def test_context_of_box(self):
        upper_ocean = self.system.boxes.upper_ocean
        deep_ocean = self.system.boxes.deep_ocean

        global_context = self.system.get_box_context()
        upper_ocean_context = self.system.get_box_context(upper_ocean)
        deep_ocean_context = self.system.get_box_context(deep_ocean)
        
        # Test accessability of the condition attributes
        self.assertEqual(global_context.T, 111 * ur.kelvin)
        self.assertEqual(upper_ocean_context.T, 333 * ur.kelvin)
        self.assertEqual(deep_ocean_context.T, 222 * ur.kelvin)

        # Test the accessability of the condition attributes of other boxes:
        self.assertEqual(global_context.upper_ocean.condition.T, 
                333 * ur.kelvin)
        self.assertEqual(global_context.deep_ocean.condition.T, 
                222 * ur.kelvin)

        self.assertEqual(upper_ocean_context.global_condition.T, 
                111 * ur.kelvin)
        self.assertEqual(upper_ocean_context.upper_ocean.condition.T, 
                333 * ur.kelvin)
        self.assertEqual(upper_ocean_context.deep_ocean.condition.T, 
                222 * ur.kelvin)

        self.assertEqual(deep_ocean_context.global_condition.T, 
                111 * ur.kelvin)
        self.assertEqual(deep_ocean_context.upper_ocean.condition.T, 
                333 * ur.kelvin)
        self.assertEqual(deep_ocean_context.deep_ocean.condition.T, 
                222 * ur.kelvin)

    def test_context_evaluation_lambda_func(self):
        upper_ocean = self.system.boxes.upper_ocean
        deep_ocean = self.system.boxes.deep_ocean

        global_context = self.system.get_box_context()
        upper_ocean_context = self.system.get_box_context(upper_ocean)
        deep_ocean_context = self.system.get_box_context(deep_ocean)

        lambda1 = lambda t, c: c.T / (111*ur.kelvin)
        self.assertEqual(lambda1(0*ur.second, global_context), 1)
        self.assertEqual(lambda1(0*ur.second, upper_ocean_context), 3)
        self.assertEqual(lambda1(0*ur.second, deep_ocean_context), 2)

        lambda2 = lambda t, c: (t / ur.second + (c.T / (111*ur.kelvin)) +
                c.upper_ocean.condition.T / (111*ur.kelvin))
        self.assertEqual(lambda2(0*ur.second, global_context), 4)
        self.assertEqual(lambda2(0*ur.second, upper_ocean_context), 6)
        self.assertEqual(lambda2(0*ur.second, deep_ocean_context), 5)

        self.assertEqual(lambda2(100*ur.second, global_context), 104)
        self.assertEqual(lambda2(100*ur.second, upper_ocean_context), 106)
        self.assertEqual(lambda2(100*ur.second, deep_ocean_context), 105)

        # Set the variable concentration to nonzero values in a copy of system:
        self.system.boxes.upper_ocean.variables.po4.mass = 5 * ur.kg
        self.system.boxes.deep_ocean.variables.po4.mass = 10 * ur.kg

        global_context = self.system.get_box_context()
        upper_ocean_context = self.system.get_box_context(upper_ocean)
        deep_ocean_context = self.system.get_box_context(deep_ocean)

        lambda3 = lambda t, c: (t / ur.second + (c.T / (111*ur.kelvin)) + 
                c.upper_ocean.condition.T / (111*ur.kelvin) + (c.po4/ur.kg)**2)
        self.assertEqual(lambda3(100*ur.second, upper_ocean_context), 131)
        self.assertEqual(lambda3(100*ur.second, deep_ocean_context), 205)

        lambda4 = lambda t, c: ((c.po4/ur.kg) / 
                (c.upper_ocean.variables.po4/ur.kg))
        self.assertEqual(lambda4(0*ur.second, upper_ocean_context), 1)
        self.assertEqual(lambda4(0*ur.second, deep_ocean_context), 2)


    #####################################################
    # Fluid and Variable Mass/Concentration Vectors/Matrices
    #####################################################

    def test_fluid_mass_1Darray(self):
        m = self.system.get_fluid_mass_1Darray()
        self.assertEqual(m[self.uo.id], 3e16*1020*ur.kg)
        self.assertEqual(m[self.do.id], 1e18*1030*ur.kg)

    def test_variable_mass_1Darray(self):
        po4 = self.system.variables['po4']

        m = self.system.get_variable_mass_1Darray(po4)
        self.assertEqual(m[self.uo.id], 0*ur.kg)
        self.assertEqual(m[self.do.id], 0*ur.kg)

    def test_variable_concentration_1Darray(self):
        po4 = self.system.variables['po4']

        c = self.system.get_variable_concentration_1Darray(po4)
        self.assertEqual(c[self.uo.id], 0 * ur.dimensionless)
        self.assertEqual(c[self.do.id], 0 * ur.dimensionless)

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def test_fluid_mass_internal_flow_2Darray(self):
        A = self.system.get_fluid_mass_internal_flow_2Darray(0*ur.second, 
                self.system.flows)
        # Check that diagonal elements are zero
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i][i], 0 * ur.kg/ur.second)
        # Check that the other values are set correctly
        uo_do_exchange_rate = (6e17*ur.kg/ur.year).to_base_units()
        self.assertEqual(A[self.uo.id][self.do.id], uo_do_exchange_rate)
        self.assertEqual(A[self.do.id][self.uo.id], uo_do_exchange_rate)

    def test_fluid_mass_flow_sink_1Darray(self):
        s = self.system.get_fluid_mass_flow_sink_1Darray(0*ur.second, 
                self.system.flows)
        # Upper Ocean Sink: Due to evaporation (3e16)
        evaporation_rate = (3e16*ur.kg/ur.year).to_base_units()
        self.assertEqual(s[self.uo.id], evaporation_rate)
        self.assertEqual(s[self.do.id], 0 * ur.kg/ur.second)

    def test_fluid_mass_flow_source_1Darray(self):
        q = self.system.get_fluid_mass_flow_source_1Darray(0*ur.second, 
                self.system.flows)
        # Upper Ocean Source: Due to river discharge (3e16)
        river_discharge_rate = (3e16*ur.kg/ur.year).to_base_units()
        self.assertEqual(q[self.uo.id], river_discharge_rate)
        self.assertEqual(q[self.do.id], 0 * ur.kg/ur.second)


    #####################################################
    # Variable Sink/Source Vectors
    #####################################################

    def test_variable_internal_flow_2Darray(self):
        var = self.system.variables['po4']
        f_flow = np.ones(self.system.N_boxes)
        A = self.system.get_variable_internal_flow_2Darray(var, 0*ur.second, 
                f_flow)

        # Check that diagonal elements are zero
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i][i], 0 * ur.kg/ur.year)
        self.assertEqual(A[self.uo.id][self.do.id], 0 * ur.kg/ur.year)
        self.assertEqual(A[self.do.id][self.uo.id], 0 * ur.kg/ur.year)

        #########
        # Alternative Test with non-zero concentrations in the boxes:
        uo = self.system.boxes.upper_ocean 
        do = self.system.boxes.deep_ocean 

        uo.variables.po4.mass = 3.06e11*ur.kg
        do.variables.po4.mass = 1.03e14*ur.kg
        
        var = self.system.variables['po4']
        f_flow = np.ones(self.system.N_boxes)
        A = self.system.get_variable_internal_flow_2Darray(var, 0*ur.second, 
                f_flow)

        # Check that diagonal elements are zero
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i][i], 0 * ur.kg/ur.second)

        # Mass Flow from upper_to_deep ocean: 
        uo_do_exchange_rate = (6e17*ur.kg/ur.year).to_base_units()

        # Mass upper_ocean: 3e16*1020kg = 3.06e19
        # PO4 mass upper_ocean: 3.06e11kg
        # PO4 concentration upper_ocean: 3.06e11 / 3.06e19 = 1e-8
        # Transported PO4 from upper to deep ocean: uo_do_exchange_rate * 1e-8
        self.assertPintQuantityAlmostEqual(A[uo.id][do.id], 
                1e-8*uo_do_exchange_rate, places=2)

        # Mass deep_ocean: 1e18*1030kg = 1.03e21
        # PO4 mass deep_ocean: 1.03e14kg
        # PO4 concentration deep_ocean: 1.03e14 / 1.03e21 = 1e-7
        # Transported PO4 from upper to deep ocean: uo_do_exchange_rate * 1e-7
        self.assertPintQuantityAlmostEqual(A[do.id][uo.id], 
                1e-7*uo_do_exchange_rate, places=2)

    def test_variable_flow_sink_1Darray(self):
        var = self.system.variables['po4']
        f_flow = np.ones(self.system.N_boxes)
        s = self.system.get_variable_flow_sink_1Darray(var, 0*ur.second, f_flow)
        self.assertEqual(s[self.uo.id], 0*ur.kg/ur.second)
        self.assertEqual(s[self.do.id], 0*ur.kg/ur.second)

    def test_variable_flow_source_1Darray(self):
        var = self.system.variables['po4']
        q = self.system.get_variable_flow_source_1Darray(var, 0*ur.second)
        river_discharge_rate = (3e16*ur.kg/ur.year).to_base_units()
        # Upper Ocean Source: 3e16 * 4.6455e-8 = 1393650000.0
        self.assertPintQuantityAlmostEqual(q[self.uo.id], 
                river_discharge_rate*4.6455e-8)
        self.assertEqual(q[self.do.id], 0 * ur.kg/ur.second)

    def test_variable_process_sink_1Darray(self):
        var = self.system.variables['po4']
        s = self.system.get_variable_process_sink_1Darray(var, 1*ur.second)
        self.assertEqual(s[self.uo.id], 0 * ur.kg/ur.second)
        self.assertEqual(s[self.do.id], 0 * ur.kg/ur.second)

    def test_variable_process_source_1Darray(self):
        var = self.system.variables['po4']
        q = self.system.get_variable_process_source_1Darray(var, 0*ur.second)
        self.assertEqual(q[self.uo.id], 0 * ur.kg/ur.second)
        self.assertEqual(q[self.do.id], 0 * ur.kg/ur.second)

    def test_variable_internal_flux_2Darray(self):
        var = self.system.variables['po4']
        A = self.system.get_variable_internal_flux_2Darray(var, 0*ur.second)
        self.assertEqual(A[self.uo.id][self.do.id], 0 * ur.kg/ur.second)
        self.assertEqual(A[self.do.id][self.uo.id], 0 * ur.kg/ur.second)

    def test_variable_flux_sink_1Darray(self):
        var = self.system.variables['po4']
        s = self.system.get_variable_flux_sink_1Darray(var, 1*ur.second)
        self.assertEqual(s[self.uo.id], 0 * ur.kg/ur.second)
        self.assertEqual(s[self.do.id], 0 * ur.kg/ur.second)

    def test_variable_flux_source_1Darray(self):
        var = self.system.variables['po4']
        q = self.system.get_variable_flux_source_1Darray(var, 0*ur.second)
        self.assertEqual(q[self.uo.id], 0 * ur.kg/ur.second)
        self.assertEqual(q[self.do.id], 0 * ur.kg/ur.second)

    def test_reaction_rate_cube(self):
        C = self.system.get_reaction_rate_3Darray(0*ur.second)
        self.assertEqual(C[self.uo.id, 0, 0], 0 * ur.kg/ur.second)
        self.assertEqual(C[self.do.id, 0, 0], 0 * ur.kg/ur.second)

if __name__ == "__main__": 
    unittest.main()




