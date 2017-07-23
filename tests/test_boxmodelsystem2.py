# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import unittest
from unittest import TestCase

import os
import math
import sys
import copy
import numpy as np
import datetime

from matplotlib import pyplot as plt

from pint import UnitRegistry
ur = UnitRegistry(autoconvert_offset_to_baseunit = True)

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
from boxsimu.simulations import boxmodelsystem2


class BoxModelSystem2Test(TestCase):
    """Test boxsimu framework using an intermediate complex box model."""

    def setUp(self, *args, **kwargs):
        self.system = boxmodelsystem2.get_system(ur)
        self.solver = Solver(self.system)
        self.la = self.system.boxes.lake
        self.uo = self.system.boxes.upper_ocean
        self.do = self.system.boxes.deep_ocean
        self.se = self.system.boxes.sediment

        self.po4 = self.system.variables.po4
        self.no3 = self.system.variables.no3
        self.phyto = self.system.variables.phyto

    def tearDown(self, *args, **kwargs):
       del(self.system)
       del(self.solver)
       del(self.la)
       del(self.uo)
       del(self.do)
       del(self.se)

       del(self.po4)
       del(self.no3)
       del(self.phyto)

    def assertPintQuantityAlmostEqual(self, q1, q2, rel_tol=1e-7):
        q1 = q1.to_base_units()
        q2 = q2.to_base_units()
        try:
            self.assertTrue(math.isclose(q1.magnitude, q2.magnitude, 
                rel_tol=rel_tol))
        except AssertionError:
            raise AssertionError(
                    '{} != {} with relative tolerance of {}'.format(
                            q1, q2, rel_tol 
                        )
            )
        self.assertEqual(q1.units, q2.units)

    #####################################################
    # Box Functions 
    #####################################################
    
    def test_mass(self):
        self.assertEqual(self.la.mass, 1e16*ur.kg + 6*ur.kg)
        self.assertEqual(self.uo.mass, 3e19*ur.kg + 15*ur.kg)
        self.assertEqual(self.do.mass, 1e21*ur.kg + 24*ur.kg)
        self.assertEqual(self.se.mass, 1e10*ur.kg + 33*ur.kg)

    def test_volume(self):
        la_context = self.system.get_box_context(self.la)
        uo_context = self.system.get_box_context(self.uo)
        do_context = self.system.get_box_context(self.do)
        se_context = self.system.get_box_context(self.se)

        self.assertEqual(self.la.get_volume(la_context), 1e16/1020 * ur.meter**3)
        self.assertEqual(self.uo.get_volume(uo_context), 3e19/1020 * ur.meter**3)
        self.assertEqual(self.do.get_volume(do_context), 1e21/1020 * ur.meter**3)
        self.assertEqual(self.se.get_volume(se_context), 1e10/2720 * ur.meter**3)

    def test_concentration(self):
        pass

    #####################################################
    # Base Functions 
    #####################################################
    
    def test_box_id(self):
        self.assertEqual(self.la.id, 1)
        self.assertEqual(self.uo.id, 3)
        self.assertEqual(self.do.id, 0)
        self.assertEqual(self.se.id, 2)

    def test_variable_id(self):
        self.assertEqual(self.no3.id, 0)
        self.assertEqual(self.phyto.id, 1)
        self.assertEqual(self.po4.id, 2)

    def test_N_boxes(self):
        self.assertEqual(self.system.N_boxes, 4)
    
    def test_N_variables(self):
        self.assertEqual(self.system.N_variables, 3)

    def test_context_of_box(self):
        global_context = self.system.get_box_context()
        lake_context = self.system.get_box_context(self.la)
        upper_ocean_context = self.system.get_box_context(self.uo)
        deep_ocean_context = self.system.get_box_context(self.do)
        sediment_context = self.system.get_box_context(self.se)
        
        # Test accessability of the condition attributes
        self.assertEqual(global_context.T, 288 * ur.kelvin)
        self.assertEqual(global_context.pH, 7.3)
        self.assertEqual(lake_context.T, 290 * ur.kelvin)
        self.assertEqual(lake_context.pH, 7.0)
        self.assertEqual(upper_ocean_context.T, 280 * ur.kelvin)
        self.assertEqual(upper_ocean_context.pH, 8.3)
        self.assertEqual(deep_ocean_context.T, 275 * ur.kelvin)
        self.assertEqual(deep_ocean_context.pH, 8.1)
        self.assertEqual(sediment_context.T, 275 * ur.kelvin)
        self.assertEqual(sediment_context.pH, 7.7)

        # Test the accessability of the condition attributes of other boxes:
        self.assertEqual(global_context.upper_ocean.condition.T, 280 * ur.kelvin)
        self.assertEqual(global_context.deep_ocean.condition.pH, 8.1)
        self.assertEqual(lake_context.upper_ocean.condition.T, 280 * ur.kelvin)
        self.assertEqual(lake_context.deep_ocean.condition.pH, 8.1)
        self.assertEqual(upper_ocean_context.lake.condition.T, 290 * ur.kelvin)
        self.assertEqual(upper_ocean_context.deep_ocean.condition.pH, 8.1)
        self.assertEqual(deep_ocean_context.upper_ocean.condition.T, 280 * ur.kelvin)
        self.assertEqual(deep_ocean_context.lake.condition.pH, 7.0)
        self.assertEqual(sediment_context.upper_ocean.condition.T, 280 * ur.kelvin)
        self.assertEqual(sediment_context.lake.condition.pH, 7.0)

    def test_context_evaluation_lambda_func(self):
        system_copy = copy.deepcopy(self.system)

        global_context = system_copy.get_box_context()
        lake_context = system_copy.get_box_context(self.la)
        upper_ocean_context = system_copy.get_box_context(self.uo)
        deep_ocean_context = system_copy.get_box_context(self.do)
        sediment_context = system_copy.get_box_context(self.se)

        lambda1 = lambda t, c: c.T / (100*ur.kelvin) + c.pH
        self.assertEqual(lambda1(0*ur.second, global_context), 2.88+7.3)
        self.assertEqual(lambda1(0*ur.second, lake_context), 2.90+7.0)
        self.assertEqual(lambda1(0*ur.second, upper_ocean_context), 2.80+8.3)
        self.assertEqual(lambda1(0*ur.second, deep_ocean_context), 2.75+8.1)
        self.assertEqual(lambda1(0*ur.second, sediment_context), 2.75+7.7)


    #####################################################
    # Fluid and Variable Mass/Concentration Vectors/Matrices
    #####################################################

    def test_fluid_mass_1Dlist_1Darray(self):
        m = self.system.get_fluid_mass_1Darray()
        self.assertEqual(m[self.la.id], 1e16 * ur.kg)
        self.assertEqual(m[self.uo.id], 3e19 * ur.kg)
        self.assertEqual(m[self.do.id], 1e21 * ur.kg)
        self.assertEqual(m[self.se.id], 1e10 * ur.kg)

    def test_variable_mass_1Darray(self):
        m = self.system.get_variable_mass_1Darray(self.po4)
        self.assertEqual(m[self.la.id], 1 * ur.kg)
        self.assertEqual(m[self.uo.id], 4 * ur.kg)
        self.assertEqual(m[self.do.id], 7 * ur.kg)
        self.assertEqual(m[self.se.id], 10 * ur.kg)

        m = self.system.get_variable_mass_1Darray(self.no3)
        self.assertEqual(m[self.la.id], 2 * ur.kg)
        self.assertEqual(m[self.uo.id], 5 * ur.kg)
        self.assertEqual(m[self.do.id], 8 * ur.kg)
        self.assertEqual(m[self.se.id], 11 * ur.kg)

        m = self.system.get_variable_mass_1Darray(self.phyto)
        self.assertEqual(m[self.la.id], 3 * ur.kg)
        self.assertEqual(m[self.uo.id], 6 * ur.kg)
        self.assertEqual(m[self.do.id], 9 * ur.kg)
        self.assertEqual(m[self.se.id], 12 * ur.kg)

    def test_variable_concentration_1Darray(self):
        def _c(var_mass, fluid_mass):
            return var_mass / (fluid_mass + var_mass) * ur.dimensionless

        c = self.system.get_variable_concentration_1Darray(self.po4)
        self.assertAlmostEqual(c[self.la.id], _c(3, 1e16))
        self.assertAlmostEqual(c[self.uo.id], _c(3.3123, 3e19))
        self.assertAlmostEqual(c[self.do.id], _c(3.492, 1e21))
        self.assertAlmostEqual(c[self.se.id], _c(2.3484, 1e10))

        c = self.system.get_variable_concentration_1Darray(self.no3)
        self.assertAlmostEqual(c[self.la.id], _c(1, 1e16))
        self.assertAlmostEqual(c[self.uo.id], _c(0.237, 3e19))
        self.assertAlmostEqual(c[self.do.id], _c(1.12437, 1e21))
        self.assertAlmostEqual(c[self.se.id], _c(9.23, 1e10))

        c = self.system.get_variable_concentration_1Darray(self.phyto)
        self.assertAlmostEqual(c[self.la.id], _c(0.324, 1e16))
        self.assertAlmostEqual(c[self.uo.id], _c(0.7429, 3e19))
        self.assertAlmostEqual(c[self.do.id], _c(4.324, 1e21))
        self.assertAlmostEqual(c[self.se.id], _c(2.824, 1e10))

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def test_fluid_mass_internal_flow_2Darray(self):
        A = self.system.get_fluid_mass_internal_flow_2Darray(0*ur.second)
        # Check that diagonal elements are zero
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i,i], 0 * ur.kg / ur.year)
        # Check that the other values are set correctly
        
        # Deep Ocean id=0 ; Lake id=1 ; Sediment id=2 ; Upper Ocean id=3
        self.assertEqual(A[self.do.id, self.la.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.do.id, self.se.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.do.id, self.uo.id], 6e17*ur.kg/ur.year)
        self.assertEqual(A[self.la.id, self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.la.id, self.se.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.la.id, self.uo.id], 2e15*ur.kg/ur.year)
        self.assertEqual(A[self.se.id, self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.se.id, self.la.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.se.id, self.uo.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.uo.id, self.do.id], 6e17*ur.kg/ur.year)
        self.assertEqual(A[self.uo.id, self.la.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.uo.id, self.se.id], 0*ur.kg/ur.year)

    def test_fluid_mass_flow_sink_1Darray(self):
        s = self.system.get_fluid_mass_flow_sink_1Darray(0*ur.second)
        self.assertEqual(s[self.la.id], 1e15*ur.kg/ur.year)
        self.assertEqual(s[self.uo.id], 2e15*ur.kg/ur.year)
        self.assertEqual(s[self.do.id], 1e11*ur.kg/ur.year)
        self.assertEqual(s[self.se.id], 0*ur.kg/ur.year)

    def test_fluid_mass_flow_source_1Darray(self):
        q = self.system.get_fluid_mass_flow_source_1Darray(0*ur.second)
        self.assertEqual(q[self.la.id], 3e15*ur.kg/ur.year)
        self.assertEqual(q[self.uo.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.do.id], 1e11*ur.kg/ur.year)
        self.assertEqual(q[self.se.id], 0*ur.kg/ur.year)

    #####################################################
    # Variable Sink/Source Vectors
    #####################################################

    def test_variable_internal_flow_2Darray(self):
        f_flow = np.ones(self.system.N_boxes)
        flow_la_uo = 2e15*ur.kg/ur.year
        flow_uo_do = 6e17*ur.kg/ur.year
        flow_do_uo = 6e17*ur.kg/ur.year

        for var in [self.po4, self.no3, self.phyto]:
            A = self.system.get_variable_internal_flow_2Darray(
                    var, 0*ur.second, f_flow)

            # Check that diagonal elements are zero
            for i in range(self.system.N_boxes):
                self.assertEqual(A[i,i], 0*ur.kg/ur.year)

            # Deep Ocean id=0 ; Lake id=1 ; Sediment id=2 ; Upper Ocean id=3
            c = self.system.get_variable_concentration_1Darray(var)
            self.assertEqual(A[self.do.id, self.la.id], 0*ur.kg/ur.year)
            self.assertEqual(A[self.do.id, self.se.id], 0*ur.kg/ur.year)
            self.assertPintQuantityAlmostEqual(A[self.do.id, self.uo.id], 
                    flow_do_uo * c[self.do.id])
            self.assertEqual(A[self.la.id, self.do.id], 0*ur.kg/ur.year)
            self.assertEqual(A[self.la.id, self.se.id], 0*ur.kg/ur.year)
            self.assertPintQuantityAlmostEqual(A[self.la.id, self.uo.id], 
                    flow_la_uo * c[self.la.id])
            self.assertEqual(A[self.se.id, self.do.id], 0*ur.kg/ur.year)
            self.assertEqual(A[self.se.id, self.la.id], 0*ur.kg/ur.year)
            self.assertEqual(A[self.se.id, self.uo.id], 0*ur.kg/ur.year)
            self.assertPintQuantityAlmostEqual(A[self.uo.id, self.do.id], 
                    flow_uo_do * c[self.uo.id])
            self.assertEqual(A[self.uo.id, self.la.id], 0*ur.kg/ur.year)
            self.assertEqual(A[self.uo.id, self.se.id], 0*ur.kg/ur.year)

    def test_variable_flow_sink_1Darray(self):
        f_flow = np.ones(self.system.N_boxes)
        flow_do_none = 1e11*ur.kg/ur.year

        for var in [self.po4, self.no3, self.phyto]:
            s = self.system.get_variable_flow_sink_1Darray(var, 0*ur.second, 
                    f_flow)
            c = self.system.get_variable_concentration_1Darray(var)
            # Lake Evaporation does not transport tracers!
            self.assertEqual(s[self.la.id], 0*ur.kg/ur.year)   
            # Upper Ocean Evaporation does not transport tracers!
            self.assertEqual(s[self.uo.id], 0*ur.kg/ur.year)
            self.assertPintQuantityAlmostEqual(s[self.do.id], 
                    flow_do_none * c[self.do.id])
            self.assertEqual(s[self.se.id], 0*ur.kg/ur.year)

    def test_variable_flow_source_1Darray(self):
        la_input_concentration = {self.po4: 4.6455e-8*ur.kg/ur.kg,
                                  self.no3: 7*4.6455e-8*ur.kg/ur.kg}
        do_input_concentration = la_input_concentration

        for var in [self.po4, self.no3, self.phyto]:
            q = self.system.get_variable_flow_source_1Darray(var, 0*ur.second)
            la_input_c = la_input_concentration.get(var, 0*ur.kg/ur.kg)
            la_inflow = 3e15*ur.kg/ur.year

            do_input_c = do_input_concentration.get(var, 0*ur.kg/ur.kg)
            do_inflow = 1e11*ur.kg/ur.year

            self.assertPintQuantityAlmostEqual(q[self.la.id], 
                    la_input_c * la_inflow)
            self.assertEqual(q[self.uo.id], 0*ur.kg/ur.year)
            self.assertPintQuantityAlmostEqual(q[self.do.id], 
                    do_input_c * do_inflow) 
            self.assertEqual(q[self.se.id], 0*ur.kg/ur.year)

    def test_variable_process_sink_1Darray(self):
        s = self.system.get_variable_process_sink_1Darray(
                self.po4, 0*ur.second)
        m = self.system.get_variable_mass_1Darray(self.po4)
        self.assertEqual(s[self.la.id], 0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(s[self.uo.id], 
                m[self.uo.id]*0.01/ur.year)
        self.assertEqual(s[self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.se.id], 0*ur.kg/ur.year)

        s = self.system.get_variable_process_sink_1Darray(
                self.no3, 0*ur.second)
        m = self.system.get_variable_mass_1Darray(self.no3)
        self.assertEqual(s[self.la.id], 0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(s[self.uo.id], 
                m[self.uo.id]*0.01/ur.year)
        self.assertEqual(s[self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.se.id], 0*ur.kg/ur.year)

        s = self.system.get_variable_process_sink_1Darray(
                self.phyto, 0*ur.second)
        m = self.system.get_variable_mass_1Darray(self.phyto)
        self.assertEqual(s[self.la.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.uo.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.se.id], 0*ur.kg/ur.year)


    def test_variable_process_source_1Darray(self):
        q = self.system.get_variable_process_source_1Darray(
                self.po4, 0*ur.second)
        self.assertEqual(q[self.la.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.uo.id], 12345 * ur.kg / ur.year)
        self.assertEqual(q[self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.se.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_process_source_1Darray(
                self.no3, 0*ur.second)
        self.assertEqual(q[self.la.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.uo.id], 123456 * ur.kg / ur.year)
        self.assertEqual(q[self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.se.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_process_source_1Darray(
                self.phyto, 0*ur.second)
        self.assertEqual(q[self.la.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.uo.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.do.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.se.id], 0*ur.kg/ur.year)

    def test_variable_internal_flux_2Darray(self):
        A = self.system.get_variable_internal_flux_2Darray(
                self.po4, 0*ur.second)
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i,i], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.do.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.do.id, self.se.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.do.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.se.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.se.id], 0 * ur.kg / ur.year)

        A = self.system.get_variable_internal_flux_2Darray(
                self.no3, 0*ur.second)
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i,i], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.do.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.do.id, self.se.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.do.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.se.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.se.id], 0 * ur.kg / ur.year)

        A = self.system.get_variable_internal_flux_2Darray(
                self.phyto, 0*ur.second)
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i,i], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.do.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertPintQuantityAlmostEqual(A[self.do.id, self.se.id], 
                self.uo.variables.phyto.mass * 0.01 / ur.year)
        self.assertEqual(A[self.do.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.se.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.la.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.se.id, self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.do.id], 
                self.uo.variables.phyto.mass * 0.1 / ur.year)
        self.assertEqual(A[self.uo.id, self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(A[self.uo.id, self.se.id], 0 * ur.kg / ur.year)

    def test_variable_flux_sink_1Darray(self):
        s = self.system.get_variable_flux_sink_1Darray(
                self.po4, 0*ur.second)
        self.assertEqual(s[self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.se.id], 0 * ur.kg / ur.year)

        s = self.system.get_variable_flux_sink_1Darray(
                self.no3, 0*ur.second)
        self.assertEqual(s[self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.se.id], 0 * ur.kg / ur.year)

        s = self.system.get_variable_flux_sink_1Darray(
                self.phyto, 0*ur.second)
        self.assertEqual(s[self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.se.id], 
                self.se.variables.phyto.mass * 0.1 / ur.year)

    def test_variable_flux_source_1Darray(self):
        q = self.system.get_variable_flux_source_1Darray(
                self.po4, 0*ur.second)
        self.assertEqual(q[self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.uo.id], 1e5 * ur.kg / ur.year)
        self.assertEqual(q[self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.se.id], 0 * ur.kg / ur.year)

        q = self.system.get_variable_flux_source_1Darray(
                self.no3, 0*ur.second)
        self.assertEqual(q[self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.uo.id], 2e4 * ur.kg / ur.year)
        self.assertEqual(q[self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.se.id], 0 * ur.kg / ur.year)

        q = self.system.get_variable_flux_source_1Darray(
                self.phyto, 0*ur.second)
        self.assertEqual(q[self.la.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.uo.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.do.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.se.id], 0 * ur.kg / ur.year)

    def test_reaction_rate_cube(self):
        C = self.system.get_reaction_rate_3Darray(
                0*ur.second)
        m_no3 = self.system.get_variable_mass_1Darray(self.no3)
        m_phyto = self.system.get_variable_mass_1Darray(self.phyto)

        rr_photosynthesis_la = 0.8 * m_no3[self.la.id] / (7.0 * ur.year)
        rr_photosynthesis_uo = 0.8 * m_no3[self.uo.id] / (7.0 * ur.year)
        rr_remineralization_la = 0.4 * m_phyto[self.la.id] / (114 * ur.year)
        rr_remineralization_uo = 0.4 * m_phyto[self.uo.id] / (114 * ur.year)
        rr_remineralization_do = 0.4 * m_phyto[self.do.id] / (114 * ur.year)
        
        photo_id = 0
        remin_id = 1

        # Lake photosynthesis
        self.assertPintQuantityAlmostEqual(C[self.la.id, self.po4.id, photo_id], 
                -rr_photosynthesis_la * 1)
        self.assertPintQuantityAlmostEqual(C[self.la.id, self.no3.id, photo_id], 
                -rr_photosynthesis_la * 7)
        self.assertPintQuantityAlmostEqual(C[self.la.id, self.phyto.id, photo_id], 
                rr_photosynthesis_la * 114)

        # Upper Ocean photosynthesis
        self.assertPintQuantityAlmostEqual(C[self.uo.id, self.po4.id, photo_id], 
                -rr_photosynthesis_uo * 1)
        self.assertPintQuantityAlmostEqual(C[self.uo.id, self.no3.id, photo_id], 
                -rr_photosynthesis_uo * 7)
        self.assertPintQuantityAlmostEqual(C[self.uo.id, self.phyto.id, photo_id], 
                rr_photosynthesis_uo * 114)

        # Lake remineralization
        self.assertPintQuantityAlmostEqual(C[self.la.id, self.po4.id, remin_id], 
                rr_remineralization_la * 1)
        self.assertPintQuantityAlmostEqual(C[self.la.id, self.no3.id, remin_id], 
                rr_remineralization_la * 7)
        self.assertPintQuantityAlmostEqual(C[self.la.id, self.phyto.id, remin_id], 
                -rr_remineralization_la * 114)

        # Upper Ocean remineralization
        self.assertPintQuantityAlmostEqual(C[self.uo.id, self.po4.id, remin_id], 
                rr_remineralization_uo * 1)
        self.assertPintQuantityAlmostEqual(C[self.uo.id, self.no3.id, remin_id], 
                rr_remineralization_uo * 7)
        self.assertPintQuantityAlmostEqual(C[self.uo.id, self.phyto.id, remin_id], 
                -rr_remineralization_uo * 114)

        self.assertPintQuantityAlmostEqual(C[self.do.id, self.po4.id, remin_id], 
                rr_remineralization_do * 1)
        self.assertPintQuantityAlmostEqual(C[self.do.id, self.no3.id, remin_id], 
                rr_remineralization_do * 7)
        self.assertPintQuantityAlmostEqual(C[self.do.id, self.phyto.id, remin_id], 
                -rr_remineralization_do * 114)

        # Rest of the Reaction Rate Cube has to be equal to zero
        self.assertEqual(C[self.do.id, self.po4.id, photo_id], 0 * ur.kg / ur.year)
        self.assertEqual(C[self.do.id, self.no3.id, photo_id], 0 * ur.kg / ur.year)
        self.assertEqual(C[self.do.id, self.phyto.id, photo_id], 0 * ur.kg / ur.year)

        self.assertEqual(C[self.se.id, self.po4.id, photo_id], 0 * ur.kg / ur.year)
        self.assertEqual(C[self.se.id, self.no3.id, photo_id], 0 * ur.kg / ur.year)
        self.assertEqual(C[self.se.id, self.phyto.id, photo_id], 0 * ur.kg / ur.year)

        self.assertEqual(C[self.se.id, self.po4.id, remin_id], 0 * ur.kg / ur.year)
        self.assertEqual(C[self.se.id, self.no3.id, remin_id], 0 * ur.kg / ur.year)
        self.assertEqual(C[self.se.id, self.phyto.id, remin_id], 0 * ur.kg / ur.year)

if __name__ == "__main__": 
    unittest.main()




