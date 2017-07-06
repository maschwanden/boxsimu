# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: aschi
"""
import unittest
from unittest import TestCase

import sys
import copy
import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot as plt

from pint import UnitRegistry
ur = UnitRegistry(autoconvert_offset_to_baseunit = True)

BOXSIMU_PATH = '/home/aschi/Documents/MyPrivateRepo/notebooks/boxsimu_project'
if not BOXSIMU_PATH in sys.path:
    sys.path.append(BOXSIMU_PATH)

from boxsimu.entities import Fluid, Variable
from boxsimu.box import Box
from boxsimu.transport import  Flow, Flux
from boxsimu.condition import Condition
from boxsimu.system import BoxModelSystem 
from boxsimu.process import Process, Reaction
from boxsimu.solver import Solver
from boxsimu import utils
from boxsimu_simulations import boxmodelsystem2


class BoxModelSystem1Test(TestCase):
    """ Tests the boxsimu framework using a simple box model described in the 
    book Modelling Methods for Marine Science.
    """
    def __init__(self, *args, **kwargs):
        # self.system = boxmodelsystem1.init_system(ur)
        # self.solver = Solver(self.system)
        # self.uo_id = self.system.boxes.upper_ocean.ID 
        # self.do_id = self.system.boxes.deep_ocean.ID

        super(BoxModelSystem1Test, self).__init__(*args, **kwargs)

    def setUp(self, *args, **kwargs):
        self.system = boxmodelsystem2.init_system(ur)
        self.solver = Solver(self.system)
        self.la_id = self.system.boxes.lake.ID
        self.uo_id = self.system.boxes.upper_ocean.ID 
        self.do_id = self.system.boxes.deep_ocean.ID
        self.se_id = self.system.boxes.sediment.ID

    def tearDown(self, *args, **kwargs):
       del(self.system)
       del(self.solver)
       del(self.uo_id)
       del(self.do_id)
        

    #####################################################
    # Base Functions 
    #####################################################
    
    def test_box_id(self):
        self.assertEqual(self.system.boxes.lake.ID, 1)
        self.assertEqual(self.system.boxes.upper_ocean.ID, 3)
        self.assertEqual(self.system.boxes.deep_ocean.ID, 0)
        self.assertEqual(self.system.boxes.sediment.ID, 2)

    def test_variable_id(self):
        self.assertEqual(self.system.variables.no3.ID, 0)
        self.assertEqual(self.system.variables.phyto.ID, 1)
        self.assertEqual(self.system.variables.po4.ID, 2)

    def test_N_boxes(self):
        self.assertEqual(self.system.N_boxes, 4)
    
    def test_N_variables(self):
        self.assertEqual(self.system.N_variables, 3)

    def test_context_of_box(self):
        lake = self.system.boxes.lake
        upper_ocean = self.system.boxes.upper_ocean
        deep_ocean = self.system.boxes.deep_ocean
        sediment = self.system.boxes.sediment

        global_context = self.system.get_box_context()
        lake_context = self.system.get_box_context(lake)
        upper_ocean_context = self.system.get_box_context(upper_ocean)
        deep_ocean_context = self.system.get_box_context(deep_ocean)
        sediment_context = self.system.get_box_context(sediment)
        
        # Test accessability of the condition attributes
        self.assertEqual(global_context.T.magnitude, 288)
        self.assertEqual(global_context.pH, 7.3)
        self.assertEqual(lake_context.T.magnitude, 290)
        self.assertEqual(lake_context.pH, 7.0)
        self.assertEqual(upper_ocean_context.T.magnitude, 280)
        self.assertEqual(upper_ocean_context.pH, 8.3)
        self.assertEqual(deep_ocean_context.T.magnitude, 275)
        self.assertEqual(deep_ocean_context.pH, 8.1)
        self.assertEqual(sediment_context.T.magnitude, 275)
        self.assertEqual(sediment_context.pH, 7.7)

        # Test the accessability of the condition attributes of other boxes:
        self.assertEqual(global_context.upper_ocean_cond.T.magnitude, 280)
        self.assertEqual(global_context.deep_ocean_cond.pH, 8.1)
        self.assertEqual(lake_context.upper_ocean_cond.T.magnitude, 280)
        self.assertEqual(lake_context.deep_ocean_cond.pH, 8.1)
        self.assertEqual(upper_ocean_context.lake_cond.T.magnitude, 290)
        self.assertEqual(upper_ocean_context.deep_ocean_cond.pH, 8.1)
        self.assertEqual(deep_ocean_context.upper_ocean_cond.T.magnitude, 280)
        self.assertEqual(deep_ocean_context.lake_cond.pH, 7.0)
        self.assertEqual(sediment_context.upper_ocean_cond.T.magnitude, 280)
        self.assertEqual(sediment_context.lake_cond.pH, 7.0)

    def test_context_evaluation_lambda_func(self):
        system_copy = copy.deepcopy(self.system)

        lake = system_copy.boxes.lake
        upper_ocean = system_copy.boxes.upper_ocean
        deep_ocean = system_copy.boxes.deep_ocean
        sediment = system_copy.boxes.sediment

        global_context = system_copy.get_box_context()
        lake_context = system_copy.get_box_context(lake)
        upper_ocean_context = system_copy.get_box_context(upper_ocean)
        deep_ocean_context = system_copy.get_box_context(deep_ocean)
        sediment_context = system_copy.get_box_context(sediment)

        lambda1 = lambda t, c: c.T / (100*ur.kelvin) + c.pH
        self.assertEqual(lambda1(0*ur.second, global_context), 2.88+7.3)
        self.assertEqual(lambda1(0*ur.second, lake_context), 2.90+7.0)
        self.assertEqual(lambda1(0*ur.second, upper_ocean_context), 2.80+8.3)
        self.assertEqual(lambda1(0*ur.second, deep_ocean_context), 2.75+8.1)
        self.assertEqual(lambda1(0*ur.second, sediment_context), 2.75+7.7)


    #####################################################
    # Fluid and Variable Mass/Concentration Vectors/Matrices
    #####################################################

    def test_fluid_mass_1Darray(self):
        m = self.system.get_fluid_mass_1Darray()
        self.assertEqual(m[self.la_id], 1e16)
        self.assertEqual(m[self.uo_id], 3e19)
        self.assertEqual(m[self.do_id], 1e21)
        self.assertEqual(m[self.se_id], 1e10)

    def test_variable_mass_1Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
        phyto = self.system.variables['phyto']

        m = self.system.get_variable_mass_1Darray(po4)
        self.assertEqual(m[self.la_id], 3)
        self.assertEqual(m[self.uo_id], 3.3123)
        self.assertEqual(m[self.do_id], 3.492)
        self.assertEqual(m[self.se_id], 2.3484)

        m = self.system.get_variable_mass_1Darray(no3)
        self.assertEqual(m[self.la_id], 1)
        self.assertEqual(m[self.uo_id], 0.237)
        self.assertEqual(m[self.do_id], 1.12437)
        self.assertEqual(m[self.se_id], 9.23)

        m = self.system.get_variable_mass_1Darray(phyto)
        self.assertEqual(m[self.la_id], 0.324)
        self.assertEqual(m[self.uo_id], 0.7429)
        self.assertEqual(m[self.do_id], 4.324)
        self.assertEqual(m[self.se_id], 2.824)

    def test_variable_concentration_1Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
        phyto = self.system.variables['phyto']

        def _c(var_mass, fluid_mass):
            return var_mass / (fluid_mass + var_mass)

        c = self.system.get_variable_concentration_1Darray(po4)
        self.assertAlmostEqual(c[self.la_id], _c(3, 1e16))
        self.assertAlmostEqual(c[self.uo_id], _c(3.3123, 3e19))
        self.assertAlmostEqual(c[self.do_id], _c(3.492, 1e21))
        self.assertAlmostEqual(c[self.se_id], _c(2.3484, 1e10))

        c = self.system.get_variable_concentration_1Darray(no3)
        self.assertAlmostEqual(c[self.la_id], _c(1, 1e16))
        self.assertAlmostEqual(c[self.uo_id], _c(0.237, 3e19))
        self.assertAlmostEqual(c[self.do_id], _c(1.12437, 1e21))
        self.assertAlmostEqual(c[self.se_id], _c(9.23, 1e10))

        c = self.system.get_variable_concentration_1Darray(phyto)
        self.assertAlmostEqual(c[self.la_id], _c(0.324, 1e16))
        self.assertAlmostEqual(c[self.uo_id], _c(0.7429, 3e19))
        self.assertAlmostEqual(c[self.do_id], _c(4.324, 1e21))
        self.assertAlmostEqual(c[self.se_id], _c(2.824, 1e10))

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def test_fluid_mass_flow_2Darray(self):
        A = self.system.get_fluid_mass_internal_flow_2Darray(0*ur.second)
        # Check that diagonal elements are zero
        for i in range(self.system.N_boxes):
            self.assertEqual(A[i,i], 0)
        # Check that the other values are set correctly
        self.assertEqual(A[self.la_id, self.uo_id], (2e15*ur.kg/ur.year).to_base_units().magnitude)
        self.assertEqual(A[self.uo_id, self.do_id], (6e17*ur.kg/ur.year).to_base_units().magnitude)
        self.assertEqual(A[self.do_id, self.uo_id], (6e17*ur.kg/ur.year).to_base_units().magnitude)

    def test_fluid_mass_flow_sink_1Darray(self):
        s = self.system.get_fluid_mass_flow_sink_1Darray(0*ur.second)
        self.assertEqual(s[self.la_id], (1e15*ur.kg/ur.year).to_base_units().magnitude)
        self.assertEqual(s[self.uo_id], (2e15*ur.kg/ur.year).to_base_units().magnitude)
        self.assertEqual(s[self.do_id], (1e11*ur.kg/ur.year).to_base_units().magnitude)

    def test_fluid_mass_flow_source_1Darray(self):
        q = self.system.get_fluid_mass_flow_source_1Darray(0*ur.second)
        self.assertEqual(q[self.la_id], (3e15*ur.kg/ur.year).to_base_units().magnitude)
        self.assertEqual(q[self.do_id], (1e11*ur.kg/ur.year).to_base_units().magnitude)

    #####################################################
    # Variable Sink/Source Vectors
    #####################################################

    def test_variable_internal_flow_2Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
        phyto = self.system.variables['phyto']

        f_flow = np.ones(self.system.N_boxes)

        flow_la_uo = (2e15*ur.kg/ur.year).to_base_units().magnitude
        flow_uo_do = (6e17*ur.kg/ur.year).to_base_units().magnitude
        flow_do_uo = (6e17*ur.kg/ur.year).to_base_units().magnitude

        for var in [po4, no3, phyto]:
            A = self.system.get_variable_internal_flow_2Darray(var, 0*ur.second, f_flow)

            # Check that diagonal elements are zero
            for i in range(self.system.N_boxes):
                self.assertEqual(A[i,i], 0)

            c = self.system.get_variable_concentration_1Darray(var)
            self.assertAlmostEqual(A[self.la_id, self.uo_id], flow_la_uo * c[self.la_id], places=4)
            self.assertAlmostEqual(A[self.uo_id, self.do_id], flow_uo_do * c[self.uo_id], places=4)
            self.assertAlmostEqual(A[self.do_id, self.uo_id], flow_do_uo * c[self.do_id], places=4)

    def test_variable_flow_sink_1Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
        phyto = self.system.variables['phyto']
        
        f_flow = np.ones(self.system.N_boxes)
        
        flow_do_none = (1e11*ur.kg/ur.year).to_base_units().magnitude

        for var in [po4, no3, phyto]:
            s = self.system.get_variable_flow_sink_1Darray(var, 0*ur.second, f_flow)

            c = self.system.get_variable_concentration_1Darray(var)
            self.assertEqual(s[self.la_id], 0)  # Lake Evaporation does not transport tracers!
            self.assertEqual(s[self.uo_id], 0)  # Upper Ocean Evaporation does not transport tracers!

            self.assertEqual(s[self.do_id], flow_do_none * c[self.do_id])

    def test_variable_flow_sink_2Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
        phyto = self.system.variables['phyto']
        f_flow = np.ones(self.system.N_boxes)
        S = self.system.get_all_variable_flow_sink_2Darray(0*ur.second, f_flow)
        flow_do_none = (1e11*ur.kg/ur.year).to_base_units().magnitude
        
        for var in [po4, no3, phyto]:
            c = self.system.get_variable_concentration_1Darray(var)
            self.assertEqual(S[self.la_id, var.ID], 0)  # Lake Evaporation does not transport tracers!
            self.assertEqual(S[self.uo_id, var.ID], 0)  # Upper Ocean Evaporation does not transport tracers!
            self.assertEqual(S[self.do_id, var.ID], flow_do_none * c[self.do_id])

    def test_variable_flow_source_1Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
        phyto = self.system.variables['phyto']

        la_input_concentration = {po4: (4.6455e-8*ur.kg/ur.kg).to_base_units().magnitude,
                                  no3: (7*4.6455e-8*ur.kg/ur.kg).to_base_units().magnitude}
        do_input_concentration = la_input_concentration

        for var in [po4, no3, phyto]:
            q = self.system.get_variable_flow_source_1Darray(var, 0*ur.second)
            la_input_c = la_input_concentration.get(var, 0)
            la_inflow = (3e15*ur.kg/ur.year).to_base_units().magnitude

            do_input_c = do_input_concentration.get(var, 0)
            do_inflow = (1e11*ur.kg/ur.year).to_base_units().magnitude

            self.assertEqual(q[self.la_id], la_input_c * la_inflow)
            self.assertEqual(q[self.uo_id], 0)
            self.assertEqual(q[self.do_id], do_input_c * do_inflow) 
            self.assertEqual(q[self.se_id], 0)

    def test_variable_flow_source_2Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
        phyto = self.system.variables['phyto']

        la_input_concentration = {po4: (4.6455e-8*ur.kg/ur.kg).to_base_units().magnitude,
                                  no3: (7*4.6455e-8*ur.kg/ur.kg).to_base_units().magnitude}
        do_input_concentration = la_input_concentration
        
        Q = self.system.get_all_variable_flow_source_2Darray(0*ur.second)

        for var in [po4, no3, phyto]:
            la_input_c = la_input_concentration.get(var, 0)
            la_inflow = (3e15*ur.kg/ur.year).to_base_units().magnitude

            do_input_c = do_input_concentration.get(var, 0)
            do_inflow = (1e11*ur.kg/ur.year).to_base_units().magnitude

            self.assertEqual(Q[self.la_id, var.ID], la_input_c * la_inflow)
            self.assertEqual(Q[self.uo_id, var.ID], 0)
            self.assertEqual(Q[self.do_id, var.ID], do_input_c * do_inflow) 
            self.assertEqual(Q[self.se_id, var.ID], 0)



    def test_variable_process_sink_1Darray(self):
        po4 = self.system.variables['po4']
        no3 = self.system.variables['no3']
         
        s = self.system.get_variable_process_sink_1Darray(po4, 1*ur.second)
        m = self.system.get_variable_mass_1Darray(po4)
        self.assertListEqual(s[self.la_id], 0)
        self.assertListEqual(s[self.uo_id], m[self.uo_id]*0.01)
        self.assertListEqual(s[self.do_id], 0)
        self.assertListEqual(s[self.se_id], 0)

        s = self.system.get_variable_process_sink_1Darray(po4, 1*ur.second)
        self.assertListEqual(s.tolist(), [0, 0])

    def test_variable_process_sink_2Darray(self):
        S = self.system.get_all_variable_process_sink_2Darray(0*ur.second)
        self.assertEqual(S[self.uo_id, 0], 0)
        self.assertEqual(S[self.do_id, 0], 0)

#     def test_variable_process_source_1Darray(self):
#         var = self.system.variables['po4']
#         q = self.system.get_variable_process_source_1Darray(var, 0*ur.second)
#         self.assertListEqual(q.tolist(), [0, 0])
# 
#     def test_variable_process_source_2Darray(self):
#         Q = self.system.get_all_variable_process_source_2Darray(0*ur.second)
#         self.assertEqual(Q[self.uo_id, 0], 0)
#         self.assertEqual(Q[self.do_id, 0], 0)
# 
#     def test_variable_internal_flux_2Darray(self):
#         A = self.system.get_variable_internal_flux_2Darray(0*ur.second)
#         self.assertEqual(A[self.uo_id, self.do_id], 0)
#         self.assertEqual(A[self.do_id, self.uo_id], 0)
# 
#     def test_variable_flux_sink_1Darray(self):
#         var = self.system.variables['po4']
#         s = self.system.get_variable_flux_sink_1Darray(var, 1*ur.second)
#         self.assertListEqual(s.tolist(), [0, 0])
# 
#     def test_variable_flux_sink_2Darray(self):
#         S = self.system.get_all_variable_flux_sink_2Darray(0*ur.second)
#         self.assertEqual(S[self.uo_id, 0], 0)
#         self.assertEqual(S[self.do_id, 0], 0)
# 
#     def test_variable_flux_source_1Darray(self):
#         var = self.system.variables['po4']
#         q = self.system.get_variable_flux_source_1Darray(var, 0*ur.second)
#         self.assertListEqual(q.tolist(), [0, 0])
# 
#     def test_variable_flux_source_2Darray(self):
#         Q = self.system.get_all_variable_flux_source_2Darray(0*ur.second)
#         self.assertEqual(Q[self.uo_id, 0], 0)
#         self.assertEqual(Q[self.do_id, 0], 0)
# 
#     def test_reaction_rate_cube(self):
#         rr_cube = self.system.get_reaction_rate_3Darray(0*ur.second)
#         self.assertEqual(rr_cube[self.uo_id,0,0], 0)
#         self.assertEqual(rr_cube[self.do_id,0,0], 0)

if __name__ == "__main__": 
    unittest.main()




