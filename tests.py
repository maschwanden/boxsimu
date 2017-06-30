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


BOXSIMU_PATH = '/home/aschi/Documents/MyPrivateRepo/notebooks/'
if not BOXSIMU_PATH in sys.path:
    sys.path.append(BOXSIMU_PATH)

from boxsimu import (Fluid, Variable, Box, Flow, Condition, 
                     BoxModelSystem, Process, Reaction, Flux, Solver)
from boxsimu import utils



class BoxModelSystem1Test(TestCase):
    """ Tests the boxsimu framework using a simple box model described in the 
    book Modelling Methods for Marine Science.
    """
    def __init__(self, *args, **kwargs):
        self.system = self.init_system()
        self.solver = Solver(self.system)
        self.uo_id = self.system.boxes.upper_ocean.ID 
        self.do_id = self.system.boxes.deep_ocean.ID

        super(BoxModelSystem1Test, self).__init__(*args, **kwargs)
        
    def init_system(self):
        V1 = 3e16
        V2 = 1e18
        FR = 3e13
        FO = 6e14
        
        M1 = V1*1020
        M2 = V2*1030
        FRM = 3e13*1020
        FOM = FO* 1020

        #############################
        # FLUIDS
        #############################
        seawater = Fluid('sea water', rho_expr=1000*ur.kg/ur.meter**3)
        
        #############################
        # VARIABLES
        #############################
        phosphate = Variable('PO4')
        
        #############################
        # BOXES
        #############################
        upper_ocean = Box(
        name='upper_ocean',
            name_long='Upper Ocean Box',
            fluid=seawater.q(M1*ur.kg), 
        )
        deep_ocean = Box(
            name='deep_ocean',
            name_long='Deep Ocean Box', 
            fluid=seawater.q(M2*ur.kg),
        )
        
        #############################
        # FLOWS
        #############################
    
        flow_downwelling = Flow(
            name='Downwelling', 
            source_box=upper_ocean, 
            target_box=deep_ocean,
            rate=6e17*ur.kg/ur.year,
        )
        
        flow_upwelling = Flow(
            name='Upwelling',
            source_box=deep_ocean, 
            target_box=upper_ocean,
            rate=6e17*ur.kg/ur.year, 
        )
        
        flow_river_water = Flow(
            name='River Inflow into Upper Ocean',
            source_box=None, 
            target_box=upper_ocean,
            rate=3e16*ur.kg/ur.year, 
        )
        flow_river_water.add_transported_variable(variable=phosphate.q(0*ur.kg), concentration=4.6455e-8*ur.kg/ur.kg)
        
        flow_upper_ocean_evaporation = Flow(
            name='Upper Ocean Evaporation',
            source_box=upper_ocean,
            target_box=None,
            rate=3e16*ur.kg/ur.year,
        )
        flow_upper_ocean_evaporation.transports_tracers = False
        
        #############################
        # SYSTEM
        #############################
        system = BoxModelSystem('Test System', 
                              [upper_ocean, deep_ocean], 
                              flows=[flow_downwelling, flow_upwelling, 
                                     flow_river_water, flow_upper_ocean_evaporation]
        )
        return system


    #####################################################
    # Base Functions 
    #####################################################

    def test_N_boxes(self):
        self.assertEqual(self.system.N_boxes, 2)
    
    def test_N_variables(self):
        self.assertEqual(self.system.N_variables, 1)

    #####################################################
    # Fluid and Variable Mass/Concentration Vectors/Matrices
    #####################################################

    def test_fluid_mass_vector(self):
        m = self.system.get_fluid_mass_vector()
        self.assertEqual(m[self.uo_id], 3e16*1020)
        self.assertEqual(m[self.do_id], 1e18*1030)

    def test_variable_mass_vector(self):
        var = self.system.variables['PO4']
        m = self.system.get_variable_mass_vector(var)
        self.assertEqual(m[self.uo_id], 0)
        self.assertEqual(m[self.do_id], 0)

    def test_variable_concentration_vector(self):
        var = self.system.variables['PO4']
        c = self.system.get_variable_concentration_vector(var)
        self.assertEqual(c[self.uo_id], 0)
        self.assertEqual(c[self.do_id], 0)

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def test_fluid_mass_flow_matrix(self):
        A = self.system.get_fluid_mass_flow_matrix(0*ur.second, self.system.flows)
        # Check that diagonal elements are zero
        for i in range(self.system.N_boxes):
            self.assertAlmostEqual(A[i,i], 0)
        # Check that the other values are set correctly
        uo_do_exchange_rate = (6e17*ur.kg/ur.year).to_base_units().magnitude
        self.assertEqual(A[self.uo_id, self.do_id], uo_do_exchange_rate)
        self.assertEqual(A[self.do_id, self.uo_id], uo_do_exchange_rate)

    def test_fluid_mass_flow_sink_vector(self):
        s = self.system.get_fluid_mass_flow_sink_vector(0*ur.second, self.system.flows)
        # Upper Ocean Sink: Due to evaporation (3e16)
        evaporation_rate = (3e16*ur.kg/ur.year).to_base_units().magnitude
        self.assertEqual(s[self.uo_id], evaporation_rate)
        self.assertEqual(s[self.do_id], 0)

    def test_fluid_mass_flow_source_vector(self):
        q = self.system.get_fluid_mass_flow_source_vector(0*ur.second, self.system.flows)
        # Upper Ocean Source: Due to river discharge (3e16)
        river_discharge_rate = (3e16*ur.kg/ur.year).to_base_units().magnitude
        self.assertEqual(q[self.uo_id], river_discharge_rate)
        self.assertEqual(q[self.do_id], 0)

    #####################################################
    # Variable Sink/Source Vectors
    #####################################################

    def test_variable_flow_sink_vector(self):
        var = self.system.variables['PO4']
        
        f_flow = np.ones(self.system.N_boxes)
        s = self.system.get_variable_flow_sink_vector(var, 0*ur.second, f_flow)
        self.assertEqual(s[self.uo_id], 0)
        self.assertEqual(s[self.do_id], 0)

    def test_variable_flow_sink_matrix(self):
        var = self.system.variables['PO4']

        f_flow = np.ones(self.system.N_boxes)
        S = self.system.get_variable_flow_sink_matrix(0*ur.second, f_flow)
        self.assertEqual(S[self.uo_id, 0], 0)
        self.assertEqual(S[self.do_id, 0], 0)

    def test_variable_flow_source_vector(self):
        var = self.system.variables['PO4']
        q = self.system.get_variable_flow_source_vector(var, 0*ur.second)
        river_discharge_rate = (3e16*ur.kg/ur.year).to_base_units().magnitude
        # Upper Ocean Source: 3e16 * 4.6455e-8 = 1393650000.0
        self.assertAlmostEqual(q[self.uo_id], river_discharge_rate*4.6455e-8)
        self.assertEqual(q[self.do_id], 0)

    def test_variable_flow_source_matrix(self):
        Q = self.system.get_variable_flow_source_matrix(0*ur.second)
        river_discharge_rate = (3e16*ur.kg/ur.year).to_base_units().magnitude
        # Upper Ocean Source: river_discharge_rate * 4.6455e-8
        self.assertAlmostEqual(Q[self.uo_id, 0], river_discharge_rate*4.6455e-8)
        self.assertEqual(Q[self.do_id, 0], 0)

    def test_variable_process_sink_vector(self):
        var = self.system.variables['PO4']
        s = self.system.get_variable_process_sink_vector(var, 1*ur.second)
        self.assertListEqual(s.tolist(), [0, 0])

    def test_variable_process_sink_matrix(self):
        S = self.system.get_variable_process_sink_matrix(0*ur.second)
        self.assertEqual(S[self.uo_id, 0], 0)
        self.assertEqual(S[self.do_id, 0], 0)

    def test_variable_process_source_vector(self):
        var = self.system.variables['PO4']
        q = self.system.get_variable_process_source_vector(var, 0*ur.second)
        self.assertListEqual(q.tolist(), [0, 0])

    def test_variable_process_source_matrix(self):
        Q = self.system.get_variable_process_source_matrix(0*ur.second)
        self.assertEqual(Q[self.uo_id, 0], 0)
        self.assertEqual(Q[self.do_id, 0], 0)

    def test_variable_flux_sink_vector(self):
        var = self.system.variables['PO4']
        s = self.system.get_variable_flux_sink_vector(var, 1*ur.second)
        self.assertListEqual(s.tolist(), [0, 0])

    def test_variable_flux_sink_matrix(self):
        S = self.system.get_variable_flux_sink_matrix(0*ur.second)
        self.assertEqual(S[self.uo_id, 0], 0)
        self.assertEqual(S[self.do_id, 0], 0)

    def test_variable_flux_source_vector(self):
        var = self.system.variables['PO4']
        q = self.system.get_variable_flux_source_vector(var, 0*ur.second)
        self.assertListEqual(q.tolist(), [0, 0])

    def test_variable_flux_source_matrix(self):
        Q = self.system.get_variable_flux_source_matrix(0*ur.second)
        self.assertEqual(Q[self.uo_id, 0], 0)
        self.assertEqual(Q[self.do_id, 0], 0)

    def test_reaction_rate_cube(self):
        rr_cube = self.system.get_reaction_rate_cube(0*ur.second)
        self.assertEqual(rr_cube[self.uo_id,0,0], 0)
        self.assertEqual(rr_cube[self.do_id,0,0], 0)

if __name__ == "__main__": 
    unittest.main()




