# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import unittest
from unittest import TestCase

import os
import sys
import copy
import numpy as np
import datetime
import math

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
from boxsimu.simulations import boxmodelsystem3
from boxsimu import ur


class BoxModelSystem3Test(TestCase):
    """Test boxsimu framework using an intermediate complex box model."""

    def setUp(self, *args, **kwargs):
        self.system = boxmodelsystem3.get_system()
        self.solver = Solver(self.system)
        self.box1 = self.system.boxes.box1
        self.box2 = self.system.boxes.box2

        self.A = self.system.variables.A
        self.B = self.system.variables.B
        self.C = self.system.variables.C
        self.D = self.system.variables.D

    def tearDown(self, *args, **kwargs):
       del(self.system)
       del(self.solver)
       del(self.box1)
       del(self.box2)

       del(self.A)
       del(self.B)
       del(self.C)
       del(self.D)

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
        self.assertEqual(self.box1.mass, 1e5*ur.kg + 6*ur.kg)
        self.assertEqual(self.box2.mass, 1e5*ur.kg + 2*ur.kg)

    def test_volume(self):
        box1_context = self.system.get_box_context(self.box1)
        box2_context = self.system.get_box_context(self.box2)

        self.assertEqual(self.box1.get_volume(box1_context),
                1e5/1000 * ur.meter**3)
        self.assertEqual(self.box2.get_volume(box1_context),
                1e5/1000 * ur.meter**3)

    def test_concentration(self):
        pass

    #####################################################
    # Reaction Functions
    #####################################################
    
    def test_reaction_get_reverse_reaction(self):
        reaction1 = self.system.reactions[0]
        reverse1 = reaction1.get_reverse_reaction('reverse reaction 1')
        self.assertEqual(reverse1.name, 'reverse reaction 1')
        self.assertEqual(reverse1.variable_reaction_coefficients.get(self.A, 0), 
                -reaction1.variable_reaction_coefficients.get(self.A, 0)) 
        self.assertEqual(reverse1.variable_reaction_coefficients.get(self.B, 0), 
                -reaction1.variable_reaction_coefficients.get(self.B, 0)) 
        self.assertEqual(reverse1.variable_reaction_coefficients.get(self.C, 0), 
                -reaction1.variable_reaction_coefficients.get(self.C, 0)) 
        self.assertEqual(reverse1.variable_reaction_coefficients.get(self.D, 0), 
                -reaction1.variable_reaction_coefficients.get(self.D, 0)) 

        reaction2 = self.system.reactions[1]
        reverse2 = reaction2.get_reverse_reaction('reverse reaction 2')
        self.assertEqual(reverse2.name, 'reverse reaction 2')
        self.assertEqual(reverse2.variable_reaction_coefficients.get(self.A, 0), 
                -reaction2.variable_reaction_coefficients.get(self.A, 0)) 
        self.assertEqual(reverse2.variable_reaction_coefficients.get(self.B, 0), 
                -reaction2.variable_reaction_coefficients.get(self.B, 0)) 
        self.assertEqual(reverse2.variable_reaction_coefficients.get(self.C, 0), 
                -reaction2.variable_reaction_coefficients.get(self.C, 0)) 
        self.assertEqual(reverse2.variable_reaction_coefficients.get(self.D, 0), 
                -reaction2.variable_reaction_coefficients.get(self.D, 0)) 

    #####################################################
    # Base Functions 
    #####################################################
    
    def test_box_id(self):
        self.assertEqual(self.box1.id, 0)
        self.assertEqual(self.box2.id, 1)

    def test_variable_id(self):
        self.assertEqual(self.A.id, 0)
        self.assertEqual(self.B.id, 1)
        self.assertEqual(self.C.id, 2)
        self.assertEqual(self.D.id, 3)

    def test_N_boxes(self):
        self.assertEqual(self.system.N_boxes, 2)
    
    def test_N_variables(self):
        self.assertEqual(self.system.N_variables, 4)

    def test_context_of_box(self):
        global_context = self.system.get_box_context()
        box1_context = self.system.get_box_context(self.box1)
        box2_context = self.system.get_box_context(self.box2)
        
        # Test accessability of the condition attributes
        self.assertEqual(global_context.T, 295 * ur.kelvin)
        self.assertEqual(box1_context.T, 290 * ur.kelvin)
        self.assertEqual(box2_context.T, 300 * ur.kelvin)

        # Test the accessability of the condition attributes of other boxes:
        self.assertEqual(global_context.box1.condition.T, 
                290 * ur.kelvin)
        self.assertEqual(global_context.global_condition.T, 
                295 * ur.kelvin) 

    def test_context_evaluation_lambda_func(self):
        global_context = self.system.get_box_context()
        box1_context = self.system.get_box_context(self.box1)
        box2_context = self.system.get_box_context(self.box2)

        lambda1 = lambda t, c: c.T / (100*ur.kelvin)
        self.assertEqual(lambda1(0*ur.second, global_context), 2.95)
        self.assertEqual(lambda1(0*ur.second, box1_context), 2.90)
        self.assertEqual(lambda1(0*ur.second, box2_context), 3.00)


    #####################################################
    # Fluid and Variable Mass/Concentration Vectors/Matrices
    #####################################################

    def test_fluid_mass_1Dlist_1Darray(self):
        m = self.system.get_fluid_mass_1Darray()
        self.assertEqual(m[self.box1.id], 1e5 * ur.kg)
        self.assertEqual(m[self.box2.id], 1e5 * ur.kg)

    def test_variable_mass_1Darray(self):
        m = self.system.get_variable_mass_1Darray(self.A)
        self.assertEqual(m[self.box1.id], 3 * ur.kg)
        self.assertEqual(m[self.box2.id], 1 * ur.kg)

        m = self.system.get_variable_mass_1Darray(self.B)
        self.assertEqual(m[self.box1.id], 3 * ur.kg)
        self.assertEqual(m[self.box2.id], 1 * ur.kg)
        
        m = self.system.get_variable_mass_1Darray(self.C)
        self.assertEqual(m[self.box1.id], 0 * ur.kg)
        self.assertEqual(m[self.box2.id], 0 * ur.kg)

        m = self.system.get_variable_mass_1Darray(self.D)
        self.assertEqual(m[self.box1.id], 0 * ur.kg)
        self.assertEqual(m[self.box2.id], 0 * ur.kg)

    def test_variable_concentration_1Darray(self):
        def _c(var_mass, fluid_mass):
            return var_mass / (fluid_mass + var_mass) * ur.dimensionless

        c = self.system.get_variable_concentration_1Darray(self.A)
        self.assertAlmostEqual(c[self.box1.id], _c(3, 1e5))
        self.assertAlmostEqual(c[self.box2.id], _c(1, 1e5))
        c = self.system.get_variable_concentration_1Darray(self.B)
        self.assertAlmostEqual(c[self.box1.id], _c(3, 1e5))
        self.assertAlmostEqual(c[self.box2.id], _c(1, 1e5))
        c = self.system.get_variable_concentration_1Darray(self.C)
        self.assertAlmostEqual(c[self.box1.id], _c(0, 1e5))
        self.assertAlmostEqual(c[self.box2.id], _c(0, 1e5))
        c = self.system.get_variable_concentration_1Darray(self.D)
        self.assertAlmostEqual(c[self.box1.id], _c(0, 1e5))
        self.assertAlmostEqual(c[self.box2.id], _c(0, 1e5))

    #####################################################
    # Mass Flow Vectors/Matrices
    #####################################################

    def test_fluid_mass_internal_flow_2Darray(self):
        A = self.system.get_fluid_mass_internal_flow_2Darray(0*ur.second)
        # Check that diagonal elements are zero
        self.assertPintQuantityAlmostEqual(A[self.box1.id, self.box1.id], 
                0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(A[self.box2.id, self.box2.id], 
                0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(A[self.box1.id, self.box2.id], 
                1e3*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(A[self.box2.id, self.box1.id], 
                0*ur.kg/ur.year)

    def test_fluid_mass_flow_sink_1Darray(self):
        s = self.system.get_fluid_mass_flow_sink_1Darray(0*ur.second)
        self.assertPintQuantityAlmostEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(s[self.box2.id], 1e3*ur.kg/ur.year)

    def test_fluid_mass_flow_source_1Darray(self):
        q = self.system.get_fluid_mass_flow_source_1Darray(0*ur.second)
        self.assertPintQuantityAlmostEqual(q[self.box1.id], 1e3*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(q[self.box2.id], 0*ur.kg/ur.year)

    #####################################################
    # Variable Sink/Source Vectors
    #####################################################

    def test_variable_internal_flow_2Darray(self):
        f_flow = np.ones(self.system.N_boxes)

        A = self.system.get_variable_internal_flow_2Darray(
                self.A, 0*ur.second, f_flow)
        c = self.system.get_variable_concentration_1Darray(self.A)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(A[self.box1.id, self.box2.id], 
                c[self.box1.id] * 1e3*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

        A = self.system.get_variable_internal_flow_2Darray(
                self.B, 0*ur.second, f_flow)
        c = self.system.get_variable_concentration_1Darray(self.B)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(A[self.box1.id, self.box2.id], 
                c[self.box1.id] * 1e3*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

        A = self.system.get_variable_internal_flow_2Darray(
                self.C, 0*ur.second, f_flow)
        c = self.system.get_variable_concentration_1Darray(self.C)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box1.id, self.box2.id], 
                c[self.box1.id] * 1e3*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

        # Important: D is not transported since the conditions in 
        # box1 make it non-mobile!
        A = self.system.get_variable_internal_flow_2Darray(
                self.D, 0*ur.second, f_flow)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box1.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

    def test_variable_flow_sink_1Darray(self):
        f_flow = np.ones(self.system.N_boxes)

        s = self.system.get_variable_flow_sink_1Darray(self.A, 
                0*ur.second, f_flow)
        c = self.system.get_variable_concentration_1Darray(self.A)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(s[self.box2.id], 
                1e3 * ur.kg/ur.year * c[self.box2.id])

        s = self.system.get_variable_flow_sink_1Darray(self.B, 
                0*ur.second, f_flow)
        c = self.system.get_variable_concentration_1Darray(self.B)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertPintQuantityAlmostEqual(s[self.box2.id], 
                1e3 * ur.kg/ur.year * c[self.box2.id])

        s = self.system.get_variable_flow_sink_1Darray(self.C, 
                0*ur.second, f_flow)
        c = self.system.get_variable_concentration_1Darray(self.C)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.box2.id], 
                1e3 * ur.kg/ur.year * c[self.box2.id])

        # Important: D is not transported since the conditions in 
        # box1 make it non-mobile!
        s = self.system.get_variable_flow_sink_1Darray(self.D, 
                0*ur.second, f_flow)
        c = self.system.get_variable_concentration_1Darray(self.D)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.box2.id], 0*ur.kg/ur.year)

    def test_variable_flow_source_1Darray(self):
        box1_input_concentration = {self.A: 1*ur.gram/ur.kg,
                                  self.B: 2*ur.gram/ur.kg}

        q = self.system.get_variable_flow_source_1Darray(self.A, 0*ur.second)
        self.assertEqual(q[self.box1.id], 
                 1e3 * ur.kg/ur.year * box1_input_concentration[self.A])
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_flow_source_1Darray(self.B, 0*ur.second)
        self.assertEqual(q[self.box1.id], 
                 1e3 * ur.kg/ur.year * box1_input_concentration[self.B])
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_flow_source_1Darray(self.C, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_flow_source_1Darray(self.D, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

    def test_variable_process_sink_1Darray(self):
        s = self.system.get_variable_process_sink_1Darray(
                self.A, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.box2.id], 0*ur.kg/ur.year)

        s = self.system.get_variable_process_sink_1Darray(
                self.B, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.box2.id], 0*ur.kg/ur.year)

        s = self.system.get_variable_process_sink_1Darray(
                self.C, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.box2.id], 0*ur.kg/ur.year)

        s = self.system.get_variable_process_sink_1Darray(
                self.D, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(s[self.box2.id], 0*ur.kg/ur.year)

    def test_variable_process_source_1Darray(self):
        q = self.system.get_variable_process_source_1Darray(
                self.A, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_process_source_1Darray(
                self.B, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_process_source_1Darray(
                self.C, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

        q = self.system.get_variable_process_source_1Darray(
                self.D, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(q[self.box2.id], 0*ur.kg/ur.year)

    def test_variable_internal_flux_2Darray(self):
        A = self.system.get_variable_internal_flux_2Darray(
                self.A, 0*ur.second)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box1.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

        A = self.system.get_variable_internal_flux_2Darray(
                self.B, 0*ur.second)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box1.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

        A = self.system.get_variable_internal_flux_2Darray(
                self.C, 0*ur.second)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box1.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

        A = self.system.get_variable_internal_flux_2Darray(
                self.D, 0*ur.second)
        self.assertEqual(A[self.box1.id, self.box1.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box1.id, self.box2.id], 0*ur.kg/ur.year)
        self.assertEqual(A[self.box2.id, self.box1.id], 0*ur.kg/ur.year)

    def test_variable_flux_sink_1Darray(self):
        s = self.system.get_variable_flux_sink_1Darray(
                self.A, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.box2.id], 0 * ur.kg / ur.year)

        s = self.system.get_variable_flux_sink_1Darray(
                self.B, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.box2.id], 0 * ur.kg / ur.year)

        s = self.system.get_variable_flux_sink_1Darray(
                self.C, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.box2.id], 0 * ur.kg / ur.year)

        s = self.system.get_variable_flux_sink_1Darray(
                self.D, 0*ur.second)
        self.assertEqual(s[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(s[self.box2.id], 0 * ur.kg / ur.year)

    def test_variable_flux_source_1Darray(self):
        q = self.system.get_variable_flux_source_1Darray(
                self.A, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.box2.id], 0 * ur.kg / ur.year)

        q = self.system.get_variable_flux_source_1Darray(
                self.B, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.box2.id], 0 * ur.kg / ur.year)

        q = self.system.get_variable_flux_source_1Darray(
                self.C, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.box2.id], 0 * ur.kg / ur.year)

        q = self.system.get_variable_flux_source_1Darray(
                self.D, 0*ur.second)
        self.assertEqual(q[self.box1.id], 0 * ur.kg / ur.year)
        self.assertEqual(q[self.box2.id], 0 * ur.kg / ur.year)

    def test_reaction_rate_cube(self):
        C = self.system.get_reaction_rate_3Darray(
                0*ur.second)
        m_A = self.system.get_variable_mass_1Darray(self.A)
        m_B = self.system.get_variable_mass_1Darray(self.B)
        m_C = self.system.get_variable_mass_1Darray(self.C)
        m_D = self.system.get_variable_mass_1Darray(self.D)

        rr_1 = 0.2 * m_B[self.box1.id] / (5 * ur.year)
        rr_2 = 0 * ur.kg / ur.year

        reaction1_id = 0
        reaction2_id = 1
        
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.A.id, reaction1_id], -rr_1*3)
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.B.id, reaction1_id], -rr_1*5)
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.C.id, reaction1_id], rr_1*2)
        self.assertEqual(
                C[self.box1.id, self.D.id, reaction1_id], rr_1*0)

        self.assertEqual(
                C[self.box1.id, self.A.id, reaction2_id], rr_2*0)
        self.assertEqual(
                C[self.box1.id, self.B.id, reaction2_id], rr_2*0)
        self.assertEqual(
                C[self.box1.id, self.C.id, reaction2_id], -rr_2*1)
        self.assertEqual(
                C[self.box1.id, self.D.id, reaction2_id], rr_2*1)

        self.system.boxes.box1.variables.C.mass = 10 * ur.kg
        C = self.system.get_reaction_rate_3Darray(
                0*ur.second)
        m_C = self.system.get_variable_mass_1Darray(self.C)

        rr_1 = 0.2 * m_B[self.box1.id] / (5 * ur.year)
        rr_2 = 0.1 * (m_C[self.box1.id]-0.5*ur.kg) / ur.year

        reaction1_id = 0
        reaction2_id = 1
        
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.A.id, reaction1_id], -rr_1*3)
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.B.id, reaction1_id], -rr_1*5)
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.C.id, reaction1_id], rr_1*2)
        self.assertEqual(
                C[self.box1.id, self.D.id, reaction1_id], rr_1*0)

        self.assertEqual(
                C[self.box1.id, self.A.id, reaction2_id], rr_2*0)
        self.assertEqual(
                C[self.box1.id, self.B.id, reaction2_id], rr_2*0)
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.C.id, reaction2_id], -rr_2*1)
        self.assertPintQuantityAlmostEqual(
                C[self.box1.id, self.D.id, reaction2_id], rr_2*1)

if __name__ == "__main__": 
    unittest.main()




