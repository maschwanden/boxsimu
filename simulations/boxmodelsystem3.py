# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: aschi

2 Box Model with 4 different Variables and 2 Reactions.
Variables: A, B, C
Reactions:
    Reaction 1 : 3A + 5B -> 2C
    Reaction 2 : If Concentration(C) > Concentration_crit : C -> D

"""

import sys
import os
import copy
import numpy as np
import datetime

from matplotlib import pyplot as plt

import boxsimu
from boxsimu import ur


def get_system():
    #############################
    # FLUIDS
    #############################

    water = boxsimu.Fluid('water', rho=1000*ur.kg/ur.meter**3)
    
    #############################
    # VARIABLES
    #############################

    A = boxsimu.Variable('A', description='Variable A')
    B = boxsimu.Variable('B', description='Variable B')
    C = boxsimu.Variable('C', description='Variable C')
    # Variable D is mobile solubale if the temperature is above 298K
    D = boxsimu.Variable('D', mobility=lambda t, c, s: c.T > 298*ur.kelvin)

    #############################
    # REACTIONS
    #############################

    reaction1 = boxsimu.Reaction(
        name = 'Reaction1',
        reaction_coefficients={A: -3, B: -5, C: 2},
        rate=lambda t, c, s: min(c.A/3, c.B/5) * 2.2 / ur.year
    )

    def rr2(t, c, s):
        """If Mass(C) > Mass_crit : C -> D."""
        m_crit = 0.5 * ur.kg 
        if c.C > m_crit:
            return (c.C-m_crit) * 0.1 / ur.year
        return 0 * ur.kg / ur.year

    reaction2 = boxsimu.Reaction(
        name = 'Reaction2',
        reaction_coefficients={C: -1, D: 1},
        rate=rr2,
    )
    
    #############################
    # BOXES
    #############################

    box1 = boxsimu.Box(
        name='box1',
        description='Box 1',
        fluid=water.q(1e5*ur.kg), 
        condition=boxsimu.Condition(T=290*ur.kelvin),
        variables=[A.q(1*ur.kg), B.q(3*ur.kg)],
        reactions=[reaction1, reaction2],
    )
    box2 = boxsimu.Box(
        name='box2',
        description='Box 2',
        fluid=water.q(1e5*ur.kg), 
        condition=boxsimu.Condition(T=300*ur.kelvin),
        variables=[A.q(2*ur.kg), B.q(1*ur.kg)],
        reactions=[reaction1, reaction2],
    )

    #############################
    # FLOWS
    #############################

    inflow_box1 = boxsimu.Flow(
        name='Inflow', 
        source_box=None, 
        target_box=box1,
        rate=1e3*ur.kg/ur.year,
        tracer_transport=True,
        concentrations={
            #A: 1 * ur.gram / ur.kg,
            B: 2 * ur.gram / ur.kg,
        }
    )

    flow_box1_to_box2 = boxsimu.Flow(
        name='FlowBox1_Box2', 
        source_box=box1, 
        target_box=box2,
        rate=1e3*ur.kg/ur.year,
        tracer_transport=True,
    )
    
    outflow_box2 = boxsimu.Flow(
        name='Outflow',
        source_box=box2, 
        target_box=None,
        rate=1e3*ur.kg/ur.year, 
        tracer_transport=True,
    )
    
    #############################
    # SYSTEM
    #############################

    bmsystem = boxsimu.BoxModelSystem(
            name='TestSystem', 
            description='Das ist die Beschreibung eines Test Systems!',
            boxes=[box1, box2],
            flows=[inflow_box1, flow_box1_to_box2, outflow_box2],
            global_condition=boxsimu.Condition(T=295*ur.kelvin),
    )
    return bmsystem



