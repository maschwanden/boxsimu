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


if not os.path.abspath(__file__ + "/../../../") in sys.path:
    sys.path.append(os.path.abspath(__file__ + "/../../../"))


from boxsimu.entities import Fluid, Variable
from boxsimu.box import Box
from boxsimu.transport import  Flow, Flux
from boxsimu.condition import Condition
from boxsimu.system import BoxModelSystem 
from boxsimu.process import Process, Reaction
from boxsimu import utils
from boxsimu import ur


def get_system():
    #############################
    # FLUIDS
    #############################

    water = Fluid('water', rho=1000*ur.kg/ur.meter**3)
    
    #############################
    # VARIABLES
    #############################

    A = Variable('A')
    B = Variable('B')
    C = Variable('C')
    # Variable D is mobile solubale if the temperature is above 298K
    D = Variable('D', mobility=lambda t,c: c.T > 298*ur.kelvin)

    #############################
    # REACTIONS
    #############################

    reaction1 = Reaction(
        name = 'Reaction1',
        variable_reaction_coefficients={A: -3, B: -5, C: 2},
        rate=lambda t, c: min(c.A/3, c.B/5) * 2.2 / ur.year
    )

    def rr2(t, c):
        """If Mass(C) > Mass_crit : C -> D."""
        m_crit = 0.5 * ur.kg 
        if c.C > m_crit:
            return (c.C-m_crit) * 0.1 / ur.year
        return 0 * ur.kg / ur.year

    reaction2 = Reaction(
        name = 'Reaction2',
        variable_reaction_coefficients={C: -1, D: 1},
        rate=rr2,
    )
    
    #############################
    # BOXES
    #############################

    box1 = Box(
        name='box1',
        name_long='Box 1',
        fluid=water.q(1e5*ur.kg), 
        condition=Condition(T=290*ur.kelvin),
        variables=[A.q(1*ur.kg), B.q(3*ur.kg)],
        reactions=[reaction1, reaction2],
    )
    box2 = Box(
        name='box2',
        name_long='Box 2',
        fluid=water.q(1e5*ur.kg), 
        condition=Condition(T=300*ur.kelvin),
        variables=[A.q(2*ur.kg), B.q(1*ur.kg)],
        reactions=[reaction1, reaction2],
    )

    #############################
    # FLOWS
    #############################

    inflow_box1 = Flow(
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

    flow_box1_to_box2 = Flow(
        name='Inflow', 
        source_box=box1, 
        target_box=box2,
        rate=1e3*ur.kg/ur.year,
        tracer_transport=True,
    )
    
    outflow_box2 = Flow(
        name='Outflow',
        source_box=box2, 
        target_box=None,
        rate=1e3*ur.kg/ur.year, 
        tracer_transport=True,
    )
    
    #############################
    # SYSTEM
    #############################

    bmsystem = BoxModelSystem(
            name='Test System', 
            boxes=[box1, box2],
            flows=[inflow_box1, flow_box1_to_box2, outflow_box2],
            global_condition=Condition(T=295*ur.kelvin),
    )
    return bmsystem



