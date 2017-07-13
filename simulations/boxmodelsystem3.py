# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: aschi

1 Box Model with 4 different Variables and 2 Reactions.
Variables: A, B, C
Reactions:
    Reaction 1 : 3A + 5B -> 2C
    Reaction 2 : If Concentration(C) > Concentration_crit : C -> D

"""

import sys
import copy
import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot as plt

BOXSIMU_PATH = '/home/aschi/Documents/MyPrivateRepo/'
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


def get_system(ur):

    #############################
    # FLUIDS
    #############################

    water = Fluid('water', rho_expr=1000*ur.kg/ur.meter**3)
    
    #############################
    # VARIABLES
    #############################

    A = Variable('A')
    B = Variable('B')
    C = Variable('C')
    D = Variable('D')

    #############################
    # REACTIONS
    #############################

    reaction1 = Reaction(
        name = 'Reaction1',
        variable_reaction_coefficients={A: -3, B: -5, C: 2},
        rate=lambda t, c: min(c.A/3, c.B/5) * 0.2 / ur.year
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
        variables=[A.q(3*ur.kg), B.q(3*ur.kg)],
        reactions=[reaction1, reaction2],
    )

    #############################
    # FLOWS
    #############################

    inflow = Flow(
        name='Inflow', 
        source_box=None, 
        target_box=box1,
        rate=1e3*ur.kg/ur.year,
        tracer_transport=True,
        concentrations={
            A: 1 * ur.gram / ur.kg,
            B: 2 * ur.gram / ur.kg,
        }
    )
    
    outflow = Flow(
        name='Outflow',
        source_box=box1, 
        target_box=None,
        rate=1e3*ur.kg/ur.year, 
        tracer_transport=True,
    )
    
    #############################
    # SYSTEM
    #############################

    bmsystem = BoxModelSystem(
            name='Test System', 
            boxes=[box1],
            flows=[inflow, outflow],
            global_condition=Condition(T=295*ur.kelvin),
    )
    return bmsystem



