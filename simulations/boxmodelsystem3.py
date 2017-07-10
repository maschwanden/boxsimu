# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: aschi

1 Box Model with 4 different Variables and 2 Reactions.
Variables: A, B, C
Reactions:
    3A + 5B -> 2C
    If Concentration(C) > Concentration_crit : C -> D

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

from boxsimu import box
from boxsimu import entities
from boxsimu import condition
from boxsimu import process
from boxsimu import system
from boxsimu import transport


def init_system(ur=None):
    #############################
    # FLUIDS
    #############################

    seawater = entities.Fluid('sea water', rho_expr=1000*ur.kg/ur.meter**3)
    
    #############################
    # VARIABLES
    #############################

    A = entities.Variable('A')
    B = entities.Variable('B')
    C = entities.Variable('C')
    D = entities.Variable('D')

    #############################
    # REACTIONS
    #############################
    reaction1_rate = lambda t, c: 1*ur.kg/ur.day
    reaction1 = Reaction(
        name = 'Reaction A and B to C',
        variable_reaction_coefficients={A: -3, B: -5, C: 2},
        rate=reaction1_rate,
    )

    def reaction2_rate(t, c):
        if c.C > 10*ur.kg:
            return 1*ur.kg
        else:
            return 0*ur.kg

    reaction2 = Reaction(
        name = 'Reaction C to D',
        variable_reaction_coefficients={C: -1, D: 1},
        rate=reaction2_rate,
    )
    
    #############################
    # BOXES
    #############################

    box1 = box.Box(
        name='box1',
        name_long='Box 1', 
        fluid=seawater.q(1e6*ur.kg),
        condition=condition.Condition(T=300*ur.kelvin),
        variables=[A.q(20*ur.kg), B.q(50*ur.kg)],
    )
    
    #############################
    # FLOWS
    #############################

    inflow = transport.Flow(
        name='Inflow', 
        source_box=None, 
        target_box=box1,
        rate=1e3*ur.kg/ur.day,
        tracer_transport=True,
        concentrations={A: 2*ur.gram/ur.kg, B: 3*ur.gram/ur.kg},
    )
    
    outflow = transport.Flow(
        name='Outflow',
        source_box=box1, 
        target_box=None,
        rate=1.02e3*ur.kg/ur.day, 
        tracer_transport=True,
    )
    
    #############################
    # SYSTEM
    #############################

    bmsystem = system.BoxModelSystem('Reaction Testing System', 
                          [box1, ], 
                          flows=[inflow, outflow],
                          global_condition=condition.Condition(T=299*ur.kelvin),
    )
    return bmsystem

