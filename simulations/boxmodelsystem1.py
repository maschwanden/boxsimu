# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)


Very simple test BoxModelSystem with only 2 boxes: An upper ocean box and a deep ocean box. 
The upper ocean is connected to the land and atmosphere by two fluid flows: river inflow and evaporation.
The upper and the deep ocean box exchange mass by up- and downwelling.
No other transports, fluxes, reactions, or processes are defined.
"""

import sys
import os
import copy
import numpy as np
import datetime

from matplotlib import pyplot as plt


if not os.path.abspath(__file__ + "/../../../") in sys.path:
    sys.path.append(os.path.abspath(__file__ + "/../../../"))


from boxsimu import box
from boxsimu import entities
from boxsimu import condition
from boxsimu import process
from boxsimu import system
from boxsimu import transport
from boxsimu import ur

def get_system():
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

    seawater = entities.Fluid('sea water', rho=1000*ur.kg/ur.meter**3)
    
    #############################
    # VARIABLES
    #############################

    po4 = entities.Variable('po4')
    
    #############################
    # BOXES
    #############################

    upper_ocean = box.Box(
        name='upper_ocean',
        name_long='Upper Ocean Box',
        fluid=seawater.q(M1*ur.kg), 
        condition=condition.Condition(T=333*ur.kelvin),
    )
    deep_ocean = box.Box(
        name='deep_ocean',
        name_long='Deep Ocean Box', 
        fluid=seawater.q(M2*ur.kg),
        condition=condition.Condition(T=222*ur.kelvin),
    )
    
    #############################
    # FLOWS
    #############################

    flow_downwelling = transport.Flow(
        name='Downwelling', 
        source_box=upper_ocean, 
        target_box=deep_ocean,
        rate=6e17*ur.kg/ur.year,
        tracer_transport=True,
    )
    
    flow_upwelling = transport.Flow(
        name='Upwelling',
        source_box=deep_ocean, 
        target_box=upper_ocean,
        rate=6e17*ur.kg/ur.year, 
        tracer_transport=True,
    )
    
    flow_river_water = transport.Flow(
        name='River Inflow into Upper Ocean',
        source_box=None, 
        target_box=upper_ocean,
        rate=3e16*ur.kg/ur.year, 
        tracer_transport=True,
        concentrations={po4: 4.6455e-8*ur.kg/ur.kg},
    )
    
    flow_upper_ocean_evaporation = transport.Flow(
        name='Upper Ocean Evaporation',
        source_box=upper_ocean,
        target_box=None,
        rate=3e16*ur.kg/ur.year,
        tracer_transport=False,
    )
    
    #############################
    # SYSTEM
    #############################

    bmsystem = system.BoxModelSystem('Test System    123 ()!!! XD & cd/', 
                          [upper_ocean, deep_ocean], 
                          flows=[flow_downwelling, flow_upwelling, 
                                 flow_river_water, flow_upper_ocean_evaporation],
                          global_condition=condition.Condition(T=111*ur.kelvin),
    )
    return bmsystem

