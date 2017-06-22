# -*- coding: utf-8 -*-

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
                     BoxModelSystem, Process, Reaction, Flux)
from boxsimu import utils


#############################
# FLUIDS
#############################
# water density parameters
A = 0.14395
B = 0.0112
C = 649.727*ur.kelvin
D = 0.05107
rho_expr_water = lambda time, c: A / (B**(1+(1-(c.T/C))**D))*ur.kg/ur.meter**3

water = Fluid('water', rho_expr=rho_expr_water, mass=8e5*ur.kg)
lakewater = copy.deepcopy(water)
lakewater.mass = 8e4*ur.kg

#############################
# CONDITIONS
#############################
condition_lake = Condition(T=290*ur.kelvin, pH=5)
condition_upper_ocean = Condition(T=280*ur.kelvin)
condition_deep_ocean = Condition(T=275*ur.kelvin)

#############################
# VARIABLES
#############################
phosphate = Variable('PO4', 1*ur.kg)
nitrate = Variable('NO3', 2*ur.kg)
organic_compound1 = Variable('OC1', 3*ur.kg)
phyto = Variable('phyto', 10*ur.kg)

#############################
# PROCESSES
#############################
process_photolytic_deg_rate = lambda t, c: -max(0,((- 10*ur.kelvin*np.cos(2*np.pi*t / (24*ur.hour))) / (10*ur.kelvin))) * 10*ur.gram/ur.hour
process_photolytic_deg = Process(
        'Photolytic Degradation', 
        organic_compound1, 
        process_photolytic_deg_rate
) 

#############################
# REACTIONS
#############################
reaction_photosynthesis = Reaction(
    name = 'Photosynthesis',
    variables = [nitrate, phosphate, phyto],
    variable_coeffs=[-7.225, -1, + 114.5],  # Redfield ratio in weight instead of mol
    rate=lambda t, c: (max(0,((- 10*ur.kelvin*np.cos(2*np.pi*t / (24*ur.hour))) / (10*ur.kelvin))) * 10)**1.4 *ur.gram/ur.hour
)

#############################
# BOXES
#############################
lake = Box(
    name='lake',
    name_long='Medium Size Lake',
    fluid=lakewater,
    #processes=[process_photolytic_deg,],
    #reactions=[reaction_photosynthesis, ],
    variables=[phosphate,], # organic_compound1],
    condition=condition_lake,
)
upper_ocean = Box(
    name='upper_ocean',
    name_long='Upper Ocean Box',
    fluid=water, 
    #processes=[process_photolytic_deg,],
    reactions=[],
    variables=[phosphate, ],
    condition=condition_upper_ocean,
)
deep_ocean = Box(
    name='deep_ocean',
    name_long='Deep Ocean Box', 
    fluid=water, 
    processes=[],
    variables=[nitrate, ],
    condition=condition_deep_ocean,
)


#############################
# FLOWS
#############################
f1 = Flow(
    name='River flow from Lake to Ocean', 
    source_box=lake, 
    target_box=upper_ocean,
    rate=0.2e5*ur.kg/ur.second,
)
f2 = Flow(
    name='Downwelling', 
    source_box=upper_ocean, 
    target_box=deep_ocean,
    rate=lambda t, c: 2.6e5*ur.kg/ur.second, 
)
f3 = Flow(
    name='Upwelling',
    source_box=deep_ocean, 
    target_box=upper_ocean,
    rate=lambda t, c: 2.5e5*ur.kg/ur.second, 
)
f4 = Flow(
    name='Deep Sea Percolation into Earth-Interior',
    source_box=deep_ocean, 
    target_box=None,
    rate=lambda t, c: 0.09e5*ur.kg/ur.second, 
)
f5 = Flow(
    name='Rain',
    source_box=None, 
    target_box=lake,
    rate=lambda t, c: min(min(np.exp(t/(800*ur.minute)),5)*0.1e5*ur.kg/ur.second, 0.21e5*ur.kg/ur.second), 
)

#############################
# FLUXES
#############################

flux1 = Flux('Biological pump', upper_ocean, deep_ocean, phyto, lambda t, c: c.box1.variables.phyto.mass * 0.1 / ur.day)



#############################
# SYSTEM
#############################
sys = BoxModelSystem('Test System', 
                      [lake, upper_ocean, deep_ocean], 
                      Condition(T=301.11, pH=8.3),
                      flows=[f1,f2,f3,f4,f5],
                      #fluxes=[flux1, ],
                      )

#sol = sys.solve_flows(100*ur.second, 1*ur.second)


sol = sys.solve(1.0*ur.min, 5*ur.second)
sol.plot_variables_masses(['NO3', 'PO4'])
sol.plot_box_masses()

#sol.plot_quantities(sol.boxes.upper_ocean.volume, 'Volume of upper ocean')
#sol.plot_quantities(sol.boxes.deep_ocean.volume, 'Volume of deep ocean')
#f = deep_ocean.get_volume_change_mag_func()
#f(1*ur.seconds)


print(datetime.datetime.now())
