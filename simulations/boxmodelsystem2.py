# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:45:10 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

Very simple test BoxModelSystem with only 4 boxes: An Upper Ocean box, a Deep Ocean box, a Lake box, and a Sediment box. 

Flows:
The Lake is connected to the atmosphere and the land by two flows: river-inflow (has a inflow concentration; does 
transport tracer) and evaporation (doesn't transport tracers). The river inflow of the Lake contains phosphate and 
nitrate with a constant concentration. The Lake is also connected to the upper ocean by a river flow towards the 
Upper Ocean (transport tracers). The Upper Ocean again is also connected to the atmosphere and the Lake by two 
flows: river-inflow (transports tracers) and evaporation (doesn't transport tracers).
The Upper Ocean and the Deep Ocean exchange mass by up- and downwelling (transport tracers).
A very small flux from the deep ocean goes into the Earth interior and a small return flux from the Earth interior into
the deep ocean is also included (Deep Ocean percolation, transports tracers and Deep Ocean founts; with input concentration).

Processes, Reactions, Fluxes:
In the Lake and the Upper Ocean Photosynthesis takes place (NO3 + PO4 -> Phyto).
In these boxes the Phytoplankton is also remineralized (Phyto -> NO3 + PO4) at a rate proportional to the available Phyto-mass.
At the same time a small part (10%) of the Phyto-mass is lost towards the deep ocean. 90% of this Phyto-mass that is
exported to the Deep Ocean is remineralized. The rest is buried in the sediment (flux into the Sediment Box). 
Additionally about 1% of the available PO4 and NO3 in the Deep Ocean box is also buried in the ocean.
In the upper ocean a constant amount of phosphate and nitrate is released from the rock bed and sediment into the 
water. At the same time (also in the upper ocean) a constant fraction of the available PO4 and NO3 is build into the
ocean bed.

Variables and Input Concentrations:
Three variables are tracked in the BoxModelSystem: Phosphate, Nitrate and Phytoplankton.
In the begining the mass of all variables in all boxes is set to negligible quantities.
Phosphate and Nitrate are transported into the lake by river-inflow. 
Phytoplankton is then produced from PO4 and NO3 by photosynthesis.

"""

import sys
import copy
import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot as plt

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


def init_system(ur=None):

    #############################
    # FLUIDS
    #############################
    seawater = Fluid('sea water', rho_expr=1020*ur.kg/ur.meter**3)
    sediment_material = Fluid('sediment material', rho_expr=2720*ur.kg/ur.meter**3)
    
    #############################
    # VARIABLES
    #############################
    po4 = Variable('po4')
    no3 = Variable('no3')
    phyto = Variable('phyto')
    
    #############################
    # PROCESSES
    #############################

    po4_release_upper_ocean = Process(
        name='PO4 release from continental shelf',
        variable=po4,
        rate=lambda t, c: 12345 * ur.kg / ur.year
    )

    po4_sink_upper_ocean = Process(
        name='PO4 sink to the continental shelf',
        variable=po4,
        rate=lambda t, c: -c.upper_ocean_var.po4 * 0.01,
    )

    no3_release_upper_ocean = Process(
        name='NO3 release from continental shelf',
        variable=no3,
        rate=lambda t, c: 123456 * ur.kg / ur.year
    )

    no3_sink_upper_ocean = Process(
        name='NO3 sink to the continental shelf',
        variable=no3,
        rate=lambda t, c: -c.upper_ocean_var.no3 * 0.01,
    )

    #############################
    # REACTIONS
    #############################

    global_photosynthesis_rate = 115e9 * ur.metric_ton / ur.year
    ratiox = global_photosynthesis_rate/(114.5*ur.kg)

    reaction_photosynthesis = Reaction(
        name = 'Photosynthesis',
        variable_reaction_coefficients={po4: -1, no3: -7.225, phyto: 114.5},
        rate=lambda t, c: min(c.po4, c.no3/7.225) * 0.8 / ur.year
    )

    reaction_remineralization = Reaction(
        name = 'Remineralization',
        variable_reaction_coefficients={po4: 1, no3: 7.225, phyto: -114.5},
        rate=lambda t, c: c.phyto * 0.4 / ur.year
    )

    #############################
    # BOXES
    #############################
    lake = Box(
        name='lake',
        name_long='Lake Box',
        fluid=seawater.q(1e16*ur.kg),
        condition=Condition(T=290*ur.kelvin, pH=7.0),
        variables=[po4.q(3*ur.kg), no3.q(1*ur.kg), phyto.q(0.324*ur.kg)],
        reactions=[reaction_photosynthesis, reaction_remineralization],
    )
    upper_ocean = Box(
        name='upper_ocean',
        name_long='Upper Ocean Box',
        fluid=seawater.q(3e19*ur.kg), 
        processes=[po4_release_upper_ocean, po4_sink_upper_ocean, 
                   no3_release_upper_ocean, no3_sink_upper_ocean],
        condition=Condition(T=280*ur.kelvin, pH=8.3),
        variables=[po4.q(3.3123*ur.kg), no3.q(0.237*ur.kg), phyto.q(0.7429*ur.kg)],
        reactions=[reaction_photosynthesis, reaction_remineralization],
    )
    deep_ocean = Box(
        name='deep_ocean',
        name_long='Deep Ocean Box',
        fluid=seawater.q(1e21*ur.kg),
        condition=Condition(T=275*ur.kelvin, pH=8.1),
        variables=[po4.q(3.492*ur.kg), no3.q(1.12437*ur.kg), phyto.q(4.324*ur.kg)],
        reactions=[reaction_remineralization],
    )
    sediment = Box(
        name='sediment',
        name_long='Sediment Box',
        fluid=seawater.q(1e10*ur.kg),
        condition=Condition(T=275*ur.kelvin, pH=7.7),
        variables=[po4.q(2.3484*ur.kg), no3.q(9.23*ur.kg), phyto.q(2.824*ur.kg)],
    )
    
    #############################
    # FLOWS
    #############################

    flow_river_to_lake = Flow(
        name='River Inflow into the Lake',
        source_box=None, 
        target_box=lake,
        rate=3e15*ur.kg/ur.year, 
        tracer_transport=True,
        concentrations={po4: 4.6455e-8*ur.kg/ur.kg, no3: 7*4.6455e-8*ur.kg/ur.kg},
    )
    
    flow_lake_evaporation = Flow(
        name='Lake Evaporation',
        source_box=lake,
        target_box=None,
        rate=1e15*ur.kg/ur.year,
        tracer_transport=False,
    )

    flow_lake_to_upper_ocean = Flow(
        name='River Flow from Lake to Upper Ocean',
        source_box=lake,
        target_box=upper_ocean,
        rate=2e15*ur.kg/ur.year,
        tracer_transport=True,
    )

    flow_upper_ocean_evaporation = Flow(
        name='Upper Ocean Evaporation',
        source_box=upper_ocean,
        target_box=None,
        rate=2e15*ur.kg/ur.year,
        tracer_transport=False,
    )

    flow_downwelling = Flow(
        name='Downwelling', 
        source_box=upper_ocean, 
        target_box=deep_ocean,
        rate=6e17*ur.kg/ur.year,
        tracer_transport=True,
    )

    flow_upwelling = Flow(
        name='Upwelling',
        source_box=deep_ocean, 
        target_box=upper_ocean,
        rate=6e17*ur.kg/ur.year, 
        tracer_transport=True,
    )

    flow_deep_ocean_percolation = Flow(
        name='Deep Ocean Percolation',
        source_box=deep_ocean,
        target_box=None,
        rate=1e11*ur.kg/ur.year,
        tracer_transport=True,
    )

    flow_deep_ocean_fount = Flow(
        name='Deep Ocean Fount',
        source_box=None,
        target_box=deep_ocean,
        rate=1e11*ur.kg/ur.year,
        tracer_transport=True,
        concentrations={po4: 4.6455e-8*ur.kg/ur.kg, no3: 7*4.6455e-8*ur.kg/ur.kg},
    )

    #############################
    # FLUXES
    #############################
    
    biological_pump = Flux(
        name='Biological Pump',
        source_box=upper_ocean,
        target_box=deep_ocean,
        variable=phyto,
        rate=lambda t, c: c.upper_ocean_var.phyto * 0.1,
    )

    oc_burial = Flux(
        name='OC burial in the sediment',
        source_box=deep_ocean,
        target_box=sediment,
        variable=phyto,
        rate=lambda t, c: (c.upper_ocean_var.phyto * 0.1) * 0.1,
    )

    #############################
    # SYSTEM
    #############################

    system = BoxModelSystem('BoxModelSystem No. 2', 
                          [lake, upper_ocean, deep_ocean, sediment], 
                          flows=[flow_lake_evaporation, flow_lake_to_upper_ocean, flow_river_to_lake, 
                                 flow_downwelling, flow_upwelling, flow_upper_ocean_evaporation, 
                                 flow_deep_ocean_percolation, flow_deep_ocean_fount],
                          fluxes=[biological_pump, oc_burial], 
                          global_condition=Condition(T=288*ur.kelvin, pH=7.3),
    )
    return system

