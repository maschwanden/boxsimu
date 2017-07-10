# -*- coding: utf-8 -*-

box_simu_path = '/home/aschi/Code'
import sys
if not box_simu_path in sys.path:
    sys.path.append(box_simu_path)

from BoxSimu import Box, Fluid, Solvent, Substance, System

from pint import UnitRegistry
ur = UnitRegistry()


fluid1 = Fluid('H2O', 1000*ur.kilogram/ur.meter**3 , volume=1000*ur.meter**3)
substance1 = Substance('salt', 'Mass per Volume')
solvent1 = Solvent('sea water', fluids=[fluid1,])
#box1 = Box('TEST BOX1', solvent, substances, processes, condition)

system1 = System('Test System', [substance1, ], boxes, fluxes=None, flows=None, condition=None,)

print('abc')