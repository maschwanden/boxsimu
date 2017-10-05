# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2016 at 07:55UTC

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import os
import sys
import inspect

from pint import UnitRegistry
# Use this instance of pint.UnitRegistry in order to avoid errors due 
# to the incompatibility between different instances thereof!
ur = UnitRegistry() 


# realpath() will make your script run, even if you symlink it :)
_cmd_folder = os.path.realpath(
    os.path.abspath(
        os.path.split(
            inspect.getfile(
                inspect.currentframe()))[0]))
if _cmd_folder not in sys.path:
    sys.path.insert(0, _cmd_folder)

# use this if you want to include modules from a subfolder
_cmd_subfolder = os.path.realpath(
    os.path.abspath(
        os.path.join(
            os.path.split(
                inspect.getfile(
                    inspect.currentframe()))[0],
            "subfolder")))
if _cmd_subfolder not in sys.path:
    sys.path.insert(0, _cmd_subfolder)

__all__ = [
    'box',
    'condition',
    'entities',
    'process',
    'solution',
    'solver',
    'system',
    'tests',
    'transport',
    'utils',
    'visualize',    
]

from .box import Box
from .condition import Condition
from .entities import Fluid, Variable
from .process import Process, Reaction
from .solution import Solution
from .solver import Solver
from .system import BoxModelSystem
from .transport import Flow, Flux


