# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 07:55:42 2016

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import os
import sys
import inspect


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
    'action',
    'box',
    'condition',
    'entities',
    'process',
    'solution',
    'solver',
    'system',
    'tests',
    'transport',
    'utils']

from . import action
from . import box
from . import condition
from . import entities
from . import errors
from . import process
from . import solution
from . import solver
from . import system
from . import transport
from . import utils

from .box import Box
from .condition import Condition
from .entities import Fluid, Variable
from .process import Process, Reaction
from .transport import Flow, Flux
from .system import BoxModelSystem
