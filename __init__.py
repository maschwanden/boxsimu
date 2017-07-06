# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 07:55:42 2016

@author: aschi
"""

import os, sys, inspect


# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"subfolder")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

__all__ = ['action', 'box', 'condition', 'entities', 'process', 'solution', 'solver', 'system', 'tests', 'transport', 'utils']

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

