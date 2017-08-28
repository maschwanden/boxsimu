# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:57:03 2017

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import os
import re
import yaml

DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), 'config/.default/')
DEFAULT_SVG_VISUALIZATION_CONFIG_FILE = os.path.join(
    DEFAULT_CONFIG_DIR, 'svg_visualization.yml')

base_config = {}

# LOAD BASE CONFIG
if os.path.exists('config.yaml'):
    base_config = yaml.safe_load(open('config.yaml', 'r')) 

# LOAD SVG VISUALIZATION CONFIG
svg_visualization_config_file = base_config.get(
        'svg_visualisation_config_file')
if svg_visualization_config_file:
    if not os.path.exists(svg_visualization_config_file):
        svg_visualization_config_file = None
if not svg_visualization_config_file:
    svg_visualization_config_file = DEFAULT_SVG_VISUALIZATION_CONFIG_FILE

svg_visualization_config = yaml.safe_load(
    open(svg_visualization_config_file, 'r'))
default_svg_visualization_config = yaml.safe_load(
    open(DEFAULT_SVG_VISUALIZATION_CONFIG_FILE, 'r'))
for key, default_value in default_svg_visualization_config.items():
    if not hasattr(svg_visualization_config, key):
        svg_visualization_config[key] = default_value



