# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:57:03 2017

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import os
import re
import yaml
import svgwrite
from svgwrite import cm, mm
import numpy as np

from . import utils as bs_utils
from .config import svg_visualization_config as svg_config



class SvgVisualization:
    """Create SVG visualizations of BoxModelSystem objects."""

    def __init__(self, verbose_names=False):
        self.verbose_names = verbose_names
        self.dwg = svgwrite.Drawing()

    def save_group_as_svg(self, group, filename):
        dwg = svgwrite.Drawing(filename=filename)
        dwg.add(group)
        dwg.save()

    # SVG FACTORY FUNCTIONS

    def system_as_svg(self, system, filename=None):
        if filename is None:
            filename = bs_utils.get_valid_filename_from_string(system.name)
        if '.' not in filename:
            filename += '.svg'
        self.save_group_as_svg(self.system_as_svg_group(system), filename)

    def box_as_svg(self, box, filename=None):
        if filename is None:
            filename = bs_utils.get_valid_svg_id_from_string(box.name)
        if '.' not in filename:
            filename += '.svg'
        self.save_group_as_svg(self.box_as_svg_group(box), filename)

    # GROUP FACTORY FUNCTIONS

    def system_as_svg_group(self, system):
        group = self.dwg.g(id='boxmodelsystem')
        group.add(self.system_boxes_as_svg_group(system))
        group.add(self.system_flows_as_svg_group(system))
        group.add(self.system_fluxes_as_svg_group(system))
        group.add(self.system_info_legend_as_svg_group(system))
        return group

    def system_boxes_as_svg_group(self, system):
        group = self.dwg.g(id='boxes')

        box_positions = self._get_box_positions(system.N_boxes)

        i = 0
        for box_name, box in system.boxes.items():
            box_group = self.box_as_svg_group(box, box_positions[i])
            group.add(box_group)
            i += 1
        return group

    def system_flows_as_svg_group(self, system):
        group = self.dwg.g(id='flows')
        return group

    def system_fluxes_as_svg_group(self, system):
        group = self.dwg.g(id='fluxes')
        return group

    def system_info_legend_as_svg_group(self, system):
        group = self.dwg.g(id='system_info_legend')
        return group

    def box_as_svg_group(self, box, box_pos):
        group = self.dwg.g(id='box_{}'.format(box.name))

        group.add(self.get_box_rect(box_pos))
        group.add(self.get_box_title(box_pos, box.name_long))

        legend_rel_y_pos = (svg_config['box_title_rel_pos_vertical'] +
                svg_config['box_subrect_top_margin_relative'])
        group.add(self.get_box_subrect_with_text(
            '{}_box_info'.format(box.name), box_pos, 
            legend_rel_y_pos, 
            ['Fluid: {}'.format(box.fluid.name), 
             'Fluid mass [kg]: {:.2e}'.format(box.fluid.mass.magnitude),
             ]))
         
        group.add(self.box_processes_as_svg_group(box))
        group.add(self.box_reactions_as_svg_group(box))
        return group

    def get_box_rect(self, box_pos):
        return self.dwg.rect(
            insert=box_pos, 
            size=(svg_config['box_rect_width'],
                svg_config['box_rect_height']),
            fill=svg_config['box_rect_color'],
            opacity=svg_config['box_rect_opacity'],
            stroke=svg_config['box_rect_stroke_color'],
            stroke_width=svg_config['box_rect_stroke_width'],
            stroke_opacity=svg_config['box_rect_stroke_opacity'],
        )

    def get_box_title(self, box_pos, string):
        x = box_pos[0] + (svg_config['box_rect_width'] * 
                svg_config['box_title_rel_pos_horizontal'])
        y = box_pos[1] + (svg_config['box_rect_height'] * 
                svg_config['box_title_rel_pos_vertical'])
        return self._get_text([x, y], string, text_type='title') 

    def get_box_subrect_with_text(self, grp_id, box_pos, rel_y_pos, 
            text_lines):
        group = self.dwg.g(id=grp_id)
        box_rect_height = svg_config['box_rect_height']
        box_rect_width = svg_config['box_rect_width']
        rel_width = svg_config['box_subrect_width_relative']
        rel_horizontal_padding = (1-rel_width)/2.0

        x = box_pos[0] + rel_horizontal_padding * box_rect_width
        y = box_pos[1] + rel_y_pos * box_rect_height
        width = rel_width * box_rect_width
        N_lines = len(text_lines)
        height = (N_lines * 1.5 + 0.5) * svg_config['box_info_font_size']
        box_subrect = self.dwg.rect(
            insert=(x,y), 
            size=(width, height),
            fill=svg_config['box_subrect_color'],
            opacity=svg_config['box_subrect_opacity'],
            stroke=svg_config['box_subrect_stroke_color'],
            stroke_width=svg_config['box_subrect_stroke_width'],
            stroke_opacity=svg_config['box_subrect_stroke_opacity'],
        )
        group.add(box_subrect)
        
        rel_text_y = 1.0 / (N_lines + 1)
        for i, text in enumerate(text_lines):
            x_line = x + 0.05 * width
            rel_pos = (i + 1) * rel_text_y
            y_line = y + rel_pos * height
            group.add(self._get_text([x_line, y_line], text, 
                horizontal_alignment='left'))
        return group




#         x_text_fluid = x_legend + 0.05 * width_legend
#         y_text_fluid = y_legend + 0.333 * height_legend
#         text_fluid = self.dwg.text('Fluid: {}'.format(box.fluid.name),
#                 insert=(x_text_fluid, y_text_fluid),
#                 fill=svg_config['box_info_font_color'],
#                 font_size=svg_config['box_info_font_size'],
#                 style='text-anchor:left; dominant-baseline:mathematical'
#         )
#         group.add(text_fluid)
#         
#         x_text_mass = x_legend + 0.05 * width_legend
#         y_text_mass = y_legend + 0.666 * height_legend
#         box_mass_text = self.dwg.text('Box Mass [kg]: {:.2e}'.format(
#             box.fluid.mass.to_base_units().magnitude),
#                 insert=(x_text_mass, y_text_mass),
#                 fill=svg_config['box_info_font_color'],
#                 font_size=svg_config['box_info_font_size'],
#                 style='text-anchor:left; dominant-baseline:mathematical'
#         )
#         group.add(box_mass_text)
# 

    def box_processes_as_svg_group(self, box):
        group = self.dwg.g(id='{}_processes'.format(box.name))
        for process in box.processes:
            group.add(self.process_as_svg_group(process))
        return group

    def box_reactions_as_svg_group(self, box):
        group = self.dwg.g(id='{}_reactions'.format(box.name))
        for reaction in box.reactions:
            group.add(self.reaction_as_svg_group(reaction))
        return group


    def process_as_svg_group(self, process):
        group = self.dwg.g(id=bs_utils.get_valid_svg_id_from_string(
            process.name))
        return group

    def reaction_as_svg_group(self, reaction):
        group = self.dwg.g(id=bs_utils.get_valid_svg_id_from_string(
            reaction.name))
        return group









    # HELPER FUNCTIONS

    def _get_box_positions(self, N_nodes):
        positions = []
        box_rect_width = svg_config['box_rect_width']
        box_rect_height = svg_config['box_rect_height']
        angle_offset = svg_config['boxes_arrangement_angle_offset']
        boxes_arrangement_type = svg_config['boxes_arrangement_type']

        if boxes_arrangement_type == 'circle':
            radius = svg_config['box_circle_radius']
            if not radius:
                radius_factor = svg_config['box_circle_radius_factor']
                radius = radius_factor * max(box_rect_width, box_rect_height)

            for i in range(N_nodes):
                angle = (i * 2 * np.pi / (N_nodes)) + angle_offset
                x = radius * np.cos(angle) 
                y = radius * np.sin(angle)
                positions.append((x,y))

        elif boxes_arrangement_type == 'half_circle':
            radius = svg_config['box_half_circle_radius']
            if not radius:
                radius_factor = svg_config['box_half_circle_radius_factor']
                radius = radius_factor * max(box_rect_width, box_rect_height)

            for i in range(N_nodes):
                angle = (i * np.pi / (N_nodes-1)) + angle_offset
                x = radius * np.cos(angle) 
                y = radius * np.sin(angle)
                positions.append((x,y))
        
        return positions



    def _get_text(self, text_pos, string, 
            horizontal_alignment='middle', text_type='info'):
        style = 'text-anchor:{}; dominant-baseline:mathematical'.format(
            horizontal_alignment)
        if text_type == 'info':
            font_size = svg_config['box_info_font_size']
        elif text_type == 'title':
            font_size = svg_config['box_title_font_size']

        return self.dwg.text(string,
                insert=text_pos,
                fill=svg_config['box_title_font_color'],
                font_size=font_size,
                style=style
        )


