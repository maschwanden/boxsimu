# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:57:03 2017

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""

import os
import re
import copy
import importlib
import svgwrite
from svgwrite import cm, mm
import numpy as np

from . import utils as bs_utils


class BoxModelSystemSvgHelper:
    """Helper Class to visualize/plot a BoxModelSystem."""

    def __init__(self):
        self.box_rect_width = 300
        self.box_rect_height = 300
        
        self.system_boxes_arrangement_type = 'circle'
        self.system_boxes_arrangement_radius = None
        self.system_boxes_arrangement_factor = 1.7
        self.system_boxes_arrangement_angle_offset = 0

        self.flow_stroke_width = 4
        self.flow_color = 'darkblue'
        self.flow_arrow_triangle_size = 4
        self.flux_stroke_width = 4
        self.flux_color = 'darkblue'
        self.flux_arrow_triangle_size = 10

        self.box_svg_helpers = None

        self.dwg = None
   
    def save_system_as_svg(self, system, filename):
        """Save the visualization of system as a SVG file."""

        if system.N_boxes == 2:
            self.system_boxes_arrangement_factor = 1.1
        elif system.N_boxes == 3:
            self.system_boxes_arrangement_factor = 1.2
        elif system.N_boxes == 4:
            self.system_boxes_arrangement_factor = 1.4
        elif system.N_boxes == 5:
            self.system_boxes_arrangement_factor = 1.6
        elif system.N_boxes == 6:
            self.system_boxes_arrangement_factor = 1.8

        # self.dwg = svgwrite.Drawing(size=self._get_system_svg_size())
        self.dwg = svgwrite.Drawing(size=('32cm', '10cm'), debug=True)
        self.dwg.viewbox(-100, 0, 600, 400)

        if not self.box_svg_helpers:
            self.box_svg_helpers = self._get_system_box_svg_helpers(system)

        system_svg_group = self.get_system_svg_group(system)
        self._save_group_as_svg(system_svg_group, filename)

    def get_system_svg_group(self, system):
        """Return a SVG representation of the BoxModelSystem instance."""

        if not self.box_svg_helpers:
            self.box_svg_helpers = self._get_system_box_svg_helpers(system)
        group_id = bs_utils.get_valid_svg_id_from_string(system.name)
        group = self.dwg.g(id=group_id)

        for box_svg_helper in self.box_svg_helpers:
            group.add(box_svg_helper.as_svg_group())

        for flow in system.flows:
            group.add(self._get_flow_arrow(flow))
        return group

    def save_box_as_svg(self, box, filename=None):
        """Return a SVG representation of the Box instance."""

        self.dwg = svgwrite.Drawing(size=self._get_box_svg_size())
        self._save_group_as_svg(self.get_box_svg_group(box), filename)

    def get_box_svg_group(self, box):
        """Return the SVG representation of the Box instance."""
        group_id = bs_utils.get_valid_svg_id_from_string(box.name)
        group = self.dwg.g(id=group_id)

        box_svg_helper = self._get_box_svg_helper(box)
        group.add(box_svg_helper.as_svg_group())
        return group

    # HELPER functions

    def _save_group_as_svg(self, group, filename):
        """Save a svgwrite group instance as a SVG file."""

        # dwg = svgwrite.Drawing(filename=filename)
        dwg = copy.deepcopy(self.dwg)
        dwg.filename = filename
        dwg.add(group)
        dwg.save()

    def _get_system_box_svg_helpers(self, system):
        """Return a list of BoxSvgHelper for all boxes of the system."""

        box_positions = self._get_box_positions(system.N_boxes)
        box_svg_helpers = [None] * system.N_boxes
        for box_name, box in system.boxes.items():
            x, y = box_positions[box.id] 
            tmp_box_svg_helper = self._get_box_svg_helper(box, x, y) 
            box_svg_helpers[box.id] = tmp_box_svg_helper
        box_svg_helpers = self._adjust_box_svg_helper_widths(box_svg_helpers)
        return box_svg_helpers

    def _get_box_svg_helper(self, box, x=0, y=0):
        box_group_id = '{}_box'.format(box.name)
        box_svg_helper =  BoxSvgHelper(
                group_id=box_group_id,
                x=x, y=y,
                width=self.box_rect_width, 
                height=self.box_rect_height, 
                text_lines=[
                    'Fluid: {}'.format(box.fluid.name),
                    'Mass: {:.3e}'.format(box.mass),
                ],
                title=box.name_long,
        )
        procsses_group_id = '{}_processes'.format(box.name)
        box_process_names = [p.name for p in box.processes]
        while len(box_process_names) < 3:
            box_process_names.append('')
        processes = box_svg_helper.add_child(
                group_id=procsses_group_id, 
                text_lines=box_process_names,
                title='Processes',
        )
        reaction_group_id = '{}_reactions'.format(box.name)
        box_reaction_names = [p.name for p in box.reactions]
        while len(box_reaction_names) < 3:
            box_reaction_names.append('')
        reactions = box_svg_helper.add_child(
                group_id=reaction_group_id, 
                text_lines=box_reaction_names,
                title='Reactions',
        )
        return box_svg_helper

    def _get_box_positions(self, N_nodes):
        positions = []
        angle_offset = self.system_boxes_arrangement_angle_offset
        radius = self.system_boxes_arrangement_radius
        if not radius:
            radius_factor = self.system_boxes_arrangement_factor
            radius = radius_factor * max(self.box_rect_width, 
                    self.box_rect_height)

        for i in range(N_nodes):
            if self.system_boxes_arrangement_type == 'half_circle':
                angle = (i * np.pi / (N_nodes-1)) + angle_offset
            else: # if self.system_boxes_arrangement_type == 'circle':
                angle = (i * 2 * np.pi / (N_nodes)) + angle_offset
            x = radius * np.cos(angle) 
            y = radius * np.sin(angle)
            positions.append((x,y))
        return positions

    def _adjust_box_svg_helper_widths(self, helpers):
        """Adjust all box_svg_helpers to the same width."""
        max_width = 0
        for helper in helpers:
            if helper.width > max_width:
                max_width = helper.width
        for helper in helpers:
            helper._width = max_width
        return helpers

    def _distance_sort_corners(self, helper, reference_point):
        """Return corners sorted on the distance to a point."""
        reference_point = np.array(reference_point)
        corners = helper.get_box_rect_corner_coordinates()
        np_corners = [np.array(c) for c in corners]
        print('corners', corners)
        distances = [np.linalg.norm(c-reference_point) for c in np_corners]
        sorted_corners = [c for (distance,c) in sorted(zip(distances,corners))]
        return sorted_corners

    def _get_center_between_points(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return (p1 + p2)/2

    def _get_conncection_point_relative_to_reference_point(self, helper, 
            reference_point):
        """Return connection point for flow lines relative to ref point."""
        sorted_corners = self._distance_sort_corners(helper, reference_point)
        p1, p2 = sorted_corners[:2]
        connection_point = self._get_center_between_points(p1, p2)
        return connection_point

    def _get_flow_arrow(self, flow):
        src_point = None
        trg_point = None
        if not flow.source_box:
            helper = self.box_svg_helpers[flow.target_box.id]
            box_center = np.array(
                    helper.get_box_rect_center_cooridnates())
            box_connection_point_to_origin = np.array(
                    self._get_conncection_point_relative_to_reference_point(
                            helper, (0,0)))
            v1 = box_center - box_connection_point_to_origin
            trg_point = box_center + v1
            src_point = trg_point + 0.5 * v1
        elif not flow.target_box:
            helper = self.box_svg_helpers[flow.source_box.id]
            box_center = np.array(
                    helper.get_box_rect_center_cooridnates())
            box_connection_point_to_origin = np.array(
                    self._get_conncection_point_relative_to_reference_point(
                            helper, (0,0)))
            v1 = box_center - box_connection_point_to_origin
            src_point = box_center + v1
            trg_point = src_point + 0.5 * v1
        else:
            src_helper = self.box_svg_helpers[flow.source_box.id]
            trg_helper = self.box_svg_helpers[flow.target_box.id]
            src_point = self._get_conncection_point_relative_to_reference_point(
                            src_helper, (0,0))
            trg_point = self._get_conncection_point_relative_to_reference_point(
                            trg_helper, (0,0))
        
        arrow = self._get_arrow(start=src_point, end=trg_point, 
                stroke_color=self.flow_color, 
                stroke_width=self.flow_stroke_width,
                triangle_size=self.flow_arrow_triangle_size)
        return arrow

    def _get_arrow(self, start, end, stroke_color, stroke_width, 
            triangle_size):
        arrow_vector = end - start
        arrow_unit_vector = arrow_vector / np.linalg.norm(arrow_vector)
        rot90_matrix = self._get_rot90_matrix()
        arrow_unit_normal_vector = np.dot(rot90_matrix, arrow_unit_vector)
        triangle_point1 = triangle_size * arrow_unit_vector
        triangle_point2 = 0.5 * triangle_size * arrow_unit_normal_vector 
        triangle_point3 = -0.5 * triangle_size * arrow_unit_normal_vector 

        end[0] += triangle_size
        arrow = self.dwg.line(start=start, end=end, stroke=stroke_color, 
                stroke_width=stroke_width)
        marker = self.dwg.marker(insert=0.75*arrow_unit_vector*triangle_size, 
                size=(triangle_size, triangle_size))
        marker.add(self.dwg.polygon([triangle_point1, triangle_point2, 
            triangle_point3], fill=stroke_color))
        self.dwg.defs.add(marker)
        arrow.set_markers((None, None, marker))
        return arrow

    def _get_rot90_matrix(self):
        angle = np.deg2rad(90)
        return np.array([[np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])

    def _get_system_svg_size(self):
        return (100, 100)

    def _get_box_svg_size(self):
        return (100, 100)


class BoxSvgHelper:
    def __init__(self, group_id, x, y, width, height=None, 
            text_lines=None, title=None):
        
        text_lines = text_lines or []
        if not height and len(text_lines) == 0:
            raise ValueError('Either height or text_lines must be given.')

        self.group_id = bs_utils.get_valid_svg_id_from_string(group_id)
        self._x = x
        self._y = y
        self._height = height
        self._width = width

        self.text_lines = text_lines
        self.text_font_size = 12
        self.text_font_color = 'black'
        self.text_alignement = 'left'

        self.title = title
        self.title_font_size = 24
        self.title_font_color = 'black'
        self.title_alignement = 'middle'
        
        self.child_title_font_size = 15
        self.child_title_font_color = 'black'
        self.child_title_alignement = 'left'

        self.title_extern = True
        self.child_title_extern = True

        self.color = 'lightgrey'
        self.opacity = 0.7
        self.stroke_color = 'black'
        self.stroke_width = 5
        self.stroke_opacity = 1
    
        self.child_relative_width = 0.925
        self.child_color = 'darkgrey'
        self.child_opacity = 0.5
        self.child_stroke_color = 'white'
        self.child_stroke_width = 3
        self.child_stroke_opacity = 1

        self._content_absolute_margin = 10
        
        # Maximal widht of the character 'W' in the title and text
        self.title_max_W_width = self.title_font_size
        self.text_max_W_width = self.text_font_size

        self.title_avg_char_width = 0.8 * self.title_max_W_width
        self.text_avg_char_width = 0.8 * self.text_max_W_width

        self.children = []

        self.dwg = svgwrite.Drawing()
        self.group = self.dwg.g(id=group_id)

    @property
    def width(self):
        """Return width of the instance."""
        width = self._width
        max_title_width = self.get_max_title_width()
        max_text_width = self.get_max_text_width()
        max_children_width = self.get_max_children_width()
        if max_title_width > width:
            width = max_title_width
        if max_text_width > width:
            width = max_text_width
        if max_children_width > width:
            width = max_children_width
        self._width = width
        self._adjust_children_width()
        return self._width

    @property
    def height(self):
        """Return height of the instance."""
        height = 0
        if self._height:
            height = self._height
        element_height = (self.get_text_height() + self.get_title_height() + 
                self.get_children_height())
        if element_height > height:
            height = element_height

        return height

    @property
    def x(self):
        """Return left edge of the instance."""
        return self._x

    @property
    def y(self):
        """Return top edge of the instance."""
        return self._y

    @property
    def content_absolute_margin(self):
        if not self._content_absolute_margin:
            width_relative = self.child_relative_width
            self._content_absolute_margin = ((1-width_relative)/2 * self._width)
        return self._content_absolute_margin

    # PUBLIC functions

    def as_svg_group(self):
        """Return the SVG representation of the instance."""
        self._adjust_children_width()
        self.group.add(self._get_svg_rect_element())
        title = self._get_svg_title_element()
        if title:
            self.group.add(title)
        text = self._get_svg_text_element()
        if text:
            self.group.add(text)
        children = self._get_svg_children_element()
        if children:
            self.group.add(children)
        return self.group

    def add_child(self, group_id, text_lines=None, height=None, 
            width_relative=None, title=None):
        """Add a child instance."""

        text_lines = text_lines or []
        if not height and len(text_lines) == 0:
            raise ValueError('Either height or text_lines must be given.')

        width_relative = self.child_relative_width
        x = self.x + self.content_absolute_margin
        y = self.get_children_bottom_y() 
        width = width_relative * self.width

        child = self.__class__(group_id, x, y, width, 
                height=height, text_lines=text_lines, title=title)

        child.title_extern = self.child_title_extern
        child.color = self.child_color        
        child.opacity = self.child_opacity
        child.stroke_color = self.child_stroke_color
        child.stroke_width = self.child_stroke_width
        child.stroke_opacity = self.child_stroke_opacity
        child.title_font_size = self.child_title_font_size 
        child.title_font_color = self.child_title_font_color
        child.title_alignement = self.child_title_alignement

        self.children.append(child)
        return child

    # RECT info functions

    def get_rect_height(self):
        """Return height of the rect element.""" 
        height = self.height
        if self.title_extern:
            height = self.height - self.get_title_height()
        return height

    def get_rect_top_y(self):
        """Return upper edge of the rect element."""
        y = self.y
        if self.title_extern:
            y = self.get_title_bottom_y()
        return y 
    
    def get_rect_bottom_y(self):
        """Return bottom edge of the rect element."""
        y = self.get_rect_top_y() + self.get_rect_height()
        return y

    # TITLE info functions

    def get_max_title_width(self):
        """Return approx. maximal width (px) of title text."""
        max_width = 0
        if self.title:
            max_width = len(self.title.strip()) * self.title_avg_char_width
        return max_width

    def get_title_height(self):
        """Return total height (with margins) of the title element.""" 
        height = 0
        if self.title:
            height = 1.5 * self.title_font_size
        return height

    def get_title_top_y(self):
        """Return upper edge of title."""
        y = self.y 
        return y 
    
    def get_title_bottom_y(self):
        """Return bottom edge of title."""
        y = self.get_title_top_y() + self.get_title_height()
        return y

    # TEXT info functions

    def get_max_text_width(self):
        """Return approx. maximal width (px) of all text lines."""
        max_width = 0
        if self.text_lines:
            for text in self.text_lines:
                tmp_width = len(text.strip()) * self.text_avg_char_width
                if tmp_width > max_width:
                    max_width = tmp_width
        return max_width

    def get_text_height(self):
        """Return total height (with margins) of the text lines.""" 
        height = 0
        if self.text_lines:
            height = ((len(self.text_lines) * 1.5 + 0.5) * 
                    self.text_font_size)
        return height

    def get_text_top_y(self):
        """Return upper edge of text lines."""
        y = self.get_title_bottom_y()
        return y

    def get_text_bottom_y(self):
        """Return bottom edge of text lines."""
        y = self.get_text_top_y() + self.get_text_height()
        return y

    # CHILD info functions

    def get_max_children_width(self):
        """Return approx. maximal width (px) of the all children."""
        max_width = 0
        if self.children:
            for boxrect in self.children:
                boxrect_width = boxrect.width
                needed_width = boxrect_width + 2 * self.content_absolute_margin
                if needed_width > max_width:
                    max_width = needed_width
        return max_width

    def get_children_height(self):
        """Return total height (with margins) of all children.""" 
        height = 0
        if self.children:
            for rect in self.children:
                # increase children height by the height of the child_rect plus 
                # a margin equal to the text_font_size
                height += rect.height + self.text_font_size
        return height

    def get_children_top_y(self):
        """Return upper edge of children."""
        y = self.get_text_bottom_y()
        return y

    def get_children_bottom_y(self):
        """Return bottom edge of children."""
        y = self.get_children_top_y() + self.get_children_height()
        return y

    def get_box_rect_corner_coordinates(self):
        """Return coordinates of corners of the rect-element of the instance.
        
        Return coordinates as a list of tuples begining with the top left, 
        followed by the top right, bottom right, and bottom left corner.

        Return:
            corner_coords (list of tuple of floats): 
                [(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)] 

        """
        # top left corner
        tl = (self.x, self.y)
        tr = (self.x + self.width, self.y)
        br = (self.x + self.width, self.y + self.height)
        bl = (self.x, self.y + self.height)
        return [tl, tr, br, bl]

    def get_box_rect_center_cooridnates(self):
        """Return coordinates of the center of the rect-element."""
        return (self.x + 0.5 * self.width, self.y + 0.5 * self.height)

    # HELPER functions

    def _get_svg_rect_element(self):
        """Return a rect svg element of the instance."""
        rect_id = '{}_rect'.format(self.group_id)
        y = self.get_rect_top_y()
        height = self.get_rect_height()
        return self._rect(self.x, y, self.width, height, rect_id)

    def _get_svg_title_element(self):
        """Return a text svg element of the instance's title."""
        if self.title:
            if self.title_alignement == 'middle':
                x = self.x + 0.5 * self.width
            elif self.title_alignement == 'left':
                if self.title_extern:
                    x = self.x
                else:
                    x = self.x + self.content_absolute_margin
            else:
                raise ValueError('title_alignement must be "middle" or "left".')

            y = self.get_title_top_y() + 0.5 * self.get_title_height()
            title_id = '{}_title'.format(self.group_id)
            return self._title(self.title.strip(), x, y, title_id)

    def _get_svg_text_element(self):
        """Return a svg group with all text lines as text svg elements."""
        if self.text_lines:
            if self.text_alignement == 'middle':
                x = self.x + 0.5 * self.width
            elif self.text_alignement == 'left':
                x = self.x + self.content_absolute_margin
            else:
                raise ValueError('text_alignement must be "middle" or "left".')

            text_group = self.dwg.g(id='{}_text'.format(self.group_id))
            rel_text_y = 1.0 / (len(self.text_lines) + 1)
            for i, text in enumerate(self.text_lines):
                rel_pos = (i + 1) * rel_text_y
                y = self.get_text_top_y() + rel_pos * self.get_text_height()
                text_group.add(self._text(text.strip(), x, y))
            return text_group

    def _get_svg_children_element(self):
        """Return the complete svg-representation of all children."""
        children_group = self.dwg.g(id='{}_children'.format(self.group_id))
        for child_rect in self.children:
            children_group.add(child_rect.as_svg_group())
        return children_group 
    
    def _rect(self, x, y, width, height, rect_id=None):
        return self.dwg.rect(
            insert=(x, y), 
            size=(width, height),
            fill=self.color,
            opacity=self.opacity,
            stroke=self.stroke_color,
            stroke_width=self.stroke_width,
            stroke_opacity=self.stroke_opacity,
            id=rect_id
        )

    def _title(self, string, x, y, title_id=None):
        style_template = 'text-anchor:{}; dominant-baseline:mathematical'
        style = style_template.format(self.title_alignement)
        return self.dwg.text(
                string,
                insert=(x,y),
                fill=self.title_font_color,
                font_size=self.title_font_size,
                style=style,
                id=title_id,
        )

    def _text(self, string, x, y, text_id=None):
        style_template = 'text-anchor:{}; dominant-baseline:mathematical'
        style = style_template.format(self.text_alignement)
        return self.dwg.text(
                string,
                insert=(x,y),
                fill=self.text_font_color,
                font_size=self.text_font_size,
                style=style,
                id=text_id,
        )

    def _adjust_children_width(self):
        """Correct/Adjust the width and x-pos of all child boxrects.
        
        Due to the dynamic width/height of the master-box, child boxes that
        are generated at different times can differ in their width. That's
        why before the final svg element is generated, all width and x-pos
        of the child boxrects are corrected first.
        
        """
        width = copy.copy(self._width)
        child_width = width - 2 * self.content_absolute_margin
        child_x = self.x + self.content_absolute_margin

        for boxrect in self.children:
            boxrect._width = child_width
            boxrect._x = child_x
            boxrect._content_absolute_margin = self.content_absolute_margin
            boxrect._adjust_children_width()

