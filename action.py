# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:23:46 2017

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

"""


class BaseAction:
    """Base Class for all Transport and Process Classes."""
    def __init__(self, *args, **kwargs):
        self.units = None

    def __str__(self):
        return '<BaseAction {}: {}>'.format(self.name, self.rate)

    def __hash__(self):
        return hash(repr(self))

    def __call__(self, *args, **kwargs):
        """Instances can be called like functions."""
        rate = self.rate  # Default of rate
        if callable(self.rate):
            rate = self.rate(*args, **kwargs)
        if not self.units:
            self.units = rate.units
        return rate.to_base_units()
