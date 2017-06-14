# -*- coding: utf-8 -*-
       
class BaseAction:
    """ Represents a basic transport of a substance or solvent."""
    
    def __str__(self):
        return '<BaseAction {}: {}>'.format(self.name, self.rate)
    
    def __call__(self, *args, **kwargs):
        rate = self.rate
        if callable(rate):
            rate = rate(*args, **kwargs)
        if not self.units:
            self.units = rate.units
        return rate
    
