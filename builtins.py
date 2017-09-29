# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 2017 at 14:20

@author: Mathias Aschwanden (mathias.aschwanden@gmail.com)

Makes various Variables available to the user.

IMPORTANT: All data has been included without warranty, express or implied. 

References:
    Molar Masses : From Wikipedia.org

"""


from . import ur
from . import entities as bs_entities


# VARIABLES
po4 = bs_entities.Variabl('PO4', molar_mass=94.9714*ur.gram/ur.mole)
PO4 = phosphate = po4
no3 = bs_entities.Variabl('NO3', molar_mass=62.00*ur.gram/ur.mole)
NO3 = nitrate = no3


# PROCESSES


