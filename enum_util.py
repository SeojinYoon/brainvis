# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:21:00 2022

@author: Seojin
"""
from enum import Enum
    
class File_validation(Enum):
    exist = 1 << 0
    only = 1 << 1
    
    exist_only = exist | only
