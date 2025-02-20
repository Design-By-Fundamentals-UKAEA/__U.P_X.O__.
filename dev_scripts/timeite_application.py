# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:58:13 2024

@author: rg5749
"""

import timeit
from sympy import Polygon, Point as SymPoint
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon
import numpy as np

# Define the line
line = np.array([(0, 0), (0, 1), (1, 1), (1, 0)])

# Define the point
point = (0.5, 0.5)

# Define the setup code for SymPy
sympy_setup = """
from sympy import Polygon, Point as SymPoint
line = [(0, 0), (0, 1), (1, 1), (1, 0)]
rect = Polygon(*line)
point = SymPoint(0.5, 0.5)
"""

# Define the SymPy code
sympy_code = """
result = rect.encloses_point(point)
"""

# Define the setup code for Shapely
shapely_setup = """
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon
line = [(0, 0), (0, 1), (1, 1), (1, 0)]
rect = ShapelyPolygon(line)
point = ShapelyPoint(0.5, 0.5)
"""

# Define the Shapely code
shapely_code = """
result = rect.contains(point)
"""

# Measure the execution time for SymPy
sympy_time = timeit.timeit(stmt=sympy_code, setup=sympy_setup, number=1000)

# Measure the execution time for Shapely
shapely_time = timeit.timeit(stmt=shapely_code, setup=shapely_setup, number=1000)

print("Execution time for SymPy:", sympy_time)
print("Execution time for Shapely:", shapely_time)
