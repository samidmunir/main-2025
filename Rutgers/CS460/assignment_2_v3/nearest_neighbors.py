"""
    2. Nearest neighbors with linear search approach: nearest_neighbors.py
        - main()
        - parse_arguments()
"""

# IMPORTS
import argparse as ARGPRS
import math as MATH
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM
import time as TIME

# CONSTANTS
from component_1 import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    OBSTACLE_MIN_SIZE,
    OBSTACLE_MAX_SIZE    
)

"""
    function parse_arguments()
    - function to parse command-line arguments.
    - return arguments object as ARGS
"""
def parse_arguments():
    PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    PARSER.add_argument('--target', type = float, nargs = '+', required = True, help = 'Target pose of robot. (N = 2 for arm, N = 3 for freeBody).')
    
    PARSER.add_argument('-k', type = int, required = True, default = 3, help = 'Number of nearest neighbors to consider.')
    
    PARSER.add_argument('--configs', type = str, required = True, help = 'Filename containg list of arm/freeBody configurations.')
    
    return PARSER.parse_args()

"""
    function main():
    - Main function to run the program.
"""
def main():
    print('\n2. Nearest neighbors with linear search approach\n')
    
    ARGS = parse_arguments()

if __name__ == '__main__':
    main()