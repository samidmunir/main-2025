"""
    2. Nearest neighbors with linear search approach: nearest_neighbors.py
        - main() -> None:
        - parse_arguments() -> dict:
"""

# IMPORTS
import argparse as ARGPRS
import math as MATH
import numpy as NP

from utils import (
    load_sample_arm_robot_configurations,
    get_k_nearest_arm_robot_configurations
)

"""
    function parse_arguments() -> dict:
"""
def parse_arguments() -> dict:
    PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    PARSER.add_argument('--target', type = float, nargs = '+', required = True, help = 'Target pose of robot. (N = 2 for arm, N = 3 for freeBody).')
    
    PARSER.add_argument('-k', type = int, required = True, default = 3, help = 'Number of nearest neighbors to consider.')
    
    PARSER.add_argument('--configs', type = str, required = True, help = 'Filename containg list of arm/freeBody configurations.')
    
    return PARSER.parse_args()

"""
    function main() -> None:
"""
def main() -> None:
    print('\n2. Nearest neighbors with linear search approach\n')
    
    ARGS = parse_arguments()
    
    if ARGS.robot == 'arm':
        CONFIGURATIONS = load_sample_arm_robot_configurations(filename = ARGS.configs)
        K_NEAREST_CONFIGURATIONS = get_k_nearest_arm_robot_configurations(configurations = CONFIGURATIONS, target_configuration = ARGS.target, k = ARGS.k)
    elif ARGS.robot == 'freeBody':
        print('*** Not yet supported ***')

if __name__ == '__main__':
    main()