"""
    2. Nearest neighbors with linear search approach: nearest_neighbors.py
        - main() -> None:
        - parse_arguments() -> dict:
"""

# IMPORTS
import argparse as ARGPRS
import math as MATH
import numpy as NP

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
        print('*** Not yet supported ***')
    elif ARGS.robot == 'freeBody':
        print('*** Not yet supported ***')

if __name__ == '__main__':
    main()