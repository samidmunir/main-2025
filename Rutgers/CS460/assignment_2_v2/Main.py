"""
    Assignment #2
    1. Generating environments
    2. Nearest neighbors with linear search approach
    3. Collision checking
    4. PRM
    5. RRT
    6. PRM*/RRT*
"""
"""
    Scripts to test each part/component.
    1. Generating environments
        - Main script will handle creating a component_1.py object and using its functions to generate environments and save them to respective files.
    2. Nearest neighbors with linear search approach
        python3 nearest_neighbors.py --robot arm --target 0.0 0.0 -k 3 --configs configs.txt
        
        python3 nearest_neighbors.py --robot freeBody --target 10.0 10.0 360.0 -k 3 --configs configs.txt --save_nearest_neighbors
    3. Collision checking
    4. PRM
    5. RRT
    6. PRM*/RRT*
"""

from component_1 import generate_environment, scene_to_file, scene_from_file
import time

"""
    function test_component_i()
"""
def test_component_i():
    print('\ntest_component_i() called.')
    print('\ttesting component_1.py functions...')
    time.sleep(2)
    
    ENVIRONMENT = generate_environment(number_of_obstacles = 5)
    scene_to_file(environment = ENVIRONMENT, filename = 'environment_test.txt')
    
    LOADED_ENVIRONMENT = scene_from_file(filename = 'environment_test.txt')
    print(f'\nLoaded environment:\n\t{LOADED_ENVIRONMENT}')
    
    print('\n\tcomponent_1.py functions tested successfully.')

def main():
    print('\nAssignment #2: Testing of each part/component.')
    print('-' * 46)
    
    # Testing component_1.py (Generating environments).
    test_component_i()

if __name__ == '__main__':
    main()