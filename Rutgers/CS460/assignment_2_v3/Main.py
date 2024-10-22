"""
    Sami Munir - sm2246
    CS460
    Assignment #2
    
    1. Generating environments
        - component_1.py
    
    2. Nearest neighbors with linear search approach
        - nearest_neighbors.py
    
    3. Collision checking
    
    4. PRM - Probabalistic road map
    
    5. RRT
    
    6. PRM* / RRT*
"""

# IMPORTS
import time

from component_1 import (
    generate_environment,
    scene_to_file,
    scene_from_file,
    visualize_scene
)

"""
    function test_component_i():
"""
def test_component_i(number_of_environments: int, initial_number_of_obstacles: int, number_of_obstacles_step_size: int):
    print('\ntest_component_i() called...')
    
    number_of_obstacles = initial_number_of_obstacles
    
    for i in range(number_of_environments):
        ENVIRONMENT = generate_environment(number_of_obstacles = number_of_obstacles)
        
        scene_to_file(environment = ENVIRONMENT, filename = f'environment_{(i + 1)}_{number_of_obstacles}.txt')
        
        visualize_scene(environment = ENVIRONMENT)
        time.sleep(1)
        
        number_of_obstacles += number_of_obstacles_step_size
    
    # LOADED_ENVIRONMENT = scene_from_file(filename = 'environment_1_10.txt')
    # visualize_scene(environment = LOADED_ENVIRONMENT)

"""
    function main():
    - Main function to run the program.
    - calls testing functions for component_1.py (generating environments)
"""
def main():
    print('\nAssignment #2\n')
    
    # calling function test_component_1()
    test_component_i(number_of_environments = 5, initial_number_of_obstacles = 10, number_of_obstacles_step_size = 5)

if __name__ == '__main__':
    main()