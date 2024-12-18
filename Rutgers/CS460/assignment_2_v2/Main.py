"""
    Assignment #2
    1. Generating environments
        - component_1.py
    2. Nearest neighbors with linear search approach
        - nearest_neighbors.py
    3. Collision checking
        - collision_checking.py
    4. PRM
        - prm.py
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
    
        python3 collision_checking.py --robot arm --map environment_5.tx
        
        python3 collision_checking.py --robot freeBody --map environment_5.txt
        
    4. PRM
    
        python3 prm.py --robot arm --start 0.0 0.0 --goal 3.57 3.57 --map environment_5.txt
        
        python3 prm.py --robot arm --start 0.0 0.0 --goal 3.57 3.57 --map environment_5.txt
        
    5. RRT
    6. PRM*/RRT*
"""

from component_1 import generate_environment, scene_to_file, scene_from_file, visualize_scene
import time

"""
    function handle_environment_creation(number_of_environments: int, initial_number_of_obstacles: int, max_number_of_obstacles: int, number_of_obstacles_step_size: int)
"""
def handle_environment_creation(number_of_environments: int, initial_number_of_obstacles: int, max_number_of_obstacles: int, number_of_obstacles_step_size: int):
    for i in range(number_of_environments):
        number_of_obstacles = initial_number_of_obstacles + (i * number_of_obstacles_step_size)
        # print(f'\nCreating environment #{i + 1}')
        environment = generate_environment(number_of_obstacles = number_of_obstacles)
        scene_to_file(environment = environment, filename = f'environment_{i + 1}.txt')
        # time.sleep(2)
        # visualize_scene(environment = environment)

"""
    function test_component_i()
"""
def test_component_i():
    # print('\ntest_component_i() called.')
    # print('\ttesting component_1.py functions...')
    # time.sleep(2)
    
    # ENVIRONMENT = generate_environment(number_of_obstacles = 5)
    # scene_to_file(environment = ENVIRONMENT, filename = 'environment_test.txt')
    # visualize_scene(environment = ENVIRONMENT)
    
    """
        Call function handle_environment_creation() to create 5 environments with the first environment having 5 obstacles, the maximum obstacles to be 25, and a step size of 5 obstacles.
    """
    handle_environment_creation(number_of_environments = 5, initial_number_of_obstacles = 5, max_number_of_obstacles = 25, number_of_obstacles_step_size = 5)
    
    time.sleep(2) # wait for 2s...
    
    LOADED_ENVIRONMENT = scene_from_file(filename = 'environment_5.txt')
    # print(f'\nLoaded environment:\n\t{LOADED_ENVIRONMENT}')
    visualize_scene(environment = LOADED_ENVIRONMENT)
    
    # print('\n\tcomponent_1.py functions tested successfully.')

def main():
    print('\nAssignment #2: Testing of each part/component.')
    print('-' * 46)
    
    # Testing component_1.py (Generating environments).
    test_component_i()

if __name__ == '__main__':
    main()