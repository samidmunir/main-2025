import time
from component_1 import generate_environment, scene_to_file, scene_from_file, visualize_scene

# Function test_component_i()
# - this function tests the functionality and features of component_1.py
def test_component_i(num_of_environments: int):
    print(f'\tCreating and visualizing {num_of_environments} environment(s).')
    num_of_obstacles = 5
    
    i = 1
    for environment in range(num_of_environments):
        environment = generate_environment(num_of_obstacles)
        visualize_scene(environment)
        scene_to_file(environment, f'environment_{i}.json')
        time.sleep(1)
        num_of_obstacles += 5
        i += 1

def main():
    print('Running the main function...')
    time.sleep(1)
    
    # Testing component_i
    print('\n\tTesting component_1 -->')
    test_component_i(num_of_environments = 1)
    
    # Testing component_ii (own implementation)
    
    # Testing component_iii (own implementation)
    
    # Testing component_iv (own implmentation)
    
    # Testing component_v (own implementation)
    
    # Testing component_vi (own implementation)
    
    # Testing component_vii (own implementation)
    
    time.sleep(1)
    print('All tests passed!')

if __name__ == "__main__":
    main()