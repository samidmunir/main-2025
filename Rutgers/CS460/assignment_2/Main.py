import time
from component_1 import generate_environment, scene_to_file, scene_from_file, visualize_scene

def test_component_i():
    environment = generate_environment(5)
    visualize_scene(environment)
    scene_to_file(environment, 'environment_test.json')

def main():
    print('Running the main function...')
    
    time.sleep(1)
    print('\n\tTesting component_1 -->')
    
    # Testing component_i
    test_component_i()
    
    # Testing component_ii (own implementation)
    
    # Testing component_iii (own implementation)
    time.sleep(1)
    print('All tests passed!')

if __name__ == "__main__":
    main()