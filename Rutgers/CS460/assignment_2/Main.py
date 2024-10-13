import time
from component_1 import generate_environment, scene_to_file, scene_from_file, visualize_scene

def test_component_1():
    # Generate a test environment with 10 obstacles.
    print('\tgenerating test environment with 10 obstacles.')
    ENVIRONMENT = generate_environment(10);
    
    # Save the ENVIRONMENT to a file.
    print('\tsaving environment to environment_test.txt')
    scene_to_file(ENVIRONMENT, 'environment_test.txt')
    
    # Load the saved ENVIRONMENT from the file.
    print('\tloading environment from environment_test.txt')
    LOADED_ENVIRONMENT = scene_from_file('environment_test.txt')
    
    # Visualize the loaded ENVIRONMENT.
    print('\tvisualizing loaded environment.')
    visualize_scene(LOADED_ENVIRONMENT)

def main():
    # Testing component_1.py
    print(f'Testing component_1.py...')
    time.sleep(1)
    test_component_1()
    time.sleep(1)
    print(f'[COMPONENT_1] - All tests passed.')
    

if __name__ == "__main__":
    main()