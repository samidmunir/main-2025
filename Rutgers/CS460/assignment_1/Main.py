import numpy as np
from component_1 import check_SOn, check_quaternion, check_SEn
from component_2 import random_rotation_matrix
from component_3 import interpolate_rigid_body, forward_propagate_rigid_body, visualize_path
from component_4 import interpolate_arm, forward_propagate_arm, visualize_arm_path

def test_component_i():
    print('\ntest_component_i() called -->')
    
    def test_check_SOn():
        print('\ttesting function check_SOn()...\n\t[TEST CASES: 5]')
        
        # Valid 2 x 2 rotation matrix (90 degrees).
        R_2x2 = np.array([[0, -1], [1, 0]])
        assert check_SOn(R_2x2) == True, 'Test Case 1 Failed!'
        
        # Non-square matrix.
        non_square_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        assert check_SOn(non_square_matrix) == False, 'Test Case 2 Failed!'
        
        # Non-orthogonal square matrix.
        non_orthogonal_matrix = np.array([[1, 2], [3, 4]])
        assert check_SOn(non_orthogonal_matrix) == False, 'Test Case 3 Failed!'
        
        # Valid 3 x 3 rotation matrix.
        R_3x3 = np.eye(3)
        assert check_SOn(R_3x3) == True, 'Test Case 4 Failed'
        
        # Matrix with determinant not equal to 1.
        matrix_det_not_1 = np.array([[2, 0], [0, 2]])
        assert check_SOn(matrix_det_not_1) == False, 'Test Case 5 Failed!'
    
    def test_check_quaternion():
        print('\n\ttesting function check_quaternion()...\n\t[TEST CASES: 3]')
        
        # Valid unit quaternion.
        q_valid = np.array([1, 0, 0, 0])
        assert check_quaternion(q_valid) == True, 'Test Case 1 Failed!'
        
        # Non-normalized quaternion.
        q_non_normalized = np.array([2, 0, 0, 0])
        assert check_quaternion(q_non_normalized) == False, 'Test Case 2 Failed!'
        
        # Vector with length != 4.
        q_invalid_length = np.array([1, 0, 0])
        assert check_quaternion(q_invalid_length) == False, 'Test Case 3 Failed!'
        
    def test_check_SEn():
        print('\n\ttesting function check_SEn()...\n\t[TEST CASES: 4]')
        
        # Valid SE(2) matrix.
        SE2_matrix = np.array([
            [0, -1, 1],
            [1, 0, 2],
            [0, 0, 1]
        ])
        assert check_SEn(SE2_matrix) == True, 'Test Case 1 Failed!'
        
        # Valid SE(3) matrix.
        SE3_matrix = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])
        assert check_SEn(SE3_matrix) == True, 'Test Case 2 Failed!'
        
        # Non-square matrix.
        non_square_matrix = np.array([
            [1, 0, 1],
            [0, 1, 2]
        ])
        assert check_SEn(non_square_matrix) == False, 'Test Case 3 Failed!'
        
        # SE(3) matrix with non-orthogonal rotation part.
        non_orthogonal_SE3 = np.array([
            [2, 0, 0, 1],
            [0, 2, 0, 2],
            [0, 0, 2, 3],
            [0, 0, 0, 1]
        ])
        assert check_SEn(non_orthogonal_SE3) == False, 'Test Case 4 Failed!'
    
    test_check_SOn()
    test_check_quaternion()
    test_check_SEn()
    
def test_component_ii():
    print('\ntest_component_ii() called -->')
    
    def test_random_rotation_matrix_naive_false():
        print('\ttesting function random_rotation_matrix(naive = False)')
        rand_rotation_matrix = random_rotation_matrix(False)
        print('\n ***')
        print(rand_rotation_matrix)
        print('***' )
    
    def test_random_rotation_matrix_naive_true():
        print('\ttesting function random_rotation_matrix(naive = True)')
        rand_rotation_matrix = random_rotation_matrix(True)
        print('\n ***')
        print(rand_rotation_matrix)
        print('***')
    
    test_random_rotation_matrix_naive_false()
    test_random_rotation_matrix_naive_true()

def test_component_iii():
    print('\ntest_component_iii() called -->')
    
    start_pose = np.array([0, 0, 0])
    goal_pose = np.array([5, 5, np.pi / 2])
    
    interpolated_path = interpolate_rigid_body(start_pose, goal_pose)
    visualize_path(interpolated_path)
    print('\ninterpolated_path:\n', interpolated_path)
    
    # start_pose = np.array([0.0, 0.0, 0.0])
    # plan = [
    #     (np.array([1.0, 0.0, 0.0]), 0.25),
    #     (np.array([1.0, 1.0, np.pi / 4]), 0.25),
    #     (np.array([-2.0, 2.0, np.pi]), 0.25),
    #     (np.array([-1.0, -1.0, np.pi / 2]), 0.25),
    #     (np.array([0.0, -2.0, -(np.pi / 2)]), 0.25)
    # ]
    # start_pose = np.array([-10.0, -10.0, 0.0])
    start_pose = np.array([0.0, 0.0, 0.0])
    # plan = [
    #     (np.array([1.0, 1.0, np.pi / 4]), 0.25),
    #     (np.array([0.0, 1.0, np.pi / 4]), 0.25),
    #     (np.array([-1.0, 0.0, np.pi / 4]), 0.25),
    #     (np.array([0.0, -1.0, np.pi / 4]), 0.25),
    #     (np.array([0.0, -1.0, 0.0]), 0.25)
    # ]
    plan = [
        (np.array([1.0, 0.0, np.pi / 4]), 0.05),
        (np.array([1.0, 0.1, np.pi / 4]), 0.05),
        (np.array([1.0, 0.2, np.pi / 4]), 0.05),
        (np.array([1.0, 0.3, np.pi / 4]), 0.05),
        (np.array([1.0, 0.4, np.pi / 4]), 0.05),
        (np.array([1.0, 0.5, np.pi / 4]), 0.05),
        (np.array([1.0, 0.6, np.pi / 4]), 0.05),
        (np.array([1.0, 0.7, np.pi / 4]), 0.05),
        (np.array([1.0, 0.8, np.pi / 4]), 0.05),
        (np.array([1.0, 0.9, np.pi / 4]), 0.05),
        (np.array([0.0, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.1, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.2, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.3, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.4, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.5, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.6, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.7, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.8, 1.0, np.pi / 4]), 0.05),
        (np.array([-0.9, 1.0, np.pi / 4]), 0.05),
        (np.array([-1.0, 0.0, np.pi / 4]), 0.30),
        (np.array([0.0, -1.0, np.pi / 4]), 0.30),
        (np.array([1.0, 0.0, np.pi / 4]), 0.30)
    ]
    
    forward_propagated_path = forward_propagate_rigid_body(start_pose, plan)
    print(forward_propagated_path)
    visualize_path(forward_propagated_path)

def test_component_iv():
    print('\ntest_component_iv() called -->')
    print('\ttesting function interpolate_arm() + visualize_arm_path()')
    start = (0, 0)
    goal = (np.pi / 4, np.pi / 6)
    path = interpolate_arm(start, goal)
    visualize_arm_path(path)
    
    # print('\n\ttesting function forward_propagate_arm() + visualize_arm_path()')
    start_pose = (0, 0)
    # """
    # plan = [
    #     (np.array([0.0, np.pi / 4]), 5),
    #     (np.array([0.1, np.pi / 4]), 5),
    #     (np.array([0.2, np.pi / 4]), 5),
    #     (np.array([0.3, np.pi / 4]), 5),
    #     (np.array([0.4, np.pi / 4]), 5),
    #     (np.array([0.5, np.pi / 4]), 5),
    # ]
    # """
    # """
    # plan = [
    #     ([0.1, 0.05], 5),
    #     ([0.2, -0.1], 3),
    #     ([0.5, 0.3], 2),
    #     ([0, 0.2], 4),
    #     ([-0.3, -0.2], 3),
    #     ([0.1, 0.0], 6)
    # ]
    # """
    plan = [
        (np.array([0.1, 0.05]), 5),
        (np.array([0.1, 0.05]), 5),
        (np.array([0.1, 0.05]), 5),
        (np.array([-0.1, -0.05]), 5),
        (np.array([-0.1, -0.05]), 5),
        (np.array([-0.1, -0.05]), 5),
        (np.array([-0.1, -0.05]), 5),
        (np.array([-0.1, -0.05]), 5),
        (np.array([-0.1, -0.05]), 5),
        (np.array([-0.1, -0.05]), 5),
    ]
    path = forward_propagate_arm(start_pose, plan)
    visualize_arm_path(path)
    

# test_component_i()
# test_component_ii()
# test_component_iii()
test_component_iv()

print('\n*** All components tested ***')