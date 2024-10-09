import numpy as np

def check_SOn(m: np.ndarray, epsilon: float = 0.01) -> bool:
    if m.shape[0] != m.shape[1]:
        return False
    
    identity_matrix = np.eye(m.shape[0])
    orthogonality_check = np.allclose(np.dot(m.T, m), identity_matrix, atol = epsilon)
    
    determinant_check = np.isclose(np.linalg.det(m), 1.0, atol = epsilon)
    
    return orthogonality_check and determinant_check

def check_quaternion(v: np.array, epsilon: float = 0.01) -> bool:
    if len(v) != 4:
        return False
    
    magnitude_squared = np.sum(np.square(v))
    
    return np.abs(magnitude_squared - 1) < epsilon

def check_SEn(m: np.ndarray, epsilon: float = 0.01) -> bool:
    n = m.shape[0] - 1
    
    if m.shape[0] != m.shape[1] or (n not in [2, 3]):
        return False
    
    rotation_matrix = m[:n, :n]
    
    if not check_SOn(rotation_matrix, epsilon):
        return False

    last_row_check = np.allclose(m[n, :], np.append(np.zeros(n), 1), atol = epsilon)
    
    return last_row_check