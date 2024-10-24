"""
    Utility file containing helper functions.
    - utils.py
        > function get_polygon_corners(center: (float, float), width: float, height: float, theta: float) -> NP.ndarray:)
"""

import numpy as NP

def get_polygon_corners(center: tuple, width: float, height: float, theta: float) -> NP.ndarray:
    width_prime, height_prime = width / 2, height / 2
    CORNERS = NP.array(
        [
            [-width_prime, -height_prime],
            [width_prime, -height_prime],
            [width_prime, height_prime],
            [-width_prime, height_prime]
        ]
    )
    
    COS_THETA, SIN_THETA = NP.cos(theta), NP.sin(theta)
    ROTATION_MATRIX = NP.array(
        [
            [COS_THETA, -SIN_THETA],
            [SIN_THETA, COS_THETA]
        ]
    )
    
    ROTATED_CORNERS = CORNERS @ ROTATION_MATRIX.T
    
    return ROTATED_CORNERS + NP.array(center)