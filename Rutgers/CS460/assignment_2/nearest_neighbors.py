import argparse
import numpy as NP

def euclidean_distance(config1, config2):
    return NP.linalg.norm(NP.array(config1) - NP.array(config2))
