import numpy as np

def compute_similarity(vector1: list, vector2: list) -> float:
    # # @TODO: Implement the actual computation of similarity between vector1 and vector2.
    # The vectors can be represented as list of tuples (Id, rating) to ease the computation.
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.
    
    return 1
    

if __name__ == "__main__":
    
    vector_a, vector_b = [(1, 2), (2, 1), (3, 0), (4, 4)], [(4, 5), (3, 2), (2, 1), (1, 2)]
    sim = compute_similarity(vector_a, vector_b)
    
    