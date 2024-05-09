import numpy as np

def compute_similarity(vector1: list, vector2: list, mean_vector1: float, mean_vector2: float) -> float:
    # # @TODO: Implement the actual computation of similarity between vector1 and vector2.
    # The current implementation returns a placeholder value of 1. Update this function 
    # to perform the appropriate similarity calculation and return the result.
    ratings1 = {key1: value1-mean_vector1 for key1, value1 in vector1}
    ratings2 = {key2: value2-mean_vector2 for key2, value2 in vector2}

    common_users = set(ratings1.keys()) & set(ratings2.keys())
    if not common_users:
        return 0  # No common users, similarity is 0
    
    # Calculate pearson similarity# Calculate person similarity
    sumAB = sum([ratings1[userId] * ratings2[userId] for userId in common_users])
    sumA = sum([ratings1[userId] ** 2 for userId in common_users])
    sumB = sum([ratings2[userId] ** 2 for userId in common_users])
    
    # Check for division by zero
    if sumA == 0 or sumB == 0: return 0

    similarity = sumAB / (np.sqrt(sumA) * np.sqrt(sumB))
    return similarity
    

if __name__ == "__main__":
    
    vector_a, vector_b = [(1, 2), (2, 1), (3, 0), (4, 4)], [(4, 5), (3, 2), (2, 1), (1, 2)]
    sim = compute_similarity(vector_a, vector_b, 1, 1)
    print(sim)
    