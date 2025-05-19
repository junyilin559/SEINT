import numpy as np

def generate_RD(num_points, num_clusters):
    """
    Randomly generates a reference distribution.
    """
    cluster_centers = np.random.rand(num_clusters)
    std_dev = np.random.uniform(0, 0.05, num_clusters)
    cluster_sizes = np.random.multinomial(num_points, np.ones(num_clusters) / num_clusters)

    points = []
    for i in range(num_clusters):
        cluster_points = cluster_centers[i] + np.random.randn(cluster_sizes[i]) * std_dev[i]
        points.append(cluster_points)
    return np.concatenate(points)

def sort_and_match_single_index(vec1, vec2):
    sorted_indices_vec1 = np.argsort(vec1)
    sorted_indices_vec1 = np.argsort(sorted_indices_vec1)
    sorted_indices_vec2 = np.argsort(vec2)
    
    return sorted_indices_vec1, sorted_indices_vec2

def get_ptd(plength, rd):
    """
    Calculates the Polar Transport Discrepancy using polar and reference distribution.
    """
    indices1, indices2 = sort_and_match_single_index(plength, rd)
    length = np.abs(plength - rd[indices2][indices1])
    return length