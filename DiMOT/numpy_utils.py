import numpy as np

def generate_RD(num_points, num_clusters):
    cluster_centers = np.random.rand(num_clusters)
    std_dev = np.random.uniform(0, 0.05, num_clusters)
    cluster_sizes = np.random.multinomial(num_points, np.ones(num_clusters) / num_clusters)

    points = []
    for i in range(num_clusters):
        cluster_points = cluster_centers[i] + np.random.randn(cluster_sizes[i]) * std_dev[i]
        points.append(cluster_points)
    return np.concatenate(points)

def sort_and_match_single_index(vec1, vec2):
    # 检查长度是否相等
    if len(vec1) != len(vec2):
        raise ValueError("两个向量的长度必须相等")
    
    # 对 vec1 排序，获取排序索引
    sorted_indices_vec1 = np.argsort(vec1)
    sorted_indices_vec1 = np.argsort(sorted_indices_vec1)
    sorted_indices_vec2 = np.argsort(vec2)
    
    return sorted_indices_vec1, sorted_indices_vec2

def get_length5(plength, npoints):
    indices1, indices2 = sort_and_match_single_index(plength, npoints)
    length = np.abs(plength - npoints[indices2][indices1])
    return length