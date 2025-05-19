import torch
from functorch import vmap

def sort_and_match_single_index(vec1, vec2):
    sorted_indices_vec1 = torch.argsort(vec1)
    sorted_indices_vec1 = torch.argsort(sorted_indices_vec1)
    sorted_indices_vec2 = torch.argsort(vec2)

    return sorted_indices_vec1, sorted_indices_vec2


def get_ptd(plength, npoints):
    """
    Calculates the Polar Transport Discrepancy using polar and reference distribution.
    """
    indices1, indices2 = sort_and_match_single_index(plength, npoints)
    length = torch.abs(plength - npoints[indices2][indices1])
    return length

def generate_points(n, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    positions = torch.rand(3, device=device)
    positions = torch.sort(positions).values
    random_indices = torch.randint(0, 3, (n,), device=device)  # 随机选择三个位置中的一个
    points = positions[random_indices] + torch.rand(n, device=device) * (
                positions[random_indices] - torch.roll(positions, shifts=1)[random_indices])
    return points

def generate_matrix(n, k, device='cuda'):
    return torch.stack([generate_points(n, device) for _ in range(k)], dim=1)

def get_ptd_batch(plength, RD):
    device = plength.device
    batched_get_ptd = vmap(get_ptd, in_dims=(None, 0))
    output = batched_get_ptd(plength, RD.T)
    return output