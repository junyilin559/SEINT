import torch
from functorch import vmap

def sort_and_match_single_index(vec1, vec2):
    # 检查长度是否相等
    if len(vec1) != len(vec2):
        raise ValueError("两个向量的长度必须相等")

    # 对 vec1 排序，获取排序索引
    sorted_indices_vec1 = torch.argsort(vec1)
    sorted_indices_vec1 = torch.argsort(sorted_indices_vec1)
    sorted_indices_vec2 = torch.argsort(vec2)

    return sorted_indices_vec1, sorted_indices_vec2


def get_length5(plength, npoints):
    indices1, indices2 = sort_and_match_single_index(plength, npoints)
    length = torch.abs(plength - npoints[indices2][indices1])
    return length


def row_softmax(matrix):
    row_max = torch.max(matrix, dim=1, keepdim=True)[0]
    exp_matrix = torch.exp(matrix - row_max)
    softmax_matrix = exp_matrix / torch.sum(exp_matrix, dim=1, keepdim=True)
    return softmax_matrix


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

def get_length_batch(plength, RD):
    device = plength.device
    batched_lfOT = vmap(get_length5, in_dims=(None, 0))
    output = batched_lfOT(plength, RD.T)
    return output