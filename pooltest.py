import torch

# 定义张量 x
x = torch.tensor([[[[ 0.4745,  0.0688],
                    [ 0.3176, -0.7667]],

                   [[-0.5901,  0.1834],
                    [ 0.1147, -0.0513]]],


                  [[[-0.7965, -0.8102],
                    [ 1.0922,  2.8834]],

                   [[-0.5387, -0.9867],
                    [ 0.9972, -1.4136]]]])

# 获取张量的形状
N, C, H, W = x.shape

# 调整维度顺序以便于在通道维度上排序
x_permuted = x.permute(0, 2, 3, 1)  # [N, H, W, C]

# 对通道维度上的值进行排序
sorted_indices = torch.argsort(x_permuted, dim=-1, descending=True)

# 恢复到原来的维度顺序
sorted_indices = sorted_indices.permute(0, 3, 1, 2)  # [N, C, H, W]

print("Sorted Indices:")
print(sorted_indices)
