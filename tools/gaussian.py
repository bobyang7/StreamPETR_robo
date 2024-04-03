import torch
import torch.nn.functional as F

# 定义高斯分布的均值和标准差
mu = 0
sigma = 1

# 生成一些高斯分布的样本
num_samples = 1000
gaussian_samples = torch.normal(mean=mu, std=sigma, size=(num_samples,num_samples))

# 计算高斯分布的累积分布函数（CDF）
cdf_values = 0.5 * (1 + torch.erf((gaussian_samples - 0) / (15 * torch.sqrt(torch.tensor(2.0)))))

# 将CDF值映射到均匀分布的样本
uniform_samples = cdf_values

# 输出均匀分布的样本
print(uniform_samples)

gaussian_samples_restored = 0 + 15 * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * torch.tensor(1e-7) - 1)

# 输出恢复的高斯分布样本
print(gaussian_samples_restored)