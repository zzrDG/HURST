import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time
from pathlib import Path

from matplotlib.colors import ListedColormap



torch.set_printoptions(edgeitems=100)

device = torch.device("cuda:4")


def accuracy(vector_x, vector_y):
    new_v = vector_x - vector_y
    new_v = torch.abs(new_v)
    new_v = torch.sum(new_v).data.cpu().numpy()
    return new_v / len(vector_x)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def MSE_torch(prediction, true_value):
    prediction = prediction.flatten(0)
    true_value = true_value.flatten(0)
    prediction = torch.round(prediction)
    mse = torch.sum(torch.square(prediction - true_value) / len(prediction))
    return mse


def plot_loss(train_loss_arr, valid_loss_arr):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(train_loss_arr, 'k', label='training loss')
    ax1.plot(valid_loss_arr, 'g', label='validation loss')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    # ax2.plot(train_mape_arr, 'r--', label='train_mape_arr')
    # ax2.plot(v_mape_arr, 'b--', label='v_mape_arr')
    ax2.legend(loc=2)
    plt.show()
    plt.clf()


def MSE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()
    mse = np.sum(np.square(prediction - true_value) / len(prediction))
    return mse


def MinMax(data):
    batch, days, x, y, feature = data.shape
    minmax = np.zeros((feature, 2))  # 用于存储每个特征的最小值和最大�?
    for i in range(feature):
        minmax[i][0] = np.min((data[:, :, :, :, i]))
        minmax[i][1] = np.max((data[:, :, :, :, i]))
    return minmax


def torch_normalization(data):
    batch, days, x, y, feature = data.shape
    minmax = torch.zeros(feature, 2)  # 用于存储每个特征的最小值和最大�?
    epsilon = 1e-10
    for i in range(feature):
        minmax[i][0] = torch.min((data[:, :, :, :, i]))
        minmax[i][1] = torch.max((data[:, :, :, :, i]))
        data[:, :, :, :, i] = (data[:, :, :, :, i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0] + epsilon)
    return data


def np_normalization(data):
    #batch, days, x, y, feature = data.shape
    origin_shape = data.shape
    batch = origin_shape[0]
    feature = origin_shape[-1]
    data = data.reshape(batch,-1,feature)
    minmax = np.zeros((feature, 2))  # 用于存储每个特征的最小值和最大�?
    epsilon = 1e-10
    for i in range(feature):
        minmax[i][0] = np.min((data[:, :, i]))
        minmax[i][1] = np.max((data[:, :, i]))
        data[:, :, i] = (data[:, :, i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0] + epsilon)
    data = data.reshape(origin_shape)
    return data


def inverse_normalization(data, minmax):
    feature = data.shape[4]
    for i in range(feature):
        data[:, :, :, :, i] = data[:, :, :, :, i] * (minmax[i][1] - minmax[i][0]) + minmax[i][0]
    return data


def dim_num(shape, kernel):
    padding = (kernel-1)//2
    new_shape = ((shape[0] - kernel[0] + 2 * padding[0]) // 1 + 1,
                 (shape[1] - kernel[1] + 2 * padding[1]) // 1 + 1,
                 (shape[2] - kernel[2] + 2 * padding[2]) // 1 + 1)
    num = new_shape[0] * new_shape[1] * new_shape[2]
    return new_shape, num



def mape(y_true, y_pred, epsilon=0.1):
    y_true = y_true.flatten(0)
    y_pred = y_pred.flatten(0)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def plot_data_distribution(data, bins=30, title='Data Distribution', xlabel='Value', ylabel='Frequency'):
    data = np.array(data)
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def calculate_mape(y_true, y_pred, epsilon=0.1):
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape


def calculate_kl_divergence(y_true, y_pred, epsilon=0.1):
    p = y_true + epsilon
    q = y_pred + epsilon
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    kl_divergence = torch.sum(p * torch.log(p / q))
    return kl_divergence


class MSEAndMAPELoss(nn.Module):
    def __init__(self, epsilon=0.1, lambda_kl=0.1):
        super(MSEAndMAPELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.epsilon = epsilon
        self.lambda_kl = lambda_kl

    def forward(self, y_true, y_pred):
        mse_loss = self.mse(y_true, y_pred)
        mape_loss = self.calculate_mape(y_true, y_pred)
        kl_loss = self.calculate_kl_divergence(y_true, y_pred)
        return mape_loss

    def calculate_mape(self, y_true, y_pred):
        y_true_cpu = y_true.cpu()
        y_pred_cpu = y_pred.cpu()
        mape = torch.mean(torch.abs((y_true_cpu - y_pred_cpu) / (y_true_cpu + self.epsilon))) * 100
        return mape

    def calculate_kl_divergence(self, y_true, y_pred):
        y_true_cpu = y_true.cpu()
        y_pred_cpu = y_pred.cpu()
        p = y_true_cpu + self.epsilon  
        q = y_pred_cpu + self.epsilon
        p = p / torch.sum(p) 
        q = q / torch.sum(q)
        kl_divergence = torch.sum(p * torch.log(p / q))
        return kl_divergence
        
        



def calculate_padding(input_size, kernel_size, stride):
    return ((stride - 1) * input_size - stride + kernel_size) // 2


def plot_loss_heatmap(y_true, y_pred, criterion, title='Loss Heatmap', xlabel='X', ylabel='Y'):
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    loss_map = criterion(y_true, y_pred)
    loss_map = loss_map.squeeze(1)
    avg_loss_map = torch.mean(loss_map, dim=0).cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_loss_map, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_ExpertHeatmaps(data):
    plt.figure(figsize=(10, 10))
    axes = []
    for i in range(data.shape[2]):
        ax = plt.subplot(2, 2, i + 1)
        cax = ax.imshow(data[:, :, i], cmap='hot', interpolation='nearest',origin='lower')
        plt.title(f'Expert {i+1}')
        axes.append(ax)
    cbar = plt.colorbar(cax, ax=axes)
    cbar.set_label('Value')
    plt.show()

def plot_Expert(data):
    shape = data.shape
    data = data.reshape(shape[0],1,64,64,38,4)
    data = data[0,0,:,:,36,:].cpu().detach().numpy() 
    #data = data[0,0,:,:,36,:].cpu().numpy()
    plt.figure(figsize=(10, 10))
    mask = np.load(r'/home/zhouzirui/NYC_data/x/mask_64_64.npy')
    axes = []
    for i in range(data.shape[2]):
        ax = plt.subplot(2, 2, i + 1)
        cax = ax.imshow(data[:, :, i]*mask.transpose(), cmap='hot', interpolation='nearest',origin='lower')
        plt.title(f'Expert {i+1}')
        axes.append(ax)
    cbar = plt.colorbar(cax, ax=axes)
    cbar.set_label('Value')
    plt.show()

def plot_mask_heatmap(mask, title, epoch, save_path=None):
    mask_np = mask.cpu().numpy()
    mask = np.load(r'/home/zhouzirui/NYC_data/x/mask_64_64.npy')
    result =  mask_np
    fig, ax = plt.subplots()
    im = ax.imshow(result.transpose(), cmap='hot', interpolation='nearest', origin='lower')
    fig.colorbar(im, ax=ax)  
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    if save_path:
        filename = f"{save_path}{title}_{epoch}.png"
        fig.savefig(filename)  
        plt.close(fig)  

    plt.show()
    
def grid_loss(pred, true_data):
    assert pred.shape == true_data.shape, "Prediction and true data must have the same shape"
    batch, days, x, y, feature = pred.shape
    map_loss = torch.zeros((x, y), device=pred.device)
    for i in range(batch):
        for j in range(days):
            for n in range(feature):
                loss = (pred[i, j, :, :, n] - true_data[i, j, :, :, n]) ** 2
                map_loss += loss
    map_loss = map_loss / (batch * days * feature)
    return map_loss


def avg_grid_loss(data, len):
    shape = data.shape
    data = data.reshape(len, -1, shape[1])
    _, x, y = data.shape
    avg_map_loss = torch.zeros((x, y))
    for i in range(len):
        avg_map_loss += data[i, :, :]
    avg_map_loss = avg_map_loss / len
    avg_map_loss = torch.where(avg_map_loss > 0.1, avg_map_loss, 0)
    return avg_map_loss


def sample_new_mask(map_softmax, mask_rate, total_grids=1128):
    x, y = map_softmax.shape
    map_flatten = map_softmax.view(-1)
    num_samples = int(mask_rate * total_grids)
    sampled_indices = torch.multinomial(map_flatten, num_samples, replacement=False)
    mask = torch.zeros(x * y)
    mask[sampled_indices] = 1
    mask = mask.view(x, y)
    return mask


def get_valid_location(mask):
    mask = mask.view(-1)
    valid_mask = mask.bool()
    valid_indices = torch.arange(mask.size(0), device=mask.device)[valid_mask]
    return valid_indices


def remove_previous_days(data):
    batch, days, x, y, feature = data.shape
    new_data = torch.zeros((batch, x, y, feature)).to(device)  #double
    for i in range(batch):
        new_data[i, :, :, :] = data[i, 6, :, :, :]
    return new_data


def st_mask(data, mask):
    depth, height, width = mask.shape
    batch = data.shape[0]
    dim = data.shape[-1]
    mask = mask.unsqueeze(0).unsqueeze(-1).expand(batch,  depth, height, width,dim)
    masked_data = data.to(device) * mask.to(device)

    return masked_data

def init_st_mask_and_weight(data, batch, kernel_size, stride, padding):
    _, depth, height, width, channels = data.shape
    conv_depth = (depth + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    conv_height = (height + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    conv_width = (width + 2 * padding[2] - kernel_size[2]) // stride[2] + 1
    init_mask = torch.ones((batch, conv_depth, conv_height, conv_width, channels), device=data.device)
    init_weights = torch.ones((batch, conv_depth, conv_height, conv_width, channels), device=data.device)
    return init_mask,init_weights



def conv_loss(data, kernel_size, pooling='avg', stride=1):
    """
    对输入数据进行卷积操作。
    
    参数:
        data (torch.Tensor): 输入数据，形状为 (depth, height, width)。
        kernel_size (int): 卷积核大小。
        pooling (str): 池化方式，'sum' 或 'avg'。
        stride (int): 卷积步长，默认为 1。
    
    返回:
        conv_data (torch.Tensor): 卷积后的数据，形状为 (conv_depth, conv_height, conv_width)。
    """
    depth, height, width = data.shape
    padding = (kernel_size - 1) // 2

    # 计算卷积后的输出尺寸
    conv_depth = (depth + 2 * padding - kernel_size) // stride + 1
    conv_height = (height + 2 * padding - kernel_size) // stride + 1
    conv_width = (width + 2 * padding - kernel_size) // stride + 1

    # 初始化卷积结果
    conv_data = torch.zeros((conv_depth, conv_height, conv_width), device=data.device)

    # 对输入数据进行填充
    padded_data = torch.zeros(
        (depth + 2 * padding, height + 2 * padding, width + 2 * padding), device=data.device
    )
    padded_data[padding:padding + depth, padding:padding + height, padding:padding + width] = data

    # 执行卷积操作
    for d in range(0, conv_depth * stride, stride):
        for h in range(0, conv_height * stride, stride):
            for w in range(0, conv_width * stride, stride):
                start_d = d
                end_d = start_d + kernel_size
                start_h = h
                end_h = start_h + kernel_size
                start_w = w
                end_w = start_w + kernel_size
                window = padded_data[start_d:end_d, start_h:end_h, start_w:end_w]
                if pooling == 'sum':
                    conv_data[d // stride, h // stride, w // stride] = window.sum()
                elif pooling == 'avg':
                    conv_data[d // stride, h // stride, w // stride] = window.mean()

    return conv_data


def time_mask(data, mask):
    batch, days, x, y, feature = data.shape
    data = data.permute(0, 4, 1, 2, 3)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, feature, days, x, y)
    masked_data = data.to(device) * mask.to(device)
    masked_data = masked_data.permute(0, 2, 3, 4, 1)
    return masked_data

def time_grid_loss(prediction, true_data, minmax):
    assert prediction.shape == true_data.shape, "Prediction and true data must have the same shape"
    batch, days, x, y = prediction.shape
    #print(prediction.shape)
    time_map_loss = torch.zeros((days, x, y), device=prediction.device)
    for i in range(batch):
        for j in range(days):
            time_map_loss[0] += (prediction[i, j, :, :] - true_data[i, j, :, :]) ** 2
    time_map_loss = time_map_loss / (batch )
    #print(time_map_loss)
    return time_map_loss


def avg_time_grid_loss(time_grid_loss):
    batch, x, y = time_grid_loss.shape
    time_grid_loss = torch.tensor(time_grid_loss.reshape(-1, 1, x, y)).to(device)
    len = time_grid_loss.shape[0]
    time_map_loss = torch.zeros((1, x, y)).to(device)
    for i in range(len):
        time_map_loss += time_grid_loss[i]
    return time_map_loss / len


def generate_temporal_spatial_mask(original_mask, map_loss, mask_rate, lambda_value, temporal_dim, spatial_dim, weights, top_ratio):
    """
    生成时空掩码和更新权重。
    
    Args:
        original_mask (torch.Tensor): 原始掩码，形状为 [H,W]。
        map_loss (torch.Tensor): 平均损失，形状为 [7,H,W]。
        mask_rate (float): 掩码率。
        lambda_value (float): 高斯平滑参数。
        temporal_dim (int): 时间维度大小。
        spatial_dim (tuple): 空间维度大小 (x, y)。
        weights (torch.Tensor): 权重，形状为 [temporal_dim, spatial_dim[0], spatial_dim[1]]。
        top_ratio (float): 权重更新比例。
    
    Returns:
        torch.Tensor: 新掩码，形状为 [temporal_dim, spatial_dim[0], spatial_dim[1]]。
        torch.Tensor: 新权重，形状为 [temporal_dim, spatial_dim[0], spatial_dim[1]]。
    """
    temporal_spatial_mask = torch.zeros((temporal_dim, spatial_dim[0], spatial_dim[1]), device=device)
    new_weights = torch.zeros_like(weights)
    
    for t in range(temporal_dim):
        # 使用 map_loss 更新每个时间步的掩码和权重
        temporal_spatial_mask[t], new_weights[t] = update_mask_new(
            original_mask[t], map_loss[t], mask_rate, lambda_value, weights[t], top_ratio
        )
    
    return temporal_spatial_mask, new_weights


def apply_gaussian_smoothing(input_tensor, lambda_value):
    """
    对输入张量进行高斯平滑。
    
    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 [x, y]。
        lambda_value (float): 高斯平滑参数。
    
    Returns:
        torch.Tensor: 平滑后的张量，形状为 [x, y]。
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    input_np = input_tensor.cpu().numpy()
    
    # 对二维数组进行高斯平滑
    smoothed_np = gaussian_filter(input_np, sigma=lambda_value)
    
    # 将 NumPy 数组转换回 PyTorch 张量
    smoothed_tensor = torch.tensor(smoothed_np, device=input_tensor.device, dtype=input_tensor.dtype)
    
    return smoothed_tensor


def update_mask_new(old_mask, map_loss, mask_rate, lambda_value, weight, top_ratio):
    """
    更新掩码和权重。
    
    Args:
        old_mask (torch.Tensor): 旧掩码，形状为 [x, y]。
        map_loss (torch.Tensor): 平均损失，形状为 [x, y]。
        mask_rate (float): 掩码率。
        lambda_value (float): 高斯平滑参数。
        weight (torch.Tensor): 权重，形状为 [x, y]。
        top_ratio (float): 权重更新比例。
    
    Returns:
        torch.Tensor: 新掩码，形状为 [x, y]。
        torch.Tensor: 新权重，形状为 [x, y]。
    """
    # 高斯平滑
    
    # 高斯平滑处理
    smoothed_loss = apply_gaussian_smoothing(map_loss, lambda_value)
    
    # 权重更新（假设实现正确）
    new_weight = update_weights_based_on_loss(weight, smoothed_loss, top_ratio)
    
    # 获取有效位置坐标（二维索引）
    valid_positions = torch.nonzero(old_mask == 1)  # shape=[num_valid, 2]
    num_valid = len(valid_positions)
    
    # 无有效区域直接返回
    if num_valid == 0:
        return old_mask.clone(), new_weight
    
    # 有效位置权重提取与归一化
    valid_weights = new_weight[valid_positions[:, 0], valid_positions[:, 1]]  # [num_valid]

    #probabilities = F.softmax(valid_weights, dim=0)  # 概率分布
    
    # 计算采样数量（添加浮点误差容限）
    num_samples = int(mask_rate * num_valid + 1e-8)
    
    # 采样索引与坐标转换
    sampled_indices = torch.multinomial(valid_weights, num_samples, replacement=False)
    sampled_positions = valid_positions[sampled_indices]  # [num_samples, 2]
    
    # 更新掩码
    new_mask = old_mask.clone()
    new_mask[sampled_positions[:,0],sampled_positions[:,1]] = 0.0  # 标记采样位置
    return new_mask, new_weight
"""
def update_weights_based_on_loss(weight, smoothed_loss, top_ratio):

    根据损失更新权重。
    
    Args:
        weight (torch.Tensor): 权重，形状为 [x, y]。
        smoothed_loss (torch.Tensor): 平滑后的损失，形状为 [x, y]。
        top_ratio (float): 权重更新比例。
    
    Returns:
        torch.Tensor: 更新后的权重，形状为 [x, y]。

    # 动态调整权重
    flat_loss = smoothed_loss.view(-1)  # 展平为 1D 张量
    sorted_loss, _ = torch.sort(flat_loss)  # 对损失排序
    
    # 计算前 top_ratio 和后 top_ratio 的阈值
    top_threshold = sorted_loss[int((1 - top_ratio) * len(sorted_loss))]
    bottom_threshold = sorted_loss[int(top_ratio * len(sorted_loss))]
    
    # 调整权重
    weight[smoothed_loss > top_threshold] *= 2.0  # 前 top_ratio 的损失，权重翻倍
    weight[smoothed_loss < bottom_threshold] *= 0.5  # 后 top_ratio 的损失，权重减半
    
    # 归一化权重
    weight = weight / (torch.sum(weight) + 1e-8)
    
    return weight
"""

def get_neighbors_indices(idx, map_shape):
    neighbors = []
    x = idx // map_shape[1]  
    y = idx % map_shape[1]  

    rows, cols = map_shape
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue  
            new_x = x + i
            new_y = y + j
            if 0 <= new_x < rows and 0 <= new_y < cols:
                neighbor_idx = new_x * cols + new_y
                neighbors.append(neighbor_idx)

    return torch.tensor(neighbors, dtype=torch.long)

def update_weights_based_on_loss(weights, time_map_loss, top_ratio=0.3, max_weight=64, min_weight=1/4):
    device = time_map_loss.device
    weights = weights.to(device)
    weights_flat = weights.view(-1)
    time_map_loss_flat = time_map_loss.view(-1)
    sorted_indices = torch.argsort(time_map_loss_flat, descending=True)
    total_elements = time_map_loss_flat.size(0)
    top_n = int(total_elements * top_ratio)
    if max_weight is None:
        max_weight = weights_flat.max() * 2  
    if min_weight is None:
        min_weight = weights_flat.min() / 2 
    for i in range(top_n):
        index = sorted_indices[i]
        new_weight = weights_flat[index] * 2
        weights_flat[index] = min(new_weight, max_weight)

    for i in range(top_n, total_elements):
        index = sorted_indices[i]
        new_weight = weights_flat[index] / 2
        weights_flat[index] = max(new_weight, min_weight)
    new_weight = weights_flat.view(weights.shape)

    return new_weight


def coords(data):
    shape = data.shape
    coordinates = torch.zeros((shape[0], shape[2], shape[3], 2)).to(device)
    #coordinates = coordinates.to(torch.double)
    for x in range(shape[2]):
        for y in range(shape[3]):
            coordinates[:, x, y, 0] = x 
            coordinates[:, x, y, 1] = y
    coordinates = coordinates.reshape(shape[0],-1,2)
    return coordinates
    


def Weight(device):
    x, y = 64, 64
    length = x * y
    W = torch.zeros((length, length), device=device)
    offset = 2  
    for i in range(x):
        for j in range(y):
            index = i * y + j
            for di in range(max(0, i - offset), min(x, i + offset + 1)):
                for dj in range(max(0, j - offset), min(y, j + offset + 1)):
                    if i == di and j == dj:
                        continue 
                    j_index = di * y + dj
                    distance = torch.sqrt((torch.tensor([i, j], device=device) - torch.tensor([di, dj], device=device)).pow(2).sum())
                    W[index, j_index] = 1 / (distance + 1)
                    W[j_index, index] = 1 / (distance + 1)  
    return W
    
def compute_local_moran(data):
    batch_size, _, num_features = data.shape
    x = 64
    y = 64
    #print(data.shape)
    # Reshape data to [batch_size, num_features, x, y]
    data = data.reshape(batch_size, num_features, x, y)
    
    # Initialize the output tensor
    local_moran = torch.zeros((batch_size, num_features, x, y), device=data.device, requires_grad=False)

    # Define the weight matrix for the convolution
    weight_matrix = torch.tensor([
        [1/8, 1/5, 1/2, 1/5, 1/8],
        [1/5, 1/2, 1, 1/2, 1/5],
        [1/2, 1, 1, 1, 1/2],
        [1/5, 1/2, 1, 1/2, 1/5],
        [1/8, 1/5, 1/2, 1/5, 1/8]
    ], dtype=data.dtype, device=data.device)

    # Define the convolution layer
    conv_layer = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
    conv_layer.weight = nn.Parameter(weight_matrix.view(1, 1, 5, 5))
    conv_layer.weight.requires_grad = False

    # Iterate over each feature
    for feature in range(num_features):
        # Extract the current feature map
        feature_map = data[:, feature, :, :].unsqueeze(1)  # [batch_size, 1, x, y]
        
        # Compute mean and standard deviation
        mean = feature_map.mean(dim=(2, 3), keepdim=True)
        std = feature_map.std(dim=(2, 3), keepdim=True) + 1e-6
        
        # Compute z-scores
        z_scores = (feature_map - mean) / std
        
        # Compute spatial lag using convolution
        spatial_lag = conv_layer(z_scores).squeeze(1)  # [batch_size, x, y]
        
        # Compute local Moran's I
        for b in range(batch_size):
            z_score = z_scores[b, 0, :, :]
            spatial_lag_value = spatial_lag[b, :, :]
            
            numerator = z_score * spatial_lag_value
            denominator = (spatial_lag_value ** 2).sum() + 1e-6
            
            local_moran_value = numerator / denominator
            
            local_moran[b, feature, :, :] = local_moran_value
    
    # Reshape the output to [batch_size, x*y, num_features]
    local_moran = local_moran.permute(0, 2, 3, 1).reshape(batch_size, x*y, num_features)
    
    return local_moran
    
def calculate_expert_losses(model_output, gate_output, target, criterion):
    batch_size, seq_len, num_experts = gate_output.shape
    expert_losses = []
    gate_output = gate_output.view(batch_size, seq_len, num_experts, 1)
    top1_gate_output = torch.zeros_like(gate_output).to(gate_output.device)
    top1_gate_output.scatter_(-2, torch.argmax(gate_output, dim=-2).unsqueeze(-2), 1)

    for i in range(num_experts):
        expert_mask = top1_gate_output[:, :, i, 0].bool()
        expert_output = model_output[expert_mask]
        expert_target = target[expert_mask]
        if expert_output.size(0) > 0: 
            expert_loss = criterion(expert_output, expert_target)
            expert_losses.append(expert_loss)

    return expert_losses
    

    
def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {num_params}")
    print(f"Total number of parameters: {total_params}")
    
def monitor_gpu():
        stats = torch.cuda.memory_stats(1)
        print(f'Allocated: {round(stats["allocated_bytes.all.current"] / 1024 ** 3, 1)} GB')
        print(f'Cached: {round(stats["reserved_bytes.all.current"] / 1024 ** 3, 1)} GB')

def calculate_local_moran_index(data, window_size=3):
    """
    计算局部莫兰指数
    
    :param data: 输入数据，形状为 [batch, 7 days, x, y, feature]
    :param window_size: 窗口大小，用于计算局部莫兰指数
    :return: 局部莫兰指数，形状为 [batch, 7 days, x, y, feature]
    """
    batch_size, days, x_dim, y_dim, feature_dim = data.shape
    
    # 初始化局部莫兰指数矩阵
    local_moran_index = np.zeros((batch_size, days, x_dim, y_dim, feature_dim))
    
    # 计算每个特征的局部莫兰指数
    for f in range(feature_dim):
        # 计算每个位置的均值
        mean_value = np.mean(data[:, :, :, :, f], axis=(2, 3), keepdims=True)
        
        # 计算每个位置的偏差
        deviation = data[:, :, :, :, f] - mean_value
        
        # 计算局部莫兰指数
        for i in range(x_dim):
            for j in range(y_dim):
                # 获取当前位置的偏差
                center_deviation = deviation[:, :, i, j]
                
                # 获取窗口内的偏差
                window_deviations = []
                for dx in range(-window_size, window_size + 1):
                    for dy in range(-window_size, window_size + 1):
                        if dx == 0 and dy == 0:
                            continue
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < x_dim and 0 <= nj < y_dim:
                            neighbor_deviation = deviation[:, :, ni, nj]
                            window_deviations.append(neighbor_deviation)
                
                # 计算窗口内偏差的加权和
                if window_deviations:
                    window_deviations = np.stack(window_deviations, axis=-1)
                    weighted_sum = np.mean(window_deviations, axis=-1)
                    
                    # 确保维度一致
                    if weighted_sum.shape[-1] == 1:
                        weighted_sum = weighted_sum.squeeze(-1)
                    else:
                        weighted_sum = np.mean(weighted_sum, axis=-1)
                    
                    local_moran_index[:, :, i, j, f] = np.sum(center_deviation * weighted_sum, axis=-1)
    
    return local_moran_index

def z_curve_2d(height, width):
    """
    生成一个二维空间数据的Z曲线顺序。
    
    参数:
    height (int): 高度
    width (int): 宽度
    
    返回:
    z_order (np.ndarray): Z曲线顺序的索引数组
    """
    z_order = np.zeros((height, width), dtype=int)
    index = 0
    for i in range(height):
        for j in range(width):
            z_order[i, j] = index
            index += 1
    return z_order

def z_curve_flatten(x, z_order):
    """
    将输入张量按照Z曲线顺序展开。
    
    参数:
    x (torch.Tensor): 输入张量，形状为 (batch_size, height, width, features)
    z_order (np.ndarray): Z曲线顺序的索引数组
    
    返回:
    flattened_x (torch.Tensor): 展开后的张量，形状为 (batch_size, height * width, features)
    """
    #print(x.shape)
    batch_size, height, width, features = x.shape
    flattened_x = x.reshape(batch_size, height * width, features)
    z_order_indices = torch.tensor(z_order, device=x.device).view(-1)
    flattened_x = flattened_x[:, z_order_indices, :]
    return flattened_x


def masked_mse_mae(all_predictions, all_labels, mask):
    """
    计算所有预测结果与原始标签之间的整体 MSE 和 MAE 误差，只考虑 mask 为 1 的地方。
    
    Args:
        all_predictions (torch.Tensor): 所有预测结果。
        all_labels (torch.Tensor): 所有原始标签。
        mask (torch.Tensor): 掩码，只计算 mask 为 1 的地方的误差。
    
    Returns:
        tuple: (MSE 误差, MAE 误差)
    """
    # 将 mask 扩展到与预测结果和标签相同的形状
    mask = torch.from_numpy(mask)  # 将 NumPy 数组转换为 Tensor
    mask = mask.unsqueeze(0).unsqueeze(-1).expand_as(all_predictions)
    
    # 只计算 mask 为 1 的地方的误差
    masked_predictions = all_predictions[mask == 1]
    masked_labels = all_labels[mask == 1]
    
    # 计算 MSE
    mse_criterion = nn.MSELoss()
    mse = mse_criterion(masked_predictions, masked_labels).item()
    
    # 计算 MAE
    mae_criterion = nn.L1Loss()  # L1Loss 用于计算 MAE
    mae = mae_criterion(masked_predictions, masked_labels).item()
    
    return mse, mae

class DistanceLoss(nn.Module):
    def __init__(self, epsilon=0.1, 
                 window_size=5, 
                 num_sample_windows=20, 
                 diversity_weight_schedule=None,
                 lambda_mse=10,
                 lambda_load_balance=0.01,
                 lambda_contrastive=0.001
                 ):
        super(DistanceLoss, self).__init__()
        self.epsilon = epsilon
        self.window_size = window_size  # 窗口大小
        self.num_sample_windows = num_sample_windows  # 采样窗口的数量
        self.diversity_weight_schedule = diversity_weight_schedule  # 多样性损失的权重调度

        # 损失权重
        self.lambda_mse = lambda_mse  # MSE 损失的权重
        self.lambda_load_balance = lambda_load_balance # 负载均衡损失的权重
        self.lambda_contrastive = lambda_contrastive

        # 损失函数
        self.mse = nn.MSELoss(reduction='none')  # MSE 损失函数
        # 掩码

    def forward(self, true, pred, gate, mask, map_mask):
        """
        计算综合损失。
        :param true: 真实值，形状为 [batch_size, 7, height, width, feature]
        :param pred: 预测值，形状为 [batch_size, 7, height, width, feature]
        :param gate: 专家权重，形状为 [batch_size, height, width, num_experts]
        :param mode: 模式 ("pretrain" 或 "predict")
        :return: 综合损失、MSE 损失、负载均衡损失、空间平滑性损失
        """
        # 将掩码转换为布尔张量
        origin_mask_expanded = map_mask.unsqueeze(0).unsqueeze(-1).expand_as(pred)
        if mask==None:
            mask_expanded = torch.ones_like(pred)
        else:
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(pred)
            mask_expanded = origin_mask_expanded -mask_expanded

        # 计算 MSE 损失
        mse_loss = self.mse(pred * mask_expanded, true * mask_expanded)
        mse_loss = mse_loss.mean()  # 计算平均损失
        # 负载均衡损失
        
        load_balance_loss = self.load_balance_loss(gate, origin_mask_expanded)


        #contrastive_loss = self.contrastive_loss(gate,origin_mask_expanded)
        # 计算总损失
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_load_balance * load_balance_loss 
            #self.lambda_contrastive * contrastive_loss
        )
                # 检查gate是否需要梯度

        return total_loss, mse_loss, load_balance_loss
        #return total_loss, mse_loss, load_balance_loss, contrastive_loss

    def load_balance_loss(self, gate, mask):
        """
        计算负载均衡损失（使用变异系数）。
        :param gate: 专家权重，形状为 [batch_size, T,height, width, num_experts]
        :param mask: 掩码，形状为 [height, width]
        :return: 负载均衡损失
        """

        # 将 mask 扩展到与 gate 相同的形状
        #mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(gate)  # [batch_size, height, width, num_experts]

        # 只计算 mask 为 1 的区域

        gate_masked = gate * mask  # 掩码区域外的 gate 值被置为 0
        expert_usage = torch.sum(gate_masked, dim=(2, 3)) / torch.sum(mask, dim=(2, 3))  # [batch_size, num_experts]

        # 计算均值
        mean_usage = torch.mean(expert_usage, dim=-1, keepdim=True)  # [batch_size, 1]

        # 计算标准差
        std_usage = torch.std(expert_usage, dim=-1, keepdim=True)  # [batch_size, 1]

        # 计算变异系数
        epsilon = 1e-8  # 避免除零
        coefficient_of_variation = std_usage / (mean_usage + epsilon)  # [batch_size, 1]

        # 惩罚变异系数
        load_balance_loss = coefficient_of_variation.mean()

        return load_balance_loss
        

    def multi_scale_spatial_loss(self,gate, mask, window_sizes=[3,5,7]):
        total_loss = 0.0
        batch, time, H,W,experts = gate.shape
        for ws in window_sizes:
            # 计算每个窗口尺寸的方差损失
            unfolded = gate.unfold(2, ws, 1).unfold(3, ws, 1)
            unfolded = unfolded.reshape(batch, time, -1, ws, ws, experts)
            window_var = torch.var(unfolded, dim=(3,4)).mean(dim=2)
            total_loss += window_var.mean()
        return total_loss / len(window_sizes)
    
    def contrastive_loss(self, gate, mask):
        device = gate.device
        batch, T, H, W, num_experts = gate.shape
        window_size = self.window_size
        half_win = window_size // 2
        gate = gate * mask

        # 步骤1：生成有效窗口坐标（保持不变）
        with torch.no_grad():
            center_mask = mask.any(dim=0).any(dim=0)[..., 0]
            valid_positions = torch.nonzero(center_mask)
            if valid_positions.size(0) == 0:
                return torch.tensor(0.0, device=device)
            valid_y, valid_x = valid_positions.unbind(dim=1)

        # 步骤2：批量提取窗口数据（保持不变）
        pad = half_win
        padded_gate = F.pad(gate, (0,0,pad,pad,pad,pad))
        window_indices = torch.arange(window_size, device=device)
        y_coords = valid_y.view(-1,1,1) + window_indices.view(1,-1,1)
        x_coords = valid_x.view(-1,1,1) + window_indices.view(1,1,-1)
        windows = padded_gate[:, :, y_coords, x_coords, :]
        flat_windows = windows.permute(1,2,0,3,4,5).reshape(-1, window_size**2, num_experts)

        # 步骤3：修改后的向量化计算（关键修改部分）
        c = (window_size // 2) * window_size + (window_size // 2)  # 中心点扁平索引
        indices = torch.arange(window_size**2, device=device)
        j = indices[indices != c]  # 所有非中心点索引
        i = torch.full_like(j, c)  # 全部使用中心点索引
        
        # 提取点对特征
        gate_i = flat_windows[:, i, :]  # [total, pairs, E]
        gate_j = flat_windows[:, j, :]
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(gate_i, gate_j, dim=-1)
        
        # 生成有效掩码（保持不变）
        valid_mask = self.get_valid_window_mask(mask, valid_y, valid_x, window_size)
        valid_pairs = valid_mask[:, i] & valid_mask[:, j]
        
        # 计算有效相似度均值
        cos_sim_valid = cos_sim * valid_pairs.float()
        num_valid_pairs = valid_pairs.sum(dim=1).clamp(min=1)
        
        # 最终损失计算（保持不变）
        window_loss = -(cos_sim_valid.sum(dim=1) / num_valid_pairs).mean()
        return window_loss

    def get_valid_window_mask(self, mask, valid_y, valid_x, window_size):
        """
        生成窗口内有效位置的掩码
        mask形状: [B, T, H, W, 1]
        valid_y/x: [N_valid]
        返回: [N_valid*B*T, window_area]
        """
        B, T, H, W, _ = mask.shape
        N_valid = valid_y.size(0)
        half_win = window_size // 2
        
        # 生成窗口坐标偏移
        offsets = torch.arange(-half_win, half_win+1, device=mask.device)
        dy, dx = torch.meshgrid(offsets, offsets)
        
        # 计算所有窗口位置的全局坐标
        global_y = valid_y.reshape(-1,1,1) + dy.reshape(1,-1,1)  # [N_valid, W, W]
        global_x = valid_x.reshape(-1,1,1) + dx.reshape(1,1,-1)  # [N_valid, W, W]
        
        # 生成有效位置掩码（是否在原始图像范围内）
        valid_mask = (global_y >= 0) & (global_y < H) & (global_x >= 0) & (global_x < W)
        
        # 与原始mask结合 [N_valid, W, W]
        original_mask = mask[0,:,:,:,0].any(dim=0).float()  # 假设batch维度一致
        global_y = torch.clamp(global_y, 0, H - 1)
        global_x = torch.clamp(global_x, 0, W - 1)       
        valid_mask = valid_mask & (original_mask[global_y, global_x] > 0.5)
        
        # 扩展维度并重整 [N_valid*B*T, window_area]
        return valid_mask.view(N_valid, -1).repeat(B*T, 1)
    
    def expert_diversity(self, gate):
        """
        计算多样性损失。
        :param gate: 专家权重，形状为 [batch_size,T, height, width, num_experts]
        :return: 多样性损失
        """
        batch_size, T,height, width, num_experts = gate.shape

        # 将掩码转换为布尔张量
        mask = torch.tensor(self.mask, dtype=torch.bool).to(gate.device)

        # 获取掩码为 1 的位置索引
        valid_indices = torch.nonzero(mask)  
        num_valid = valid_indices.shape[0]

        # 如果掩码中没有有效位置，直接返回 0
        if num_valid == 0:
            return torch.tensor(0.0, dtype=torch.float32).to(gate.device)

        # 随机采样 X 个窗口的中心位置
        num_samples = min(self.num_sample_windows, num_valid)  # 采样窗口的数量
        sampled_indices = valid_indices[torch.randperm(num_valid)[:num_samples]]  # 随机采样

        # 初始化多样性损失
        diversity_loss = 0.0

        # 遍历采样的窗口中心
        for idx in sampled_indices:
            i, j = idx  # 窗口中心的 (height_idx, width_idx)

            # 获取当前窗口内的 gate 和 mask
            window_start_h = max(0, i - self.window_size // 2)
            window_end_h = min(height, i + self.window_size // 2 + 1)
            window_start_w = max(0, j - self.window_size // 2)
            window_end_w = min(width, j + self.window_size // 2 + 1)

            gate_window = gate[:, window_start_h:window_end_h, window_start_w:window_end_w, :]
            mask_window = mask[window_start_h:window_end_h, window_start_w:window_end_w]

            # 计算窗口内的范数
            norms = torch.norm(gate_window, dim=3)  # (batch, window_height, window_width)

            # 计算窗口内的最大权重专家索引
            max_expert_indices = torch.argmax(gate_window, dim=3)  # (batch, window_height, window_width)

            # 初始化窗口内的相似性矩阵
            cos_sim_window = torch.zeros((batch_size, window_end_h - window_start_h, window_end_w - window_start_w), dtype=torch.float32).to(gate.device)

            # 遍历窗口内的偏移量
            for dx in range(-self.window_size // 2, self.window_size // 2 + 1):
                for dy in range(-self.window_size // 2, self.window_size // 2 + 1):
                    if dx == 0 and dy == 0:
                        continue  # 跳过自身

                    # 计算窗口内的点积和范数乘积
                    gate_rolled = torch.roll(gate_window, shifts=(dx, dy), dims=(1, 2))
                    norms_rolled = torch.roll(norms, shifts=(dx, dy), dims=(1, 2))
                    max_expert_indices_rolled = torch.roll(max_expert_indices, shifts=(dx, dy), dims=(1, 2))

                    dot_product = torch.einsum('bhwc,bhwc->bhw', gate_window, gate_rolled)
                    norm_product = norms * norms_rolled

                    # 计算余弦相似度
                    cos_sim_ij = dot_product / (norm_product + 1e-8)

                    # 判断最大权重专家是否相同
                    same_max_expert = max_expert_indices == max_expert_indices_rolled

                    # 累加相似性
                    cos_sim_window += cos_sim_ij * same_max_expert.float()

            # 计算窗口内的平均相似性
            cos_sim_window /= (self.window_size * self.window_size - 1)

            # 应用窗口内的 mask
            mask_window_expanded = mask_window.unsqueeze(0).expand(batch_size, -1, -1)
            cos_sim_final = cos_sim_window[mask_window_expanded]

            # 累加窗口的多样性损失
            diversity_loss += torch.mean(cos_sim_final)

        # 计算平均多样性损失
        diversity_loss /= num_samples

        return diversity_loss
    
    def region_compactness_loss(self,gate, mask):
        B, T, H, W, E = gate.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y_coords = y_coords.float().view(1, 1, H, W, 1).to(gate.device)  # [1, 1, H, W, 1]
        x_coords = x_coords.float().view(1, 1, H, W, 1).to(gate.device)
        
        # 所有专家并行计算 [B, T, H, W, E]
        expert_masks = gate * mask  # 直接广播mask到E维度
        
        # 计算均值 [B, T, E]
        sum_masks = expert_masks.sum(dim=(2,3)) + 1e-8  # [B, T, E]
        y_mean = (expert_masks * y_coords).sum(dim=(2,3)) / sum_masks  # [B, T, E]
        x_mean = (expert_masks * x_coords).sum(dim=(2,3)) / sum_masks  # [B, T, E]
        
        # 计算方差 [B, T, E]
        var_y = (expert_masks * (y_coords - y_mean.unsqueeze(2).unsqueeze(3))**2).sum(dim=(2,3)) / sum_masks
        var_x = (expert_masks * (x_coords - x_mean.unsqueeze(2).unsqueeze(3))**2).sum(dim=(2,3)) / sum_masks
        
        # 平均所有维度
        return (var_y + var_x).mean()
    
def print_grad_stats(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(
                f"{name}梯度 - 均值: {param.grad.mean().item():.6f}",
                f" 绝对值均值: {param.grad.abs().mean().item():.6f}",
                f" 最大值: {param.grad.max().item():.6f}",
                f" 非零比例: {(param.grad != 0).float().mean().item():.3f}"
            )
        else:
            print(f"{name}梯度为None")

class DownStream_DistanceLoss(nn.Module):
    def __init__(self, epsilon=0.1, window_size=3, num_sample_windows=10, diversity_weight_schedule=None):
        super(DownStream_DistanceLoss, self).__init__()
        self.epsilon = epsilon
        self.window_size = window_size  # 窗口大小
        self.num_sample_windows = num_sample_windows  # 采样窗口的数量
        self.diversity_weight_schedule = diversity_weight_schedule  # 多样性损失的权重调度

        # 损失权重
        self.lambda_mse = 1  # MSE 损失的权重

        self.lambda_load_balance = 0.1  # 负载均衡损失的权重
        self.lambda_smooth = 0.01
        # 损失函数
        self.mse = nn.MSELoss(reduction='none')  # MSE 损失函数
        # 掩码
        self.origin_mask = np.load(r'/home/zhouzirui/NYC_data/x/mask_64_64.npy')  # 掩码"/home/zhouzirui/Data/Chicago/mask.npy"  /home/zhouzirui/Data/iowa/mask_128_64.npy

    def forward(self, true, pred, gate, mode):
        """
        计算综合损失。
        :param true: 真实值，形状为 [batch_size, 7, height, width, feature]
        :param pred: 预测值，形状为 [batch_size, 7, height, width, feature]
        :param gate: 专家权重，形状为 [batch_size, height, width, num_experts]
        :param mode: 模式 ("pretrain" 或 "predict")
        :return: 综合损失、MSE 损失、负载均衡损失、空间平滑性损失
        """
        # 将掩码转换为布尔张量
        origin_mask = torch.tensor(self.origin_mask, dtype=torch.bool).to(pred.device)
        
        mask_expanded = origin_mask.unsqueeze(0).unsqueeze(-1).expand_as(pred)

        # 计算 MSE 损失
        mse_loss = self.mse(pred * mask_expanded, true * mask_expanded)
        mse_loss = mse_loss.mean()  # 计算平均损失

        #mae_loss = self.mae(pred * mask_expanded, true * mask_expanded)
        #mae_loss = mae_loss.mean()  # 计算平均损失
        # 负载均衡损失
  
        load_balance_loss = self.load_balance_loss(gate, mask_expanded)

        # 空间平滑性损失
        #spatial_loss = self.multi_scale_spatial_loss(gate, mask_expanded)
        #smooth_loss = self.smooth_loss(gate,mask_expanded)
        # 计算总损失
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_load_balance * load_balance_loss
            #self.lambda_smooth * smooth_loss
        )

        return total_loss, mse_loss

    def forward(self, true, pred, gate, mode):

        origin_mask = torch.tensor(self.origin_mask, dtype=torch.bool).to(pred.device)
        
        mask_expanded = origin_mask.unsqueeze(0).unsqueeze(-1).expand_as(pred)


        mse_loss = self.mse(pred * mask_expanded, true * mask_expanded)
        mse_loss = mse_loss.mean()  


  
        load_balance_loss = self.load_balance_loss(gate, mask_expanded)

        smooth_loss = self.smooth_loss(gate,mask_expanded)
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_load_balance * load_balance_loss 
        )

        return total_loss, mse_loss

    def load_balance_loss(self, gate, mask):

        mask = mask.unsqueeze(1).expand_as(gate)
        gate_masked = gate * mask  


        expert_usage = torch.sum(gate_masked, dim=(2, 3)) / torch.sum(mask, dim=(2, 3))  


        mean_usage = torch.mean(expert_usage, dim=-1, keepdim=True) 


        std_usage = torch.std(expert_usage, dim=-1, keepdim=True)  


        epsilon = 1e-8 
        coefficient_of_variation = std_usage / (mean_usage + epsilon)  


        load_balance_loss = coefficient_of_variation.mean()

        return load_balance_loss
    
    def smooth_loss(self, gate, mask):
        device = gate.device
        batch, T, H, W, num_experts = gate.shape
        window_size = self.window_size
        half_win = window_size // 2
        mask = mask.unsqueeze(1).expand_as(gate)
        gate = gate * mask
        with torch.no_grad():
            center_mask = mask.any(dim=0).any(dim=0)[..., 0]
            valid_positions = torch.nonzero(center_mask)
            if valid_positions.size(0) == 0:
                return torch.tensor(0.0, device=device)
            valid_y, valid_x = valid_positions.unbind(dim=1)

        pad = half_win
        padded_gate = F.pad(gate, (0,0,pad,pad,pad,pad))
        window_indices = torch.arange(window_size, device=device)
        y_coords = valid_y.view(-1,1,1) + window_indices.view(1,-1,1)
        x_coords = valid_x.view(-1,1,1) + window_indices.view(1,1,-1)
        windows = padded_gate[:, :, y_coords, x_coords, :]
        flat_windows = windows.permute(1,2,0,3,4,5).reshape(-1, window_size**2, num_experts)


        c = (window_size // 2) * window_size + (window_size // 2)  
        indices = torch.arange(window_size**2, device=device)
        j = indices[indices != c] 
        i = torch.full_like(j, c) 
        
        gate_i = flat_windows[:, i, :]  
        gate_j = flat_windows[:, j, :]
        cos_sim = F.cosine_similarity(gate_i, gate_j, dim=-1)
        

        valid_mask = self.get_valid_window_mask(mask, valid_y, valid_x, window_size)
        valid_pairs = valid_mask[:, i] & valid_mask[:, j]

        cos_sim_valid = cos_sim * valid_pairs.float()
        num_valid_pairs = valid_pairs.sum(dim=1).clamp(min=1)

        window_loss = -(cos_sim_valid.sum(dim=1) / num_valid_pairs).mean()
        return window_loss
    def get_valid_window_mask(self, mask, valid_y, valid_x, window_size):
        """
        生成窗口内有效位置的掩码
        mask形状: [B, T, H, W, 1]
        valid_y/x: [N_valid]
        返回: [N_valid*B*T, window_area]
        """
        B, T, H, W, _ = mask.shape
        N_valid = valid_y.size(0)
        half_win = window_size // 2
        
        # 生成窗口坐标偏移
        offsets = torch.arange(-half_win, half_win+1, device=mask.device)
        dy, dx = torch.meshgrid(offsets, offsets)
        
        # 计算所有窗口位置的全局坐标
        global_y = valid_y.reshape(-1,1,1) + dy.reshape(1,-1,1)  # [N_valid, W, W]
        global_x = valid_x.reshape(-1,1,1) + dx.reshape(1,1,-1)  # [N_valid, W, W]
        
        # 生成有效位置掩码（是否在原始图像范围内）
        valid_mask = (global_y >= 0) & (global_y < H) & (global_x >= 0) & (global_x < W)
        
        # 与原始mask结合 [N_valid, W, W]
        original_mask = mask[0,:,:,:,0].any(dim=0).float()  # 假设batch维度一致
        global_y = torch.clamp(global_y, 0, H - 1)
        global_x = torch.clamp(global_x, 0, W - 1)       
        valid_mask = valid_mask & (original_mask[global_y, global_x] > 0.5)
        
        # 扩展维度并重整 [N_valid*B*T, window_area]
        return valid_mask.view(N_valid, -1).repeat(B*T, 1)

PLOT_CONFIG = {
    "dpi": 300,               # 输出分辨率
    "cmap": "viridis",        # 颜色映射方案
    "file_format": "png",     # 保存格式(png/jpg/pdf)
    "layout": (8, 8)          # 子图排列 (rows, cols)
}   
def create_heatmaps(data, output_dir):
    """生成并保存所有热力图"""
    # 遍历每个特征
    #data = data.detach().cpu().numpy()
    output_dir = Path(r"batch_heatmaps/") 
    for feature_idx in range(4):
        # 创建带日期的子目录
        feature_dir = output_dir / f"feature_{feature_idx:03d}"
        feature_dir.mkdir(exist_ok=True)
        
        # 生成每天的热力图
        for day in range(data.shape[0]):
            plt.figure(figsize=(10, 8))
            # 绘制热力图
            plt.imshow(data[day,..., feature_idx].transpose(), 
                      cmap=PLOT_CONFIG["cmap"],
                      vmin=np.min(data[day,..., feature_idx]), 
                      vmax=np.max(data[day,..., feature_idx]),
                      origin='lower')
            
            plt.colorbar(label="Feature Value")
            plt.title(f"Day {day} - Feature {feature_idx}")
            plt.axis("off")
            
            # 保存单个文件
            filename = feature_dir / f"day_{day:02d}.{PLOT_CONFIG['file_format']}"
            plt.savefig(filename, 
                       dpi=PLOT_CONFIG["dpi"], 
                       bbox_inches="tight")
            plt.close()
            
            print(f"Saved: {filename}")