# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *
import torch.optim as optim
from Dataset import *
from predictor import *
from tqdm import tqdm
import time
import matplotlib.colors as mcolors
import os
import torch.optim.lr_scheduler as lr_scheduler
import logging
import json
import datetime

# 配置日志
def configure_logging(log_file):
    logging.basicConfig(
                    level=logging.DEBUG,  # 设置日志级别为 DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
                    filename='/home/zhouzirui/STMOE/load_balancing.log',  # 日志文件名
                    filemode='w'  # 覆盖模式（每次运行时清空日志文件）
                )
    logging.info("Logging started.")

def visualize_leaf_gates(prediction, gate_output, leaf_expert_ids, epoch, expert_indices_save_path, all_label):
    # Load mask
    mask = np.load(r"/home/zhouzirui/Data/Chicago/chicago_mask.npy")  # (64, 64)
    
    num_leaf_experts = len(leaf_expert_ids)
    l  = len(prediction)
    # Custom color map
    custom_colors = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Blue
        (1.0, 0.4980392156862745, 0.054901960784313725),  # Orange
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # Green
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Red
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # Purple
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # Brown
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # Pink
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # Gray
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # Yellow
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # Cyan
    ]
    # Handle case when number of experts exceeds custom colors
    if num_leaf_experts > len(custom_colors):
        custom_colors = custom_colors * (num_leaf_experts // len(custom_colors) + 1)
    custom_colors = custom_colors[:num_leaf_experts]  # Trim to needed colors
    custom_colors.append((0.2, 0.2, 0.2))  # Mask color (dark gray)

    # Create custom color map
    cmap = ListedColormap(custom_colors)
    
    # Get top-1 expert indices
    top_expert_indices = torch.argmax(gate_output[5], dim=-1).squeeze().detach().cpu().numpy()

    top_expert_indices = top_expert_indices.reshape(7, 64, 80)
    mask = np.broadcast_to(mask, top_expert_indices.shape)  # (7, 128, 64)
    
    # Process masked regions
    top_expert_indices_masked = top_expert_indices * mask
    top_expert_indices_masked[mask == 0] = 100  # Set masked areas to special value
    # Create figure with two subplots   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot 1: Unmasked expert indices
    im1 = ax1.imshow(top_expert_indices[6].transpose(), cmap=cmap, interpolation='nearest', 
                     origin='lower', vmin=0, vmax=num_leaf_experts)
    ax1.set_title("Top-1 Expert Indices (No Mask)")
    ax1.axis('off')

    # Subplot 2: Masked expert indices
    im2 = ax2.imshow(top_expert_indices_masked[6].transpose(), cmap=cmap, interpolation='nearest', 
                     origin='lower', vmin=0, vmax=num_leaf_experts)
    ax2.set_title("Top-1 Expert Indices (With Mask)")
    ax2.axis('off')

    # Add color bar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], ticks=np.arange(num_leaf_experts + 1))
    cbar.set_ticklabels([str(expert_id) for expert_id in leaf_expert_ids] + ['Masked'])
    cbar.set_label('Expert ID')

    # Set title and save image
    plt.suptitle(f"Epoch {epoch + 1} Top-1 Expert Indices", fontsize=14, fontweight='bold')
    save_path = os.path.join(expert_indices_save_path, f"top_expert_indices_epoch_{epoch + 1}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # ===== MODIFIED STATISTICS CALCULATION SECTION =====
    # 修改：计算每个专家区域的预测误差
    all_pred = prediction # [N, H, W, 1]
    all_label_np = all_label  # [N, H, W, 1]
    mask_np = mask  # [H, W]
    
    # 获取门控输出（所有样本和时间步）
    # 注意：这里我们使用所有样本，而不仅是一个时间步
    gate_assign_all = torch.argmax(gate_output, dim=-1).cpu().numpy()  # [N, T, H, W] 或 [N, H, W]
    
    # 确保mask形状兼容
    if gate_assign_all.ndim == 4:
        mask_expanded = np.broadcast_to(mask_np, (gate_assign_all.shape[0], gate_assign_all.shape[1], 
                                                 mask_np.shape[1], mask_np.shape[2]))
    else:
        mask_expanded = np.broadcast_to(mask_np, (gate_assign_all.shape[0], mask_np.shape[0], mask_np.shape[1]))
    
    expert_stats = {}
    for expert_id in leaf_expert_ids:
        expert_stats[expert_id] = {
            'mse_sum': 0.0,
            'mae_sum': 0.0,
            'mape_sum': 0.0,  # MAPE 总和
            'pixel_count': 0,
            'non_zero_count': 0  # 非零真实值点数
        }
    
    # 添加masked区域
    expert_stats['masked'] = {
        'mse_sum': 0.0,
        'mae_sum': 0.0,
        'mape_sum': 0.0,
        'pixel_count': 0,
        'non_zero_count': 0
    }
    
    # 获取所有像素的坐标
    mask_flat = mask_expanded.ravel() == 0
    gate_flat = gate_assign_all.ravel()
    pred_flat = all_pred.reshape(-1, 1).squeeze()
    label_flat = all_label_np.reshape(-1, 1).squeeze()
    
    # 计算每个位置的总误差
    errors = (pred_flat - label_flat) ** 2
    abs_errors = np.abs(pred_flat - label_flat)
    # 计算 MAPE（仅对真实值非零的点）
    non_zero_mask = label_flat != 0  # 真实值非零的点
    percentage_errors = np.zeros_like(abs_errors)
    percentage_errors[non_zero_mask] = abs_errors[non_zero_mask] / np.abs(label_flat[non_zero_mask])
    # 处理每个专家区域
    print(abs_errors[non_zero_mask][1000],label_flat[non_zero_mask][1000])
    for expert_idx, expert_id in enumerate(leaf_expert_ids):
        # 找出属于该专家的像素（非mask区域）
        expert_mask = (gate_flat == expert_idx) & (~mask_flat)
        pixel_indices = np.where(expert_mask)[0]
        
        if len(pixel_indices) > 0:
            expert_stats[expert_id]['mse_sum'] += np.sum(errors[pixel_indices])
            expert_stats[expert_id]['mae_sum'] += np.sum(abs_errors[pixel_indices])
            
            # 只计算非零点的 MAPE
            non_zero_indices = pixel_indices[non_zero_mask[pixel_indices]]
            if len(non_zero_indices) > 0:
                
                expert_stats[expert_id]['mape_sum'] += np.sum(percentage_errors[non_zero_indices])

            expert_stats[expert_id]['pixel_count'] += len(pixel_indices)
            expert_stats[expert_id]['non_zero_count'] += len(non_zero_indices)
    
    # 处理masked区域
    masked_indices = np.where(mask_flat)[0]
    if len(masked_indices) > 0:
        expert_stats['masked']['mse_sum'] += np.sum(errors[masked_indices])
        expert_stats['masked']['mae_sum'] += np.sum(abs_errors[masked_indices])
        
        # 只计算非零点的 MAPE
        non_zero_indices = masked_indices[non_zero_mask[masked_indices]]
        if len(non_zero_indices) > 0:
            expert_stats['masked']['mape_sum'] += np.sum(percentage_errors[non_zero_indices])
        
        expert_stats['masked']['pixel_count'] += len(masked_indices)
        expert_stats['masked']['non_zero_count'] += len(non_zero_indices)
    
    # 计算各种误差指标
    total_mse = 0.0
    total_mae = 0.0
    total_mape = 0.0
    total_pixels = 0
    total_non_zero = 0
    
    for expert_id in leaf_expert_ids + ['masked']:
        if expert_stats[expert_id]['pixel_count'] > 0:
            # MSE 和 MAE
            expert_stats[expert_id]['MSE'] = expert_stats[expert_id]['mse_sum'] / expert_stats[expert_id]['pixel_count']
            expert_stats[expert_id]['MAE'] = expert_stats[expert_id]['mae_sum'] / expert_stats[expert_id]['pixel_count']
            
            # MAPE（仅在有非零点的点上计算）
            if expert_stats[expert_id]['non_zero_count'] > 0:
                expert_stats[expert_id]['MAPE'] = expert_stats[expert_id]['mape_sum'] / expert_stats[expert_id]['non_zero_count'] * 100
            else:
                expert_stats[expert_id]['MAPE'] = float('nan')
            
            total_mse += expert_stats[expert_id]['mse_sum']
            total_mae += expert_stats[expert_id]['mae_sum']
            total_mape += expert_stats[expert_id]['mape_sum']
            total_pixels += expert_stats[expert_id]['pixel_count']
            total_non_zero += expert_stats[expert_id]['non_zero_count']
        else:
            expert_stats[expert_id]['MSE'] = float('nan')
            expert_stats[expert_id]['MAE'] = float('nan')
            expert_stats[expert_id]['MAPE'] = float('nan')
    
    # 计算整体误差
    overall_mse = total_mse / total_pixels
    overall_mae = total_mae / total_pixels
    if total_non_zero > 0:
        overall_mape = (total_mape / total_pixels) * 100
    else:
        overall_mape = float('nan')
    
    # 打印统计信息
    print(f"\nExpert Region Error Statistics - Epoch {epoch + 1}:")
    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Expert ID", "MSE", "MAE", "MAPE (%)", "Pixel Count", "Non-zero Count"))
    
    for expert_id in leaf_expert_ids + ['masked']:
        stats = expert_stats[expert_id]
        mape_str = f"{stats['MAPE']:.2f}" if not np.isnan(stats['MAPE']) else "N/A"
        print("{:<10} {:<15.6f} {:<15.6f} {:<15} {:<15} {:<15}".format(
            expert_id, 
            stats['MSE'], 
            stats['MAE'], 
            mape_str, 
            stats['pixel_count'],
            stats['non_zero_count']))
    
    overall_mape_str = f"{overall_mape:.2f}" if not np.isnan(overall_mape) else "N/A"
    print(f"\nOverall MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}, MAPE: {overall_mape_str}%")
    
    # 保存统计信息
    stats_save_path = os.path.join(expert_indices_save_path, f"expert_error_stats_epoch_{epoch + 1}.txt")
    with open(stats_save_path, 'w') as f:
        f.write(f"Expert Region Error Statistics - Epoch {epoch + 1}\n")
        f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15}\n".format(
            "Expert ID", "MSE", "MAE", "MAPE (%)", "Pixel Count", "Non-zero Count"))
        
        for expert_id in leaf_expert_ids + ['masked']:
            stats = expert_stats[expert_id]
            mape_str = f"{stats['MAPE']:.2f}" if not np.isnan(stats['MAPE']) else "N/A"
            f.write("{:<10} {:<15.6f} {:<15.6f} {:<15} {:<15} {:<15}\n".format(
                expert_id, 
                stats['MSE'], 
                stats['MAE'], 
                mape_str, 
                stats['pixel_count'],
                stats['non_zero_count']))
        
        f.write(f"\nOverall MSE: {overall_mse:.6f}\n")
        f.write(f"Overall MAE: {overall_mae:.6f}\n")
        f.write(f"Overall MAPE: {overall_mape_str}%\n")
    
    print(f"\nExpert error statistics saved to: {stats_save_path}")

def train(model, train_dataset, train_dataloader, valid_dataset, valid_dataloader,criterion, optimizer, num_epochs, device, log_file, expert_save_path, model_save_path):
    train_loss_arr = []
    valid_loss_arr = []
    train_mse_loss_arr = []
    valid_mse_loss_arr = []
    best_loss = np.inf
    patience_counter = 0
    patience = 100
    # 配置日志
    configure_logging(log_file)
    prompt_flag = 1
    # 初始化学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)
    #ex_model = torch.load(r'pth/predict-model/20250218_112916/model.pth')
    #model.load_state_dict(ex_model.state_dict())
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录每一轮开始的时间
        model.train()
        avg_train_loss = 0.0
        avg_train_mse_loss = 0.0
        # 训练阶段
        for data, label ,feature_idx in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output, gate_output, leaf_expert_ids,gate_logits = model(data, prompt_flag)
            #outputs = model(data, prompt_flag)
            #label = train_dataset.data_denormalization(label,feature_idx)
            #output = train_dataset.data_denormalization(output,feature_idx)
            #gate_output = gate_output.to(device)
            train_loss, mse_loss = criterion(output, label, gate_logits,mode='predict')
            #train_loss = criterion(outputs, label)
            train_loss.backward()
            optimizer.step()
            avg_train_loss += train_loss.item()
            avg_train_mse_loss += mse_loss.item()
        avg_train_loss /= len(train_dataloader)
        avg_train_mse_loss /= len(train_dataloader)
        train_loss_arr.append(avg_train_loss)
        train_mse_loss_arr.append(avg_train_mse_loss)


        model.eval()
        avg_valid_loss = 0.0
        avg_valid_mse_loss = 0.0
        with torch.no_grad():
            for data, label,feature_idx in valid_dataloader:
                data = data.to(device)
                label = label.to(device)
                outputs, gate_output, leaf_expert_ids,gate_logits = model(data, prompt_flag)
                #outputs = model(data, prompt_flag)
                #outputs = valid_dataset.data_denormalization(outputs,feature_idx)
                #label = valid_dataset.data_denormalization(label,feature_idx)

                #gate_output = gate_output.to(device)
                v_loss,mse_loss, = criterion(outputs, label, gate_logits,mode='predict')
                #v_loss = criterion(outputs, label)
                avg_valid_loss += v_loss.item()
                avg_valid_mse_loss += mse_loss.item()

        avg_valid_loss /= len(valid_dataloader)
        avg_valid_mse_loss /= len(valid_dataloader)
        valid_loss_arr.append(avg_valid_loss)
        valid_mse_loss_arr.append(avg_valid_mse_loss)
        # 计算每一轮的时间
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        #logging.info(f"Epoch {epoch + 1}, Time: {epoch_time:.2f}s, Training Loss: {avg_train_loss},Training MSE:{avg_train_mse_loss} , Validation Loss: {avg_valid_loss}, ,Validation MSE:{avg_valid_mse_loss}, Learning Rate: {current_lr}")
        #print(f"Epoch {epoch + 1}, Time: {epoch_time:.2f}s, Training Loss: {avg_train_loss},Training MSE:{avg_train_mse_loss} , Validation Loss: {avg_valid_loss}, ,Validation MSE:{avg_valid_mse_loss}, Learning Rate: {current_lr}")
        print(f"Epoch {epoch + 1}, Time: {epoch_time:.2f}s, Training Loss: {avg_train_loss},Training MSE_Loss: {avg_train_mse_loss}, Validation Loss: {avg_valid_loss}, Validation MSE_Loss: {avg_valid_mse_loss},Learning Rate: {current_lr}")
        # 更新学习率调度器
        scheduler.step(avg_valid_loss)

        # 保存模型时绘制专家图
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0  # 重置计数器
            torch.save(model, model_save_path)  # 使用动态生成的模型保存路径
            logging.info(f"Saved model with Validation Loss: {best_loss}")

            # 绘制专家图
            #visualize_leaf_gates(data[0],gate_output[0], leaf_expert_ids,epoch, expert_save_path)

        # 检查早停条件
        if avg_valid_loss >= best_loss:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered. Stopping training...")
                break

    # 绘制损失曲线并保存
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_arr, label='Training Loss')
    plt.plot(valid_loss_arr, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(expert_save_path, "loss_curves.png"))
    plt.show()

    return train_loss_arr, valid_loss_arr

def test(test_dataset, test_dataloader, criterion, device,  run_dir):
    """
    :param model: 模型
    :param test_dataloader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 设备
    :param expert_ids: 叶节点专家的 ID 列表
    :param run_dir: 运行结果保存路径
    """
    criterion = nn.MSELoss()
    #model = torch.load(r'pth/predict-model/20241216_172158/model.pth')

    #model = torch.load(r'pth/predict-model/20250305_201515/one-for-all_model.pth',map_location=device)
    #model = torch.load(r'pth/predict_base_model/20250306_102527-zero-model/zero-shot-base-model.pth',map_location=device)
    model = torch.load(r'pth/predict_base_model/max_experts=4_r=0.4_embed_dim=32_moe_dim=64_encoder_dim=128_window_size=9/20250407_202235/one-for-all.pth',map_location=device)
    model.eval()
    mask = np.load(r"/home/zhouzirui/NYC_data/x/mask_64_64.npy") 
    mask_tensor = torch.tensor(mask, dtype=torch.bool).to(device)
    all_predictions = []
    all_labels = []
    all_gate = []
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, label, feature_idx in test_dataloader:
            # 1. 将原始输入数据移动到设备
            data = data.to(device)
            label = label.to(device)  # 如果需要计算损失，可能需要将标签也移动到设备
            
            # 2. 模型推理
            output, gate_output, leaf_expert_ids, _ = model(data, prompt_flag=1)
            feature_idx = feature_idx[0].item()  # 确保是标量值
            
            # 3. 将结果移回CPU进行反归一化
            output_cpu = output.cpu()
            label_cpu = label.cpu()
            
            # 4. 反归一化 - 确保在Dataset.py中已修复设备问题
            output_denorm = test_dataset.data_denormalization(output_cpu, feature_idx)
            label_denorm = test_dataset.data_denormalization(label_cpu, feature_idx)
            
            # 5. 后续处理使用反归一化后的值
            
            output_flat = output_denorm.view(output_denorm.size(0), -1)
            label_flat = label_denorm.view(label_denorm.size(0), -1)
            mask_t = mask_tensor.unsqueeze(0).unsqueeze(-1).expand_as(label_denorm).view(label_denorm.size(0), -1).to(output_flat.device)

            # 6. 计算误差
            mse = torch.mean((output_flat * mask_t - label_flat) ** 2, dim=1)
            mae = torch.mean(torch.abs(output_flat * mask_t - label_flat), dim=1)
            
            # 7. 保存结果
            all_predictions.append(output_denorm)
            all_labels.append(label_denorm)
            all_gate.append(gate_output.cpu())
            
            # 8. 累加误差
            total_mse += torch.sum(mse).item()
            total_mae += torch.sum(mae).item()
            # 绘制真实值和预测值
            true_value = label_denorm[0, :, :, 0].cpu().numpy()
            outputs_cpu = output_denorm[0, :, :, 0].cpu().numpy()
            gate_output_cpu = gate_output[0, :, :].cpu().numpy()
            total_samples += len(label)
            # 计算每个专家的权重
            #expert_weights = torch.argmax(gate_output, dim=-1).squeeze().cpu().numpy()
            all_predictions.append(output_denorm.cpu())  # 转移到 CPU，避免占用 GPU 内存
            all_labels.append(label_denorm.cpu())  # 转移到 CPU

            # 绘制真实值和预测值
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            vmin = np.min(true_value)
            vmax = np.max(true_value)
            im1 = ax1.imshow((true_value*mask).transpose(), cmap='hot', interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
            ax1.set_title('True Value')
            ax1.axis('off')
            im2 = ax2.imshow((outputs_cpu*mask).transpose(), cmap='hot', interpolation='nearest', origin='lower', vmin=vmin, vmax=vmax)
            ax2.set_title('Predictions')
            ax2.axis('off')
            cbar = fig.colorbar(im1, ax=[ax1, ax2])
            cbar.set_label('Value')
            plt.savefig(os.path.join(run_dir, f"test_prediction_{data.shape[0]}.png"))
            plt.close(fig)
            # 绘制每个专家的权重
    # 计算整体的 MSE 误差
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    #all_predictions = torch.cat(all_predictions, dim=0)
    #all_labels = torch.cat(all_labels, dim=0)
    #all_gate = torch.cat(all_gate,dim=0)
    
    if len(all_predictions) > 0:  # 确保列表非空
        # 修改：保存拼接后的张量，而不是直接覆盖列表
        all_outputs_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_labels, dim=0)
    
    all_outputs_np = all_outputs_tensor.numpy()  # 使用保存的张量
    all_targets_np = all_targets_tensor.numpy()  # 使用保存的张量
    mse_per_sample = np.mean((all_outputs_np - all_targets_np)**2, axis=(1, 2, 3))

    # 输出一些统计信息
    print("MSE统计信息:")
    print(f"最小值: {mse_per_sample.min():.4f}, 最大值: {mse_per_sample.max():.4f}")
    print(f"平均值: {mse_per_sample.mean():.4f}, 标准差: {mse_per_sample.std():.4f}")

    # 2. 计算MSE的变异系数
    epsilon = 1e-8  # 防止除零的小值

    # 变异系数 = 标准差 / 均值（用于衡量离散程度）
    cv_mse = np.std(mse_per_sample) / (np.mean(mse_per_sample) + epsilon)

    # 3. 可选：计算其他统计量
    # 计算空间变异系数（样本间对比）
    spatial_mean = np.mean(mse_per_sample)
    spatial_std = np.std(mse_per_sample)
    cv_mse_spatial = spatial_std / (spatial_mean + epsilon)

    # 4. 结果输出
    print("\n===== MSE变异系数结果 =====")
    print(f"MSE样本间变异系数: {cv_mse:.6f}")
    print(spatial_mean)
    print(spatial_std)

# 5. 扩展分析：空间和时间维度的变异系数
    def calculate_spatial_cv(data, epsilon=1e-8):
        """计算空间维度的变异系数"""
        # 数据形状: [样本数, 时间, 高度, 宽度, ...]
        space_mean = np.mean(data, axis=(1, 2, 3))
        space_std = np.std(data, axis=(1, 2, 3))
        return space_std / (space_mean + epsilon)

    # 对预测值和目标值分别计算空间变异系数
    cv_outputs_spatial = calculate_spatial_cv(all_outputs_np)
    cv_targets_spatial = calculate_spatial_cv(all_targets_np)

    print("\n===== 空间变异系数分析 =====")
    print(f"预测值平均空间变异系数: {np.mean(cv_outputs_spatial):.6f}")
    print(f"目标值平均空间变异系数: {np.mean(cv_targets_spatial):.6f}")

    # 6. 扩展分析：误差分布的空间模式
    # 计算平均绝对误差的空间分布
    abs_error = np.abs(all_outputs_np - all_targets_np)
    mean_abs_error = np.mean(abs_error, axis=0)  # 对所有样本取平均

    # 输出空间分布情况
    print("\n===== 空间平均绝对误差分布 =====")
    print(f"最小绝对误差: {mean_abs_error.min():.4f}")
    print(f"最大绝对误差: {mean_abs_error.max():.4f}")
    print(f"空间平均绝对误差: {np.mean(mean_abs_error):.4f}")
    #visualize_leaf_gates(all_outputs_np,all_gate[0], leaf_expert_ids, 0, os.path.join(run_dir, f"top_expert_indices_{data.shape[0]}"),all_targets_np)


    print(f'Overall Test MSE: {avg_mse} MAE:{avg_mae}')
    return avg_mse, avg_mse 

def main():
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    #index = [0,1,2,3,4,5]# accident
    #index = [0,1,2,3,4,6]# crime
    #index = [0,1,2,3,5,6]# Blocked Driveway
    #index = [0,1,2,4,5,6]# Illegal Parking
    index = [3,4,5,6]
    #index = [6]
    #input_data = np.load(r'/home/zhouzirui/Data/Chicago/chicago_data.npy')[...,0]
    input_data = np.load(r'/home/zhouzirui/Data/one-for-all/7F.npy')[...,5]
    #input_data = np.load(r"/home/zhouzirui/Data/Chicago/chicago_data.npy")[...,4]

    #input_data = np.load(r'/home/zhouzirui/Data/iowa/iowa_data.npy')[...,1]
    input_data = np.clip(input_data, 0, None)
    input_data = torch.from_numpy(input_data).to(device).float().unsqueeze(-1)
    #input_data = np.load(r'/home/zhouzirui/Data/features/illegal_parking.npy')
    #input_data = torch.from_numpy(input_data).to(device).float().unsqueeze(-1)


    #test_data = np.load(r'/home/zhouzirui/NYC_data/y/data/zero_shot_finetune_features.npy')[292:365,...,-1]
    #test_data = torch.from_numpy(test_data).to(device).float().unsqueeze(-1)
    prompt_dim = 128
    kernel_size = 7

    dropout = 0.1
    num_epochs = 1000
    r = 0
    batch_size = 16
    lr = 0.00001

    train_size = int(0.6 * len(input_data))
    valid_size = int(0.2 * len(input_data))
    test_size = len(input_data) - train_size - valid_size
    
    x_train_data = input_data[:train_size]
    x_valid_data = input_data[train_size:train_size + valid_size]
    x_test_data = input_data[train_size + valid_size:]
    #x_test_data = test_data

    train_dataset = preDataset(x_train_data)
    valid_dataset = preDataset(x_valid_data)
    test_dataset = preDataset(x_test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,collate_fn=valid_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=test_dataset.collate_fn)

    # 加载预训练模型
    pretrained_model_save_path = r'pth/pretrain-model/one-for-all-ab_study_loss/nyc-max_experts=6_r=0.4_embed_dim=64_moe_dim=128_encoder_dim=128_window_size=5_beta=0.4/20250803_125454/model_structure.pth'
    #pretrained_model_save_path = r'pth/pretrain-model/20250223_182118/20250223_182118.pth'
    #pretrained_model_save_path = r'pth/pretrain-model/one-for-all/20250313_212306/20250313_212306.pth'
    pretrained_model = torch.load(pretrained_model_save_path,map_location=device)
    pretrained_params = torch.load(r'pth/pretrain-model/one-for-all-ab_study_loss/nyc-max_experts=6_r=0.4_embed_dim=64_moe_dim=128_encoder_dim=128_window_size=5_beta=0.4/20250803_125454/model_param.pth', 
                                   map_location=device)
    pretrained_model.load_state_dict(pretrained_params)


    # 冻结指定层的参数
    #for name, param in pretrained_model.st_embed.named_parameters():
    #    param.requires_grad = False

    for name, param in pretrained_model.moe_layer.named_parameters():
        if 'gate' or 'expert'in name:
            param.requires_grad = True  # 保留 parent_gate 的训练
        else:
            param.requires_grad = False  # 冻结其他参数


     #冻结 encoder 中除了多头注意力和归一化层之外的所有参数
    for name, param in pretrained_model.encoder.named_parameters():
        if "attn" not in name and "norm" not in name:
            param.requires_grad = False

    st_embed_input_dim = 64  # 应该是 1
    encoder_dim = 128  # 应该是 encoder_dim

    # 初始化下游任务模型
    downstream_model = STpredictor(pretrained_model,  st_embed_input_dim, encoder_dim,   prompt_dim, kernel_size,device)
    #downstream_model = torch.load(r'pth/predict-model/20250108_203233/model.pth')
    downstream_model = downstream_model.to(device)
    #for name, param in downstream_model.named_parameters():
    #    print(f"{name}: {param.requires_grad}")

    # 定义损失函数和优化器
    criterion = DownStream_DistanceLoss()
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(downstream_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")# 动态生成文件夹路径
    base_dir = r'pth/predict_base_model/nyc-no_loss/'
    run_dir = os.path.join(base_dir, current_time)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, f"{current_time}_log.txt")# 日志文件路径
    configure_logging(log_file)
    model_save_path = os.path.join(run_dir, "one-for-all.pth") # 模型保存路径
    expert_save_path = os.path.join(run_dir, "experts")    # 专家图保存路径
    os.makedirs(expert_save_path, exist_ok=True)
    config_file = os.path.join(run_dir, f"{current_time}_config.json")    # 配置文件路径

    # 记录所有参数
    config = {
        "current_time": current_time,
        "prompt_dim": prompt_dim,
        "kernel_size": kernel_size,  # 处理 kernel_size
        "dropout": dropout,
        "num_epochs": num_epochs,
        "r": r,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "device": str(device)
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Configuration saved to {config_file}")
    
    # 训练和测试
    #train(downstream_model, train_dataset, train_dataloader, valid_dataset,valid_dataloader, criterion, optimizer, num_epochs, device, log_file, expert_save_path, model_save_path)
    test(test_dataset,test_dataloader, criterion, device,  run_dir)

if __name__ == '__main__':
    main()