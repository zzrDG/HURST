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


def configure_logging(log_file):
    logging.basicConfig(
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filename='', 
                    filemode='w'  
                )
    logging.info("Logging started.")

def visualize_leaf_gates(prediction, gate_output, leaf_expert_ids, epoch, expert_indices_save_path, all_label):
    # Load mask
    mask = np.load() 
    
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
    all_pred = prediction # [N, H, W, 1]
    all_label_np = all_label  # [N, H, W, 1]
    mask_np = mask  # [H, W]
    
    gate_assign_all = torch.argmax(gate_output, dim=-1).cpu().numpy()  # [N, T, H, W] 或 [N, H, W]
    
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
            'mape_sum': 0.0,  
            'pixel_count': 0,
            'non_zero_count': 0  
        }
    
    expert_stats['masked'] = {
        'mse_sum': 0.0,
        'mae_sum': 0.0,
        'mape_sum': 0.0,
        'pixel_count': 0,
        'non_zero_count': 0
    }
    
    mask_flat = mask_expanded.ravel() == 0
    gate_flat = gate_assign_all.ravel()
    pred_flat = all_pred.reshape(-1, 1).squeeze()
    label_flat = all_label_np.reshape(-1, 1).squeeze()
    
    errors = (pred_flat - label_flat) ** 2
    abs_errors = np.abs(pred_flat - label_flat)
    non_zero_mask = label_flat != 0  
    percentage_errors = np.zeros_like(abs_errors)
    percentage_errors[non_zero_mask] = abs_errors[non_zero_mask] / np.abs(label_flat[non_zero_mask])

    
    for expert_idx, expert_id in enumerate(leaf_expert_ids):
    
        expert_mask = (gate_flat == expert_idx) & (~mask_flat)
        pixel_indices = np.where(expert_mask)[0]
        
        if len(pixel_indices) > 0:
            expert_stats[expert_id]['mse_sum'] += np.sum(errors[pixel_indices])
            expert_stats[expert_id]['mae_sum'] += np.sum(abs_errors[pixel_indices])
            
            non_zero_indices = pixel_indices[non_zero_mask[pixel_indices]]
            if len(non_zero_indices) > 0:
                
                expert_stats[expert_id]['mape_sum'] += np.sum(percentage_errors[non_zero_indices])

            expert_stats[expert_id]['pixel_count'] += len(pixel_indices)
            expert_stats[expert_id]['non_zero_count'] += len(non_zero_indices)
    
    masked_indices = np.where(mask_flat)[0]
    if len(masked_indices) > 0:
        expert_stats['masked']['mse_sum'] += np.sum(errors[masked_indices])
        expert_stats['masked']['mae_sum'] += np.sum(abs_errors[masked_indices])
        
        non_zero_indices = masked_indices[non_zero_mask[masked_indices]]
        if len(non_zero_indices) > 0:
            expert_stats['masked']['mape_sum'] += np.sum(percentage_errors[non_zero_indices])
        
        expert_stats['masked']['pixel_count'] += len(masked_indices)
        expert_stats['masked']['non_zero_count'] += len(non_zero_indices)
    
    total_mse = 0.0
    total_mae = 0.0
    total_mape = 0.0
    total_pixels = 0
    total_non_zero = 0
    
    for expert_id in leaf_expert_ids + ['masked']:
        if expert_stats[expert_id]['pixel_count'] > 0:
 
            expert_stats[expert_id]['MSE'] = expert_stats[expert_id]['mse_sum'] / expert_stats[expert_id]['pixel_count']
            expert_stats[expert_id]['MAE'] = expert_stats[expert_id]['mae_sum'] / expert_stats[expert_id]['pixel_count']
            
         
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
    

    overall_mse = total_mse / total_pixels
    overall_mae = total_mae / total_pixels
    if total_non_zero > 0:
        overall_mape = (total_mape / total_pixels) * 100
    else:
        overall_mape = float('nan')
    

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

    configure_logging(log_file)
    prompt_flag = 1

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

    for epoch in range(num_epochs):
        start_time = time.time() 
        model.train()
        avg_train_loss = 0.0
        avg_train_mse_loss = 0.0

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
    
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch + 1}, Time: {epoch_time:.2f}s, Training Loss: {avg_train_loss},Training MSE_Loss: {avg_train_mse_loss}, Validation Loss: {avg_valid_loss}, Validation MSE_Loss: {avg_valid_mse_loss},Learning Rate: {current_lr}")

        scheduler.step(avg_valid_loss)


        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0  
            torch.save(model, model_save_path)  
            logging.info(f"Saved model with Validation Loss: {best_loss}")

            #visualize_leaf_gates(data[0],gate_output[0], leaf_expert_ids,epoch, expert_save_path)


        if avg_valid_loss >= best_loss:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered. Stopping training...")
                break


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
    
    criterion = nn.MSELoss()
    
    model = torch.load()
    model.eval()
    mask = np.load() 
    mask_tensor = torch.tensor(mask, dtype=torch.bool).to(device)
    all_predictions = []
    all_labels = []
    all_gate = []
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, label, feature_idx in test_dataloader:
            
            data = data.to(device)
            label = label.to(device)  
            

            output, gate_output, leaf_expert_ids, _ = model(data, prompt_flag=1)
            feature_idx = feature_idx[0].item() 
            
    
            output_cpu = output.cpu()
            label_cpu = label.cpu()
            
    
            output_denorm = test_dataset.data_denormalization(output_cpu, feature_idx)
            label_denorm = test_dataset.data_denormalization(label_cpu, feature_idx)
            

            
            output_flat = output_denorm.view(output_denorm.size(0), -1)
            label_flat = label_denorm.view(label_denorm.size(0), -1)
            mask_t = mask_tensor.unsqueeze(0).unsqueeze(-1).expand_as(label_denorm).view(label_denorm.size(0), -1).to(output_flat.device)

    
            mse = torch.mean((output_flat * mask_t - label_flat) ** 2, dim=1)
            mae = torch.mean(torch.abs(output_flat * mask_t - label_flat), dim=1)
            

            all_predictions.append(output_denorm)
            all_labels.append(label_denorm)
            all_gate.append(gate_output.cpu())
            
 
            total_mse += torch.sum(mse).item()
            total_mae += torch.sum(mae).item()
            
            true_value = label_denorm[0, :, :, 0].cpu().numpy()
            outputs_cpu = output_denorm[0, :, :, 0].cpu().numpy()
            gate_output_cpu = gate_output[0, :, :].cpu().numpy()
            total_samples += len(label)
            
            #expert_weights = torch.argmax(gate_output, dim=-1).squeeze().cpu().numpy()
            all_predictions.append(output_denorm.cpu())  
            all_labels.append(label_denorm.cpu())  

 
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
            
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    
    if len(all_predictions) > 0:  # 确保列表非空
       
        all_outputs_tensor = torch.cat(all_predictions, dim=0)
        all_targets_tensor = torch.cat(all_labels, dim=0)
    
    all_outputs_np = all_outputs_tensor.numpy()  
    all_targets_np = all_targets_tensor.numpy()  
    mse_per_sample = np.mean((all_outputs_np - all_targets_np)**2, axis=(1, 2, 3))



    print(f'Overall Test MSE: {avg_mse} MAE:{avg_mae}')
    return avg_mse, avg_mse 

def main():
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    index = [3,4,5,6]

    input_data = np.load()
    input_data = np.clip(input_data, 0, None)
    input_data = torch.from_numpy(input_data).to(device).float()
   
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


    train_dataset = preDataset(x_train_data)
    valid_dataset = preDataset(x_valid_data)
    test_dataset = preDataset(x_test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,collate_fn=valid_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=test_dataset.collate_fn)

 
    pretrained_model_save_path = r'pth/pretrain-model/one-for-all-ab_study_loss/nyc-max_experts=6_r=0.4_embed_dim=64_moe_dim=128_encoder_dim=128_window_size=5_beta=0.4/20250803_125454/model_structure.pth'
    
    pretrained_model = torch.load(pretrained_model_save_path,map_location=device)
    pretrained_params = torch.load(r'pth/pretrain-model/one-for-all-ab_study_loss/nyc-max_experts=6_r=0.4_embed_dim=64_moe_dim=128_encoder_dim=128_window_size=5_beta=0.4/20250803_125454/model_param.pth', 
                                   map_location=device)
    pretrained_model.load_state_dict(pretrained_params)




    for name, param in pretrained_model.moe_layer.named_parameters():
        if 'gate' or 'expert'in name:
            param.requires_grad = True  # 保留 parent_gate 的训练
        else:
            param.requires_grad = False  # 冻结其他参数


    for name, param in pretrained_model.encoder.named_parameters():
        if "attn" not in name and "norm" not in name:
            param.requires_grad = False

    st_embed_input_dim = 64  # 应该是 1
    encoder_dim = 128  # 应该是 encoder_dim


    downstream_model = STpredictor(pretrained_model,  st_embed_input_dim, encoder_dim,   prompt_dim, kernel_size,device)
   
    downstream_model = downstream_model.to(device)
   

    
    criterion = DownStream_DistanceLoss()
   
    optimizer = optim.Adam(downstream_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = r''
    run_dir = os.path.join(base_dir, current_time)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, f"{current_time}_log.txt")
    configure_logging(log_file)
    model_save_path = os.path.join(run_dir, "one-for-all.pth") 
    expert_save_path = os.path.join(run_dir, "experts")    
    os.makedirs(expert_save_path, exist_ok=True)
    config_file = os.path.join(run_dir, f"{current_time}_config.json")  


    config = {
        "current_time": current_time,
        "prompt_dim": prompt_dim,
        "kernel_size": kernel_size,  
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
    
    
    train(downstream_model, train_dataset, train_dataloader, valid_dataset,valid_dataloader, criterion, optimizer, num_epochs, device, log_file, expert_save_path, model_save_path)
    test(test_dataset,test_dataloader, criterion, device,  run_dir)

if __name__ == '__main__':

    main()

