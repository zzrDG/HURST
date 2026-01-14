import torch
import torch.optim as optim
import torch.utils.data
from Dataset import *
from model import *
from mask import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch.nn.functional as F
import os
import datetime
from torch.amp import GradScaler, autocast
import itertools
import logging
import matplotlib.colors as mcolors
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import json
import random  # 添加的导入
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步执行CUDA操作
os.environ["TORCH_USE_CUDA_DSA"] = "1"      # 启用设备端断言

# 设置设备
def configure_logging(log_file_path):

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    #logging.info("Logging started.")

# 保存检查点
def save_checkpoint(epoch, model, optimizer, train_loss_arr, valid_loss_arr, checkpoint_save_path):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_arr': train_loss_arr,
        'valid_loss_arr': valid_loss_arr
    }
    torch.save(checkpoint, checkpoint_save_path)

def load_checkpoint(model, optimizer, checkpoint_save_path):
 
    checkpoint = torch.load(checkpoint_save_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss_arr = checkpoint['train_loss_arr']
    valid_loss_arr = checkpoint['valid_loss_arr']
    train_mse_loss_arr = checkpoint['train_mse_loss_arr']
    valid_mse_loss_arr = checkpoint['valid_mse_loss_arr']
    return epoch, train_loss_arr, valid_loss_arr, train_mse_loss_arr, valid_mse_loss_arr

def save_config(config_dict, config_file_path):

    for key, value in config_dict.items():
        if isinstance(value, np.ndarray):
            config_dict[key] = value.tolist()  
    with open(config_file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)  

def load_and_preprocess_data(train_data_path, test_data_path, device):

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)[0:30]
    train_data = torch.from_numpy(train_data).to(device).to(torch.float)
    test_data = torch.from_numpy(test_data).to(device).to(torch.float)
    return train_data,  test_data

def visualize_leaf_gates(gate_output, leaf_expert_ids, epoch, expert_indices_save_path,mask):

    num_leaf_experts = len(leaf_expert_ids)


    custom_colors = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  
        (1.0, 0.4980392156862745, 0.054901960784313725),  
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  #
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), 
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  #
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  #
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  
    ]
    
    if num_leaf_experts > len(custom_colors):
        custom_colors = custom_colors * (num_leaf_experts // len(custom_colors) + 1)
    custom_colors = custom_colors[:num_leaf_experts]  
    custom_colors.append((0.2, 0.2, 0.2))  


    cmap = ListedColormap(custom_colors)

    top_expert_indices = torch.argmax(gate_output, dim=-1).squeeze().detach().cpu().numpy()  
    top_expert_indices = top_expert_indices.reshape(7,H, W)  
    mask = np.broadcast_to(mask, top_expert_indices.shape)  

    top_expert_indices_masked = top_expert_indices * mask  
    top_expert_indices_masked[mask == 0] = num_leaf_experts  


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

   
    im1 = ax1.imshow(top_expert_indices[6].transpose(), cmap=cmap, interpolation='nearest', origin='lower', vmin=0, vmax=num_leaf_experts)
    ax1.set_title("Top-1 Expert Indices (No Mask)")
    ax1.axis('off')


    im2 = ax2.imshow(top_expert_indices_masked[6].transpose(), cmap=cmap, interpolation='nearest', origin='lower', vmin=0, vmax=num_leaf_experts)
    ax2.set_title("Top-1 Expert Indices (With Mask)")
    ax2.axis('off')


    cbar = fig.colorbar(im2, ax=[ax1, ax2], ticks=np.arange(num_leaf_experts + 1))
    cbar.set_ticklabels([str(expert_id) for expert_id in leaf_expert_ids] + ['Masked'])
    cbar.set_label('Expert ID')

    plt.suptitle(f"Epoch {epoch + 1} Top-1 Expert Indices", fontsize=14, fontweight='bold')
    save_path = os.path.join(expert_indices_save_path, f"top_expert_indices_epoch_{epoch + 1}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def train(model, train_dataset,valid_dataset,train_dataloader, valid_dataloader,  optimizer, num_epochs, device, lambda_mse,lambda_load_balance ,lambda_contrastive,patience):
    seed = 42
    set_seed(seed)
    
    train_loss_arr = []
    valid_loss_arr = []
    train_mse_loss_arr = []
    valid_mse_loss_arr = []
    best_loss = np.inf
    loss_threshold = 1e-4  
    map_mask_np = np.load(r'')  
    map_mask = mask =torch.from_numpy(map_mask_np).to(torch.float).unsqueeze(0).expand(7,H,W).to(device)
    lambda_value = 1  
    weights = torch.ones((7, H, W), device=device)  
    persistent_expert_losses = {}
    criterion = DistanceLoss(epsilon=0.1, 
                            window_size= window_size,
                            num_sample_windows=20,
                            lambda_mse= lambda_mse,
                            lambda_load_balance = lambda_load_balance,
                            lambda_contrastive = lambda_contrastive
                            )
    mask, weights = generate_temporal_spatial_mask(map_mask, torch.rand(7,H, W).to(device), mask_rate=r, lambda_value=lambda_value,
                                                    temporal_dim=7, spatial_dim=(H, W), weights=weights,
                                                    top_ratio=0.2)
 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30,cooldown=0, min_lr=1e-05, eps=1e-08)
   
    
   
    early_stopping_counter = 0
    max_experts = model.moe_layer.get_max_experts()
    early_stopping_enabled = False

    for epoch in range(num_epochs):
        model.train()
        adjust_window = 5
        avg_train_loss = 0.0
        avg_train_mse_loss = 0.0
        avg_train_load_loss = 0.0
        avg_train_spatial_loss = 0.0
       
        
        for data,feature_idx_batch in train_dataloader:
       
            optimizer.zero_grad()
            
            output, gate_output, leaf_expert_ids,gate_logits = model(data, mask)

            train_loss, mse_loss, load_balancing_loss = criterion(data, 
                                                                                output, 
                                                                                gate_logits,
                                                                                mask,
                                                                                map_mask)
            train_loss.backward()

            optimizer.step()
            avg_train_loss += train_loss.item()
            avg_train_mse_loss += mse_loss.item()
            avg_train_load_loss += load_balancing_loss.item()


        avg_train_loss /= len(train_dataloader)
        avg_train_mse_loss /= len(train_dataloader)
        avg_train_load_loss /= len(train_dataloader)
   
        train_loss_arr.append(avg_train_loss)
        train_mse_loss_arr.append(avg_train_mse_loss)

        model.eval()
        avg_valid_loss = 0.0
        avg_valid_mse_loss = 0.0
        avg_valid_load_loss = 0.0
        avg_valid_spatial_loss = 0.0
        expert_losses = {}
        valid_loss_list = []  

        with torch.no_grad():
            for data,feature_idx_batch in valid_dataloader:

      
                output, gate_output, leaf_expert_ids,gate_logits = model(data,None)
            
                valid_loss, valid_mse_loss, valid_load_balancing_loss= criterion(data, 
                                                                                    output, 
                                                                                    gate_logits,
                                                                                    mask,
                                                                                    map_mask,
                                                                                    )
     
                positional_loss = (output - data) ** 2 
                valid_loss_list.append(positional_loss.detach())  
                for idx in feature_idx_batch.unique():
                        valid_dataset.update_feature_loss(idx.item(), valid_loss.item())
         
                avg_valid_loss += valid_loss.item()
                avg_valid_mse_loss += valid_mse_loss.item()
                avg_valid_load_loss += valid_load_balancing_loss.item()
           
                if leaf_expert_ids:
                    for i, expert_id in enumerate(leaf_expert_ids):
                        if expert_id not in expert_losses:
                            expert_losses[expert_id] = []
                        
               
                        gate_weight = gate_output[:, :, :, :, i].unsqueeze(-1) 
                        map_mask_expand = map_mask.unsqueeze(0).unsqueeze(-1).expand_as(output)

                        gate_weight = gate_weight * map_mask_expand
                        expert_loss = F.mse_loss(output * gate_weight , data * gate_weight ) / torch.sum(gate_weight)
                        expert_losses[expert_id].append(expert_loss.item())
        print(time.time())        
     
        avg_valid_loss /= len(valid_dataloader)
        avg_valid_mse_loss /= len(valid_dataloader)
        avg_valid_load_loss /= len(valid_dataloader)
      
        valid_loss_arr.append(avg_valid_loss)
        valid_mse_loss_arr.append(avg_valid_mse_loss)

        final_output = torch.cat(valid_loss_list, dim=0)  # [total_batch, 7, x, y]
        
        map_loss = torch.mean(final_output, dim=(0, -1), keepdim=False)  # 形状为 [7, x, y]
        # 生成掩码和权重
        mask, weights = generate_temporal_spatial_mask(map_mask, map_loss, mask_rate=r, lambda_value=lambda_value,
                                                    temporal_dim=7, spatial_dim=(H, W), weights=weights,
                                                    top_ratio=0.2)
        # 更新专家损失
        for expert_id, loss_list in expert_losses.items():
            avg_loss = sum(loss_list) / len(loss_list)
            if expert_id not in persistent_expert_losses:
                persistent_expert_losses[expert_id] = []
            persistent_expert_losses[expert_id].append(avg_loss)


        if len(valid_mse_loss_arr) >= adjust_window:
            # 计算最近窗口期内的损失变化率
            recent_losses = valid_mse_loss_arr[-adjust_window:]
            loss_change = np.abs(np.diff(recent_losses)).mean()  # 计算平均变化量
            loss_mean = np.mean(recent_losses)
            
            # 如果平均变化量小于平均损失的1%，则认为趋于稳定
            if loss_change < loss_mean * 0.01:
                # 调整lambda参数
                current_lambda_mse = max(current_lambda_mse * 0.5, 1)  # lambda_mse最低不低于0.1
                current_lambda_load_balance = min(current_lambda_load_balance * 2.0, 1)  # 最高不超过10.0
                
                # 更新损失函数的lambda值
                criterion.lambda_mse = current_lambda_mse
                criterion.lambda_load_balance = current_lambda_load_balance
                
                logging.info(f"Lambda values adjusted - MSE: {current_lambda_mse:.2f}, "
                            f"Load Balance: {current_lambda_load_balance:.2f}")
                

        if persistent_expert_losses:
            model.moe_layer.update_losses(persistent_expert_losses)

        # 记录日志和保存模型
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        num_experts = len(model.moe_layer.experts)

        if num_experts >= max_experts:
            early_stopping_enabled = True
        
        if avg_valid_mse_loss < best_loss:
            best_loss = avg_valid_mse_loss
            torch.save(model.state_dict(), model_param_path)
            visualize_leaf_gates(gate_output[0], leaf_expert_ids, epoch,expert_indices_save_path,map_mask_np)
            logging.info(f"Model saved at {model_save_path} with  mse_loss: {avg_valid_mse_loss:.6f}")
            if early_stopping_enabled:
                early_stopping_counter = 0
        elif early_stopping_enabled:
            early_stopping_counter += 1
        
            
        num_layers = model.moe_layer.get_current_depth()
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"{current_time}, Epoch {epoch + 1}, "
                    f"Training Loss: {avg_train_loss:.6f}, "
                    f"Training MSE Loss: {avg_train_mse_loss:.6f}, "
                    f"Training Load Loss: {avg_train_load_loss:.6f}, "
                    f"Training Spatial Loss:{avg_train_spatial_loss:.6f},"
                    f"Validation Loss: {avg_valid_loss:.6f}, "
                    f"Validation MSE Loss: {avg_valid_mse_loss:.6f}, "
                    f"Validation Load Loss: {avg_valid_load_loss:.6f}, "
                    f"Number of Experts: {num_experts}, "
                    f"Number of Layers: {num_layers}, "
                    f"Best Loss: {best_loss}, "
                    f"Learning Rate: {current_lr}")

        # 更新模型状态

        model.moe_layer.update_total_loss(avg_valid_mse_loss)
        model.moe_layer.check_and_add_experts(loss_threshold)

        if len(model.moe_layer.experts) > num_experts or epoch == 0:
            best_loss = np.inf
            torch.save(model, model_save_path)

        # 保存 MOE 结构
        model.moe_layer.print_or_save_structure(save_path=model_structure_path)

        plt.close('all')

        # 更新学习率调度器
        scheduler.step(avg_valid_loss)

        # 检查早停条件
        if early_stopping_enabled and early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # 保存训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_arr, label='Training Loss')
    plt.plot(valid_loss_arr, label='Validation Loss')
    plt.plot(train_mse_loss_arr, label='Training MSE Loss')
    plt.plot(valid_mse_loss_arr, label='Validation MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(version_dir, f"{current_date}_pre-training_validation_loss_curves.png"))
    plt.show()


def test(test_dataloader, criterion, device):
    model = torch.load(model_save_path)
    #model = torch.load(r'pth/pretrain-model/20241213_113018/20241213_113018.pth')
    #model.moe_layer.print_or_save_structure()
    model.eval()
    #criterion = nn.MSELoss()
    avg_test_loss = 0.0
    with torch.no_grad():
        for temporal_data, spatial_data, data in test_dataloader:

            output, gate_output, leaf_expert_ids = model(data,temporal_data, spatial_data, None)
            visualize_leaf_gates(gate_output[0], leaf_expert_ids, 1300)
            _,test_loss,_,_ = criterion(output, data, gate_output)
            #test_loss = criterion(output, data)
            avg_test_loss += test_loss.item()

    avg_test_loss /= len(test_dataloader)
    print(f'Test Loss: {avg_test_loss}')
    visualize_leaf_gates(gate_output[0], leaf_expert_ids, 1500)

    expert_info = []
    for expert in model.moe_layer.experts:
        expert_info.append({
            'id': expert.id,
            'is_leaf': expert.is_leaf
        })

    for info in expert_info:
        print(f"Expert ID: {info['id']}, Is Leaf: {info['is_leaf']}")

    leaf_count = sum(1 for info in expert_info if info['is_leaf'])
    print(f"Number of Leaf Experts: {leaf_count}")


# 主函数
def main():
    # 动态生成文件夹路径
    seed = 42  # 设置随机种子
    set_seed(seed)

    # 记录所有参数
    config = {
        "current_date": current_date,
        "num_days": num_days,
        "time_feature_dim": time_feature_dim,
        "spatial_feature_dim": spatial_feature_dim,
        "embed_dim": embed_dim,
        "moe_dim": moe_dim,
        "feedforward_dim": feedforward_dim,
        "kernel_size": kernel_size,  
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dropout": dropout,
        "num_epochs": num_epochs,
        "weight_top_ratio": weight_top_ratio,
        "r": r,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "train_size": train_size,
        "valid_size": valid_size,
    }

    # 保存配置到 JSON 文件
    save_config(config, config_file_path)

    # 配置日志
    configure_logging(log_file_path)
    logging.info(f"Configuration saved to {config_file_path}")
    #logging.info(f"Current configuration: {json.dumps(config, indent=4)}")

    # 模型初始化

    model = SpatioTemporalTransformer(
        time_feature_dim=time_feature_dim,
        spatial_feature_dim=spatial_feature_dim,
        embed_dim = embed_dim,
        encoder_dim = encoder_dim,
        moe_dim=moe_dim, 
        decoder_dim = decoder_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        kernel_size=kernel_size,
        mask_rate=r,
        max_experts=max_experts,
        max_depth=max_depth,
        device=device
    )

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

    # 训练和测试
    train(model, train_dataset,valid_dataset,train_dataloader, valid_dataloader,  optimizer, num_epochs, device,
          lambda_mse,lambda_load_balance ,lambda_contrastive,patience)
    #test(test_dataloader, criterion, device)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    seed = 42  # 设置随机种子
    set_seed(seed)
    # 设置设备
    device = torch.device("cuda:4")
    #time_feature = np.load(r'/home/zhouzirui/NYC_data/y/data/time_features.npy')[0:500]
    #spatial_feaure = np.load(r'/home/zhouzirui/NYC_data/y/data/space_features.npy')
    #index = [0,1,2,3,4,5]# accident
    #index = [0,1,2,3,4,6]# crime
    #index = [0,1,2,3,5,6]# Blocked Driveway
    #index = [0,1,2,4,5,6]# Illegal Parking
    #index = [6]
    index = [0,1,2,3,4,5,6]
    pretrain_data = np.load(r"/home/zhouzirui/Data/one-for-all/7F.npy") #spatial_temporal_feature
    #pretrain_data = np.load(r'/home/zhouzirui/Data/iowa/data.npy')#spatial_temporal_feature
    #pretrain_data = np.load(r'/home/zhouzirui/Data/iowa/data.npy')[...,index]
    pretrain_data = np.clip(pretrain_data, 0, None)
    #time_feature = torch.from_numpy(time_feature).to(torch.float).to(device)
    #spatial_feaure = torch.from_numpy(spatial_feaure).to(torch.float).to(device)
    pretrain_data = torch.from_numpy(pretrain_data).to(torch.float).to(device)
    #test_data = torch.from_numpy(test_data).to(device).to(torch.float)
    #train_data = torch.cat([time_feature,spatial_feaure,pretrain_data],dim=-1)
    #train_data = pretrain_data
    
     
    H = 64
    W = 64
    # 模型参数
    num_days = 7
    time_feature_dim = 15
    spatial_feature_dim = 18
    embed_dim = 64
    moe_dim = 128
    encoder_dim = 128
    decoder_dim = 128
    feedforward_dim = 256
    kernel_size = 5
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    max_depth = 3
    max_experts = 6  
    dropout = 0.1
    num_epochs = 1600
    weight_top_ratio = 0.4
    r = 0.4
    batch_size = 16
    lr = 0.001
    window_size = 5

    lambda_mse = 10
    lambda_load_balance = 0.01 
    lambda_contrastive = 0.001

    patience = 80
    # 数据划分
    train_size = int(0.6 * len(pretrain_data))
    valid_size = int(0.2 * len(pretrain_data))
    test_size = int(0.2 * len(pretrain_data))

    train_x = pretrain_data[0:train_size]
    valid_x = pretrain_data[train_size:train_size + valid_size]
    test_x = pretrain_data[train_size + valid_size:]
    
    train_dataset = a_Dataset(train_x)
    valid_dataset = a_Dataset(valid_x)
    test_dataset = a_Dataset(test_x)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=train_dataset.collate_fn,worker_init_fn=lambda worker_id: set_seed(seed + worker_id))
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0, collate_fn=valid_dataset.collate_fn,worker_init_fn=lambda worker_id: set_seed(seed + worker_id))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0,collate_fn=test_dataset.collate_fn,worker_init_fn=lambda worker_id: set_seed(seed + worker_id))

    param_dir = f"nyc-max_experts={max_experts}_r={r}_embed_dim={embed_dim}_moe_dim={moe_dim}_encoder_dim={encoder_dim}_window_size={window_size}_beta={weight_top_ratio}"
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = r'/home/zhouzirui/STMOE/pth/pretrain-model/one-for-all-ab_study_loss/'
    version_dir = os.path.join(base_dir, param_dir, current_date)  # 修改后的路径
    os.makedirs(version_dir, exist_ok=True)

    # 配置文件路径
    config_file_path = os.path.join(version_dir, "_config.json")
    log_file_path = os.path.join(version_dir, "_log.txt")
    model_save_path = os.path.join(version_dir, "model_structure.pth")
    model_param_path = os.path.join(version_dir, "model_param.pth")
    model_structure_path = os.path.join(version_dir, "_moe_structure.png")
    
    expert_indices_save_path = os.path.join(version_dir, "expert")
    os.makedirs(expert_indices_save_path, exist_ok=True)
    # 主函数
    main()