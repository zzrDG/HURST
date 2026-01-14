import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from mask import *
from utils import *
from model import *
from prompt import *

class ParallelMOE(nn.Module):
    def __init__(self, leaf_experts_A, leaf_expert_ids_A, d_model, output_dim, device):
       
        super(ParallelMOE, self).__init__()
        self.device = device
        self.H = 64
        self.W = 64
        self.experts = nn.ModuleList()  
        self.expert_ids = []  
        self.expert_mlps = nn.ModuleList()  
        for i, expert_A in enumerate(leaf_experts_A):
            
            depth = expert_A.current_depth
            expert = Expert(d_model, output_dim, is_leaf=True, current_depth=depth).to(device)
            expert.conv.weight.data.copy_(expert_A.conv.weight.data)
            expert.conv.bias.data.copy_(expert_A.conv.bias.data)
            expert.id = leaf_expert_ids_A[i]  
            self.experts.append(expert)
            self.expert_ids.append(expert.id)

         
            expert_mlp = MLP(output_dim, hidden_dims=64, output_dim=output_dim, num_layers=1)  
            self.expert_mlps.append(expert_mlp)

       
        self.gate = nn.Linear(d_model, len(self.experts))  
        self.gate_LM = nn.Linear(1, len(self.experts))

        

    def forward(self, x, tuning_LM, pre_gate):

        batch_size, seq_len, d_model = x.shape
        temperature = 0.1

      
        gate_weights = self.gate(x) + pre_gate  # (batch_size, seq_len, num_experts)
        if tuning_LM is not None:
            gate_LM_weights = self.gate_LM(tuning_LM)
            gate_weights = F.softmax((gate_weights + gate_LM_weights) / temperature, dim=-1)
        else:
            gate_weights = F.softmax(gate_weights / temperature, dim=-1)  

       
        expert_outputs = []
        for expert, expert_mlp in zip(self.experts, self.expert_mlps):
            expert_output = expert(x)  # (batch_size, seq_len, output_dim)
            expert_output = expert_mlp(expert_output)  
            expert_outputs.append(expert_output)

        
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch_size, seq_len, output_dim, num_experts)

        
        combined_output = torch.sum(expert_outputs * gate_weights.unsqueeze(2), dim=-1)  # (batch_size, seq_len, output_dim)

       
        combined_output = self.global_mlp(combined_output)  # (batch_size, seq_len, output_dim)

        return combined_output, gate_weights, self.expert_ids


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers):

        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dims
            out_dim = output_dim if i == num_layers - 1 else hidden_dims
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())  
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.reverse_3d =nn.ConvTranspose3d(
            in_channels=input_dim,  
            out_channels=hidden_dim,        
            kernel_size=7,    
            stride=4,         
            padding=3,        
            output_padding= (2, 3, 3) 
        )
        self.linear =nn.Linear(7,1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x,t,h,w):

        batch_size, seq_len,dim = x.shape
       
        x = x.permute(0,2,1)
        x = x.reshape(batch_size, dim,t,h,w)
        x = self.reverse_3d(x)
        H = x.shape[3]
        W = x.shape[4]
        x = x.permute(0,1,4,3,2)   #batch_size, dim,w,h,t
        x = self.linear(x).squeeze(-1)
        x = x.permute(0,2,3,1)
        x = self.mlp(x)   
        
        return x
    
class STpredictor(nn.Module):
    def __init__(self, pretrained_model, embeding_dim, encoder_dim, prompt_dim,  kernel_size,device):
        super(STpredictor, self).__init__()
        self.pretrained_model = pretrained_model
        self.prompt = SpatioTemporalPromptGenerator(input_dim=embeding_dim, 
                                                    output_dim=embeding_dim, 
                                                    kernel_size= kernel_size, 
                                                    key_dim=embeding_dim, 
                                                    num_keys=embeding_dim, 
                                                    value_dim=embeding_dim )
        self.l  = nn.Linear(64,64)
        self.predictor = Predictor(encoder_dim, 64, 1)


    def forward(self, x, prompt_flag):
        shape = x.shape  #[bs,T,H,W,1]
        #x = torch_normalization(x)s
        
        embed = self.pretrained_model.st_embed(x)
        batch_size, T, H, W, _ = embed.shape
        if prompt_flag:
            prompt = self.prompt(embed)   #[bs, H*W , value_dim]
            #prompt = torch.concat([ps, pt], dim=2)
            embed = embed + prompt
        
        #embed = embed.permute(0,4,1,2,3)  # [batch, H*W, T*hidden_dim]
        embed = self.pretrained_model.Embed_to_MoE(embed)  # [batch, H*W, model_hidden_dim]
        MoE_output, gate_output, leaf_expert_ids,gate_logits = self.pretrained_model.moe_layer(embed)
        MoE_output = MoE_output.permute(0,4,1,2,3)
        #MoE_output = self.l(embed)
        #MoE_output = MoE_output.permute(0,4,1,2,3)
        MoE_output = self.pretrained_model.Moe_to_Encoder(MoE_output)
        _,dim,t,h,w = MoE_output.shape
        MoE_output = MoE_output.permute(0,2,3,4,1).reshape(batch_size,-1,dim)
        enc_output = self.pretrained_model.encoder(MoE_output,t,h,w)
        pred = self.predictor(enc_output,t,h,w)
        pred = pred.view(shape[0], shape[2], shape[3],1)
        #return pred
        return pred, gate_output, leaf_expert_ids,gate_logits