import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from mask import *
from utils import *
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import uuid
import os
from torch.nn.utils.weight_norm import weight_norm
from collections import defaultdict
from collections import OrderedDict

class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size=3, dropout=0.2):
        super(TemporalEmbedding, self).__init__()
        self.tcn = nn.Sequential(
            *[TemporalBlock(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, stride=1, dropout=dropout) 
              for i in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_dim, output_dim)  
        self.gelu =nn.GELU()
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.tcn(x)

        x = x.permute(0, 2, 3, 4, 1)
    
        x = self.fc(x)
        x = self.gelu(x)
        return x

class TemporalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv3d(input_dim, output_dim, kernel_size, stride=stride, padding=(kernel_size-1)//2))
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv3d(input_dim, output_dim, 1)
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SpatialEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super(SpatialEmbedding, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size-1) // 2,
            device=device
        )

    def forward(self, x):

        batch, T, H, W, feature = x.shape
        outputs = []

       
        for t in range(T):
            
            x_t = x[:, t, :, :, :]
         
            x_t = x_t.permute(0, 3, 1, 2)

            out_t = self.conv2d(x_t)  #

            outputs.append(out_t)


        output = torch.stack(outputs, dim=1) 
        output = output.permute(0,1,3,4,2)    
        return output

class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, input_features, model_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super().__init__()

        self.temporal_conv = nn.Conv3d(
            in_channels=input_features,
            out_channels=model_dim,
            kernel_size=(3, 1, 1),  
            padding=(1, 0, 0),
            device=device
        )
        self.spatial_conv = nn.Conv3d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=(1, 3, 3),  
            padding=(0, 1, 1),
            device=device
        )
        self.bn1 = nn.BatchNorm3d(model_dim)
        

        self.conv_block = nn.Sequential(
            nn.Conv3d(model_dim, model_dim, kernel_size, padding=padding,device=device),
            nn.BatchNorm3d(model_dim),
            nn.GELU(),
            nn.Conv3d(model_dim, model_dim, kernel_size, padding=padding,device=device),
            nn.BatchNorm3d(model_dim)
        ) 
        


        self.shortcut = nn.Sequential()
        if input_features != model_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_features, model_dim, kernel_size=1,device=device),
                nn.BatchNorm3d(model_dim)
            )

    def forward(self, x):
       
        x = x.permute(0, 4, 1, 2, 3)
        

        residual = self.shortcut(x)
        

        x = F.gelu(self.bn1(self.spatial_conv(self.temporal_conv(x))))


        x = self.conv_block(x)
        

        

        x += residual
        x = F.gelu(x)
        

        x = x.permute(0, 2, 3, 4, 1)  # [batch, days, H, W, features]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_time=7, max_height=128, max_width=128):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.max_time = max_time
        self.max_height = max_height
        self.max_width = max_width
        self.d_model = d_model
        

        position_t = torch.arange(0, max_time, dtype=torch.float).unsqueeze(1)
        div_term_t = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe_t = torch.zeros(max_time, d_model)
        pe_t[:, 0::2] = torch.sin(position_t * div_term_t)
        pe_t[:, 1::2] = torch.cos(position_t * div_term_t)
        self.register_buffer('pe_t', pe_t)  # [max_time, d_model]
        
 
        position_h = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        position_w = torch.arange(0, max_width, dtype=torch.float).unsqueeze(1)
        div_term_hw = torch.exp(torch.arange(0, d_model // 2, 2, dtype=torch.float) * -(math.log(10000.0) / (d_model // 2)))
        

        pe_h = torch.zeros(max_height, d_model // 2)
        pe_h[:, 0::2] = torch.sin(position_h * div_term_hw)
        pe_h[:, 1::2] = torch.cos(position_h * div_term_hw)
        

        pe_w = torch.zeros(max_width, d_model // 2)
        pe_w[:, 0::2] = torch.sin(position_w * div_term_hw)
        pe_w[:, 1::2] = torch.cos(position_w * div_term_hw)
        

        pe_h = pe_h.unsqueeze(1).expand(-1, max_width, -1)  # [max_height, max_width, d_model//2]
        pe_w = pe_w.unsqueeze(0).expand(max_height, -1, -1)  # [max_height, max_width, d_model//2]
        pe_hw = torch.cat([pe_h, pe_w], dim=-1)  # [max_height, max_width, d_model]
        self.register_buffer('pe_hw', pe_hw)

    def forward(self, x,t,h,w):
        """
        x: [batch_size, time_steps, height, width, d_model]
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size,t,h,w,-1)
        

        if t > self.max_time or h > self.max_height or w > self.max_width:
            raise ValueError("Input dimensions exceed maximum allowed dimensions.")
        

        x = x + self.pe_t[:t].unsqueeze(1).unsqueeze(1)  # [batch_size, time_steps, height, width, d_model]
        

        x = x + self.pe_hw[:h, :w].unsqueeze(0).unsqueeze(0)  # [batch_size, time_steps, height, width, d_model]
        x = x.reshape(batch_size,seq_len,-1)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # Initialize weights
        nn.init.normal_(self.W_Q.weight, mean=0, std=torch.sqrt(torch.tensor(2.0 / (d_model + d_k))))
        nn.init.normal_(self.W_K.weight, mean=0, std=torch.sqrt(torch.tensor(2.0 / (d_model + d_k))))
        nn.init.normal_(self.W_V.weight, mean=0, std=torch.sqrt(torch.tensor(2.0 / (d_model + d_v))))
        nn.init.normal_(self.W_out.weight, mean=0, std=torch.sqrt(torch.tensor(2.0 / (d_model + d_v))))

    def forward(self, Q, K, V, attn_mask=None, training=True):
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # Multi-head split
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)

        # Pre-process mask
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)  # Broadcast to all heads
            attn_mask = attn_mask.bool()

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)  # Attention weights
        if training:
            attns = self.dropout(attns)

        # Calculate output
        output = torch.matmul(attns, V)
        output = output.transpose(1, 2).contiguous().view(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src ,t,h,w,src_mask=None):
        src = self.pos_encoder(src ,t,h,w)
        output = self.transformer_encoder(src, mask=src_mask)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model // nhead, d_model // nhead, d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model // nhead, d_model // nhead, d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt2 = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt))
        tgt = tgt + self.dropout1(tgt2)

        # Encoder-decoder attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)

        # Feedforward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.pos_decoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, t,h,w,tgt_mask=None, memory_mask=None):
        tgt = self.pos_decoder(tgt, t,h,w)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)
        


class Linear_Gate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear_Gate, self).__init__()
        self.gate = nn.Linear(input_dim, output_dim)
        #self.LM_gate = nn.Conv2d(1, output_dim, kernel_size=kernel_size, padding=kernel_size//2)  # st_feature
        self._children = [] 
        self.id = uuid.uuid4()  

    def forward(self, x):
 
        x = x.to(self.gate.weight.device)

        gate_output = self.gate(x).unsqueeze(-1)
     
        gate_output = F.softmax(gate_output, dim=-2).to(device)
        

        temperature = 0.1  
        gate_output = F.softmax(gate_output  / temperature, dim=-2)
        return gate_output
        
    def children(self):
        return iter(self._children)

    def __repr__(self):
        return f"Gate(id={self.id})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id
    

class Expert(nn.Module):
    _id_counter = 0  
    
    def __init__(self, d_model, output_dim, is_leaf=True, current_depth=0, max_depth=3):
        super().__init__()
        self.id = Expert._id_counter  
        Expert._id_counter += 1  
        
 
        self.kernel_time = 1      
        self.kernel_space = 2**(max_depth - current_depth ) + 1    
        #self.kernel_size = 2**(max_depth - current_depth) + 1
        self.padding = (self.kernel_space - 1) // 2
        self.stride = (1, 1, 1)
        
        self.conv = nn.Conv3d(d_model, output_dim, 
                            kernel_size=(self.kernel_time, self.kernel_space, self.kernel_space),
                            padding=(0, self.padding, self.padding),
                            stride=self.stride,
                            device=device)
        self._children = []
        self.is_leaf = is_leaf
        self.parent_gate = None
        self.gate_weight = None
        self.gate_logits = None
        self.child_gate = None
        self.current_depth = current_depth
        self.max_depth = max_depth

    def add_child(self, child):
        self._children.append(child)
        self.is_leaf = False

    def forward(self, x):
        # [batch_size, num_days, height, width, dim] 
        x = x.permute(0, 4, 1, 2, 3)  # [batch, dim, t,h, w]
        x = self.conv(x)  # [batch, output_dim, t, h, w]
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1) # [batch, t,h, w, dim]
        return x

    def children(self):
        return iter(self._children)

    def __repr__(self):
        
        return f"Expert(id={self.id})"


class Gate(nn.Module):
    _id_counter = 0
    def __init__(self, input_dim, output_dim, kernel_size=5):
        super(Gate, self).__init__()
        self.kernel_time = 1      
        self.kernel_space = kernel_size  
        self.padding = (self.kernel_space - 1) // 2
        self.stride = (1, 1, 1)
        self.conv = nn.Conv3d(
                            input_dim, 
                            output_dim, 
                            kernel_size=(self.kernel_time, self.kernel_space, self.kernel_space),
                            padding=(0, self.padding, self.padding),
                            stride=self.stride,
                            device= device
                            )
        
        self.gelu = nn.GELU()
        
        self._children = []  
        self.id = Gate._id_counter
        Gate._id_counter += 1  
        self.temperature = 1 


    def forward(self, x):
        # [batch_size, num_days, height, width, dim]    
        x = x.to(self.conv.weight.device)
        
        x = x.permute(0, 4, 1, 2, 3)  # [batch, dim, num_days,height, width]
        logits = self.conv(x).permute(0, 2, 3, 4, 1)# [batch, num_days,height, width,experts]
 
        logits = self.gelu(logits)

        gumbel_logits = logits 
        temperature = 0.03  
        gate_output = F.softmax(gumbel_logits/ temperature, dim=-1)


        return gate_output, gate_output
        
    def children(self):
        return iter(self._children)

    def __repr__(self):
        return f"Gate(id={self.id})"

class DynamicMOE(nn.Module):
    def __init__(self, input_dim, output_dim, max_experts, max_depth, device):
        super(DynamicMOE, self).__init__()
        self.max_depth = max_depth  
        self.gates = nn.ModuleList([Gate(input_dim, 2, kernel_size=9).to(device)])
        self.experts = nn.ModuleList([Expert(input_dim, output_dim, is_leaf=True, current_depth=0, max_depth=max_depth).to(device)for _ in range(2)])
        
       
        self.expert_losses = {}  
        self.best_losses = {}  
        self.moe_structure = nx.DiGraph()  
        self.moe_structure.add_node(self.experts[0])
        self.moe_structure.add_node(self.experts[1])
        self.moe_structure.add_node(self.gates[0])
        self.moe_structure.add_edges_from([(self.gates[0], self.experts[0]), (self.gates[0], self.experts[1])])
        self.max_experts = max_experts
        self.total_losses = []
        self.total_loss_patience = 100  
        self.gates[0]._children = [self.experts[0], self.experts[1]]
        self.experts[0].parent_gate = self.gates[0]
        self.experts[1].parent_gate = self.gates[0]

        self.path_registry = defaultdict(list)

    
    def forward(self, x):
        x = x.to(device)
        leaf_expert_info = OrderedDict()  
        
        self._forward_recursive(
            x, self.gates[0], leaf_expert_info,
            parent_gate_weight=torch.ones(x.size(0), x.size(1),x.size(2),x.size(3)).to(device)
        )

        if not leaf_expert_info:
            return torch.zeros_like(x), None, [], None


        leaf_outputs = torch.stack([v[0] for v in leaf_expert_info.values()], dim=-2).to(device)  # [B, S, N, D]
        leaf_weights = torch.stack([v[1] for v in leaf_expert_info.values()], dim=-1).to(device)  # [B, S, N]
        leaf_logits = torch.stack([v[2] for v in leaf_expert_info.values()], dim=-1).to(device)  # [B, S, N]

        weighted_outputs = leaf_outputs * leaf_weights.unsqueeze(-1)
        output = weighted_outputs.sum(dim=-2)
        return output, leaf_weights, list(leaf_expert_info.keys()), leaf_logits

    def _forward_recursive(self, x, gate, leaf_expert_info, parent_gate_weight=None):
        gate_output, logits = gate(x)
        
        if parent_gate_weight is not None:
            gate_output = gate_output * parent_gate_weight.unsqueeze(-1)

        for expert_idx, expert in enumerate(gate._children):

            expert_weight = gate_output[..., expert_idx]
            combined_weight = expert_weight * (parent_gate_weight if parent_gate_weight is not None else 1.0)
            
            if expert.is_leaf:
                if expert.id not in leaf_expert_info:
                    expert_input = x * expert_weight.unsqueeze(-1)
                    expert_output = expert(expert_input)
                    leaf_expert_info[expert.id] = (
                        expert_output,
                        expert_weight,
                        logits[..., expert_idx]
                    )
            else:
                child_gate = expert.child_gate  
                self._forward_recursive(
                    x, 
                    child_gate,
                    leaf_expert_info,
                    parent_gate_weight=combined_weight
                )

    def get_leaf_experts_and_gates(self):
        leaf_experts = []
        leaf_gates = set()
        for expert in self.experts:
            if expert.is_leaf:
                leaf_experts.append(expert)
                leaf_gates.add(expert.parent_gate)
        return leaf_experts, leaf_gates
        
    def initialize_child_from_parent(self, child_conv1, child_conv2, parent_conv):

        child_kernel_size = child_conv1.kernel_size[1]
        parent_kernel_size = parent_conv.kernel_size[1]


        assert parent_kernel_size > child_kernel_size, "Parent kernel size must be larger than child kernel size"

        start = (parent_kernel_size - child_kernel_size) // 2
        end = start + child_kernel_size


        parent_weight = parent_conv.weight.data
        child_weight = parent_weight[:, :, :, start:end,start:end]

        scale_factor = 0.5  
        child_conv1.weight.data = child_weight * scale_factor
        child_conv2.weight.data = child_weight * scale_factor

      
        if child_conv1.bias is not None:
            nn.init.zeros_(child_conv1.bias)
            nn.init.zeros_(child_conv2.bias)

    def add_expert(self, parent_expert):
      
        if all(param.requires_grad is False for param in parent_expert.parameters()):
            print('Parent expert is already frozen.')
            return

        if parent_expert.current_depth >= self.max_depth:
            print(f"Cannot add expert: Parent expert depth {parent_expert.current_depth} exceeds max depth {self.max_depth}")
            return

      
        if len(self.experts) >= self.max_experts:
            print(f"Cannot add expert: Maximum number of experts {self.max_experts} reached")
            return

        new_current_depth = parent_expert.current_depth + 1

   
        new_expert1 = Expert(parent_expert.conv.in_channels, parent_expert.conv.out_channels, 
                            is_leaf=True, current_depth=new_current_depth, max_depth=self.max_depth).to(device)
        new_expert2 = Expert(parent_expert.conv.in_channels, parent_expert.conv.out_channels, 
                            is_leaf=True, current_depth=new_current_depth, max_depth=self.max_depth).to(device)


        self.initialize_child_from_parent(new_expert1.conv, new_expert2.conv, parent_expert.conv)


        for param in parent_expert.parameters():
            param.requires_grad = False


        if parent_expert.parent_gate:
            for param in parent_expert.parent_gate.parameters():
                param.requires_grad = False


        parent_expert.add_child(new_expert1)
        parent_expert.add_child(new_expert2)
        self.experts.append(new_expert1)
        self.experts.append(new_expert2)


        new_gate = Gate(parent_expert.conv.in_channels, 2, kernel_size=3).to(device)
        new_gate._children = [new_expert1, new_expert2]
        self.gates.append(new_gate)


        self.moe_structure.add_edges_from([(parent_expert, new_gate)])
        self.moe_structure.add_edges_from([(new_gate, new_expert1), (new_gate, new_expert2)])


        new_expert1.parent_gate = new_gate
        new_expert2.parent_gate = new_gate


        parent_expert.is_leaf = False
        parent_expert.child_gate = new_gate
        print(f"Added two new experts from parent expert {parent_expert.id}. New experts: {new_expert1.id}, {new_expert2.id}")

    def update_losses(self, losses):
        for expert_id, loss_list in losses.items():
            if expert_id not in self.expert_losses:
                self.expert_losses[expert_id] = []
            self.expert_losses[expert_id].extend(loss_list)  
            if expert_id not in self.best_losses or min(loss_list) < self.best_losses[expert_id]:
                self.best_losses[expert_id] = min(loss_list)  

    def check_and_add_experts(self, loss_threshold):
        if len(self.total_losses) < self.total_loss_patience:
            return 
        no_improvement_count = 0
       
        current_loss = self.total_losses[-1]
        best_loss = min(self.total_losses)
        

        for loss in reversed(self.total_losses):
                if loss > best_loss:
                    no_improvement_count += 1
                else:
                    break


        if len(self.total_losses) > 1:
            loss_change_rate = abs((current_loss - self.total_losses[-2]) / self.total_losses[-2])
        else:
            loss_change_rate = float("inf")  

        if (no_improvement_count >= self.total_loss_patience or 
            (loss_change_rate < loss_threshold and current_loss > best_loss)) and len(self.experts) < self.max_experts:

            worst_expert = None
            worst_loss = -np.inf
            for expert_id, losses in self.expert_losses.items():
                expert = next(expert for expert in self.experts if expert.id == expert_id)
                if expert.is_leaf:
                    current_loss = losses[-1]
                    if current_loss > worst_loss:
                        worst_loss = current_loss
                        worst_expert = expert

            if worst_expert is not None:
                print(f"Splitting expert {worst_expert.id} with current loss: {worst_loss}")
                self.add_expert(worst_expert)
            self.no_improvement_epochs = 0

            self.total_losses = []
            self.expert_losses = {}

    def print_or_save_structure(self,save_path):
        plt.close('all')  
        plt.figure(figsize=(20, 20))
        pos = nx.nx_agraph.graphviz_layout(self.moe_structure, prog='dot')
        nx.draw(self.moe_structure, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
        plt.title("MOE Structure")
    

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    

        plt.savefig(save_path)
        

    def get_current_depth(self):

        current_depth = 0
        for expert in self.experts:
            depth = self.get_expert_depth(expert)
            if depth > current_depth:
                current_depth = depth
        return current_depth

    def get_expert_depth(self, expert, depth=0):

        if not hasattr(expert, '_children') or len(expert._children) == 0:
            return depth
        max_depth = depth
        for child in expert._children:
            child_depth = self.get_expert_depth(child, depth + 1)
            if child_depth > max_depth:
                max_depth = child_depth
        return max_depth

    def update_total_loss(self, loss):
        self.total_losses.append(loss)
        
    
    def get_max_experts(self):
        return self.max_experts

    def __repr__(self):
        experts_str = "\n".join([f"  (Expert {i}): {expert}" for i, expert in enumerate(self.experts)])
        gates_str = "\n".join([f"  (Gate {i}): {gate}" for i, gate in enumerate(self.gates)])
        return f"DynamicMOE(\n{experts_str}\n{gates_str}\n)"   
    
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, 
                 time_feature_dim,             
                 spatial_feature_dim,         
                 embed_dim,
                 encoder_dim,                  
                 moe_dim,                      
                 decoder_dim,
                 nhead,                        
                 num_encoder_layers,           
                 num_decoder_layers,           
                 feedforward_dim,              
                 dropout,                     
                 kernel_size,                 
                 mask_rate,                  
                 max_experts,                 
                 max_depth,                    
                 device                        
                 ):
        super(SpatioTemporalTransformer, self).__init__()

        self.time_feature_dim = time_feature_dim
        self.spatial_feature_dim = spatial_feature_dim

        self.st_embed = SpatioTemporalEmbedding(
            input_features=1,
            model_dim=embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size-1) // 2
        )


        self.Embed_to_MoE = nn.Sequential(
            nn.Linear(embed_dim, moe_dim),
            nn.GELU()  
        )
        
        #MoE Layer
        self.moe_layer = DynamicMOE(
            input_dim=moe_dim,
            output_dim=moe_dim,
            max_experts=max_experts,
            max_depth=max_depth,
            device=device
        )   

        self.Moe_to_Encoder = nn.Conv3d(
            in_channels=moe_dim,
            out_channels=encoder_dim,
            kernel_size=7,  
            stride=4,       
            padding=3       
        )

   
        self.encoder = TransformerEncoder(
            d_model=encoder_dim,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout
        )


        self.decoder = TransformerDecoder(
            d_model=decoder_dim,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout
        )

        self.output_layer = nn.Sequential(
            nn.ConvTranspose3d(
            in_channels=encoder_dim,  
            out_channels=1,          
            kernel_size=7,  
            stride=4,       
            padding=3,      
            output_padding=(2, 3,3)  
        )

        )

    def forward(self, x, mask):
             
        embed = self.st_embed(x) 
        

        batch_size, T, H, W, _ = embed.shape
        if mask is not None:
            embed = st_mask(embed, mask) 


        embed = self.Embed_to_MoE(embed)

        MoE_output, gate_output, leaf_expert_ids,leaf_gate_logits = self.moe_layer(embed)

        MoE_output = MoE_output.permute(0,4,1,2,3)
        
        MoE_output = self.Moe_to_Encoder(MoE_output)

        batch_size,dim,t,h,w = MoE_output.shape
        MoE_output = MoE_output.permute(0,2,3,4,1).reshape(batch_size,-1,dim)
        enc_output = self.encoder(MoE_output,t,h,w)

        dec_output = self.decoder(enc_output, enc_output,t,h,w).reshape(batch_size,t,h,w,dim).permute(0,4,1,2,3)

        output = self.output_layer(dec_output)
        output = output.permute(0,2,3,4,1)

        return output, gate_output, leaf_expert_ids,leaf_gate_logits