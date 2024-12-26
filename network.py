import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv,global_mean_pool
from torch_geometric.data import Data




# ===================== GNN MODEL =====================
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        # 定义四层图卷积，每层的输入和输出维度
        self.conv1 = GraphConv(input_dim, hidden_dim)    
        self.conv2 = GraphConv(hidden_dim, hidden_dim)  
        self.conv3 = GraphConv(hidden_dim, hidden_dim)  
        self.conv4 = GraphConv(hidden_dim, output_dim)  

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层图卷积并激活
        x = F.relu(self.conv1(x, edge_index))
        
        # 第二层图卷积并激活
        x = F.relu(self.conv2(x, edge_index)) + x
        
        # 第三层图卷积并激活
        x = F.relu(self.conv3(x, edge_index)) + x
        # 第四层图卷积，无激活
        x = self.conv4(x, edge_index)
        # 图池化：平均池化操作
        h_G = global_mean_pool(x, batch)  # [batch_size, output_dim]
        return x, h_G

# ===================== ATTENTION LAYER =====================
class AttentionLayer(nn.Module):
    def __init__(self, node_embed_dim, global_info_dim, hidden_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.node_embed_dim = node_embed_dim
        self.global_info_dim = global_info_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.W_Q = nn.Linear(node_embed_dim, hidden_dim)
        self.W_K = nn.Linear(node_embed_dim, hidden_dim)
        self.W_V = nn.Linear(node_embed_dim, hidden_dim)
        
        self.W_Q_global = nn.Linear(global_info_dim, hidden_dim)
        self.W_K_global = nn.Linear(global_info_dim, hidden_dim)
        self.W_V_global = nn.Linear(global_info_dim, hidden_dim)
        
        self.W_output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, node_embeddings, global_information):
        
        Q_node = self.W_Q(node_embeddings)
        K_node = self.W_K(node_embeddings)
        V_node = self.W_V(node_embeddings)
        
        Q_global = self.W_Q_global(global_information)
        K_global = self.W_K_global(global_information)
        V_global = self.W_V_global(global_information)
        
        Q = Q_node + Q_global
        K = K_node + K_global
        V = V_node + V_global
        
        attention_scores = torch.softmax((Q @ K.transpose(-2, -1)) / (self.hidden_dim ** 0.5), dim=-1)
        weighted_sum = attention_scores @ V
        
        # Aggregate over nodes (e.g., take mean)
        aggregated = weighted_sum.mean(dim=1, keepdim=True)  # [1, hidden_dim]
        
        state_embeddings = self.W_output(aggregated)
        return state_embeddings  # [1, output_dim]

# ===================== ACTOR & CRITIC =====================
class Actor(nn.Module):
    def __init__(self, 
                 node_input_dim, 
                 node_hidden_dim, 
                 node_output_dim,
                 global_dim,
                 attention_hidden_dim,
                 final_state_dim,
                 action_dim,
                 edge_index):
        super(Actor, self).__init__()

        # 定义GNN和注意力模块
        self.gnn = GNNModel(node_input_dim, node_hidden_dim, node_output_dim)
        self.attention = AttentionLayer(node_embed_dim=node_output_dim,
                                        global_info_dim=global_dim,
                                        hidden_dim=attention_hidden_dim,
                                        output_dim=final_state_dim)
        
        # 定义全连接网络
        self.fc1 = nn.Linear(final_state_dim, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, 256)              
        self.fc3 = nn.Linear(256, 128)               
        self.fc4 = nn.Linear(128, action_dim)       
        
        # 保存边结构
        self.edge_index = edge_index


    def forward(self, node_features, global_info):
        # GNN 前向传播
        data = Data(x=node_features, edge_index=self.edge_index)
        node_embed,h_G = self.gnn(data)
        h_G = h_G.unsqueeze(1)
        # 将 h_G 加入到 global_info 中
        enhanced_global_info = torch.cat([global_info, h_G], dim=-1)
        # Attention 前向传播
        state_embed = self.attention(node_embed, enhanced_global_info)
        # Actor 网络
        x = F.relu(self.fc1(state_embed))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  
        return self.fc4(x)

# ===================== CRITIC =====================
class Critic(nn.Module):
    def __init__(self, 
                 node_input_dim, 
                 node_hidden_dim, 
                 node_output_dim,
                 global_dim,
                 attention_hidden_dim,
                 final_state_dim,
                 edge_index):
        super(Critic, self).__init__()

        # 定义GNN和注意力模块
        self.gnn = GNNModel(node_input_dim, node_hidden_dim, node_output_dim)
        self.attention = AttentionLayer(node_embed_dim=node_output_dim,
                                        global_info_dim=global_dim,
                                        hidden_dim=attention_hidden_dim,
                                        output_dim=final_state_dim)
        
        # 定义全连接网络
        self.fc1 = nn.Linear(final_state_dim, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, 128)              
        self.fc3 = nn.Linear(128, 64)               
        self.fc4 = nn.Linear(64, 1)
        
        # 保存边结构
        self.edge_index = edge_index

    def forward(self, node_features, global_info):
        # GNN 前向传播
        data = Data(x=node_features, edge_index=self.edge_index)
        node_embed,h_G = self.gnn(data)
        h_G = h_G.unsqueeze(1)
        # 将 h_G 加入到 global_info 中
        enhanced_global_info = torch.cat([global_info, h_G], dim=-1)
        # Attention 前向传播
        state_embed = self.attention(node_embed, enhanced_global_info)

        # Critic MLP 前向传播

        x = F.relu(self.fc1(state_embed))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value

