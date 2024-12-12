#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:31:52 2024

@author: liuxiaohang
"""

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===================== ENVIRONMENT =====================
class RCPSPISEnv(gym.Env):
    def __init__(self, num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints):
        super(RCPSPISEnv, self).__init__()
        
        self.num_projects = num_projects
        self.project_data = project_data  # {'duration': [], 'resources': [], 'base_benefit': []}
        self.enhancement_matrix = enhancement_matrix  #值应该大于等于1 1表示没有激励
        self.precedence_constraints = precedence_constraints
    
        self.resource_capacity = np.array(resource_capacity)
        self.max_time = max_time

        # Observation space
        self.observation_space = spaces.Dict({
            "completed": spaces.MultiBinary(num_projects),
            "ongoing": spaces.MultiBinary(num_projects),
            "pending": spaces.MultiBinary(num_projects),
            "available_resources": spaces.Box(0, np.inf, shape=(len(resource_capacity),), dtype=np.float32),
            "current_time": spaces.Box(0, max_time, shape=(1,), dtype=np.float32),
            "realized_benefits": spaces.Box(0, np.inf, shape=(num_projects,), dtype=np.float32),
        })

        # Action space
        self.action_space = spaces.Discrete(num_projects + 1)

        self.reset()
        
    def reset(self):
        self.completed = np.zeros(self.num_projects, dtype=int)
        self.ongoing = np.zeros(self.num_projects, dtype=int)
        self.pending = np.array([1 if len(self.precedence_constraints.get(i, [])) == 0 else 0 for i in range(self.num_projects)], dtype=int)
        self.available_resources = self.resource_capacity.copy()
        self.current_time = 0
        self.project_end_times = np.zeros(self.num_projects, dtype=int)
        self.accumulated_benefit = 0
        self.realized_benefits = self.project_data['base_benefit'].copy()
        return self.get_state()
    
    def get_state(self):
        return {
            "completed": self.completed,
            "ongoing": self.ongoing,
            "pending": self.pending,
            "available_resources": self.available_resources,
            "current_time": np.array([self.current_time], dtype=np.float32),
            "realized_benefits": self.realized_benefits
        }
    
    def _update_pending(self):
        for i in range(self.num_projects):
            if self.completed[i] == 1 or self.ongoing[i] == 1:
                self.pending[i] = 0
            else:
                precedence_satisfied = all(self.completed[pre] == 1 for pre in self.precedence_constraints.get(i, []))
                if precedence_satisfied:
                    resource_feasible = np.all(self.available_resources >= self.project_data['resources'][i])
                    time_feasible = self.current_time + self.project_data['duration'][i] <= self.max_time
                    if resource_feasible and time_feasible:
                        self.pending[i] = 1
                    else:
                        self.pending[i] = 0
                else:
                    self.pending[i] = 0
    
    def reward(self):
        if np.sum(self.completed) == 0:
            return 1
        incentive = np.prod(self.enhancement_matrix**self.completed[:, None], axis=0)
        enhance_benefit = incentive * self.realized_benefits
        qe = (1 + enhance_benefit) * self.completed
        qe[qe == 0] = 1
        r = np.prod(qe)
        return r
    
    def step(self, action):
        reward = 0
        done = False
        
        if action == self.num_projects:  # Skip
            if np.any(self.ongoing):
                next_completion_time = np.min(self.project_end_times[self.ongoing > 0])
                assert next_completion_time > self.current_time, "Time update error!"
                time_duration = next_completion_time - self.current_time
                self.current_time = next_completion_time
                completed_projects = np.where(self.project_end_times == self.current_time)[0]
                unit_reward = self.reward()
                reward = unit_reward * time_duration
                self.accumulated_benefit += reward
                for project in completed_projects:
                    self.ongoing[project] = 0
                    self.completed[project] = 1
                    self.available_resources += self.project_data['resources'][project]
                    base_benefit = self.project_data['base_benefit'][project]
                    std_dev = base_benefit * 0.1
                    self.realized_benefits[project] = np.random.normal(base_benefit, std_dev)
                self._update_pending()
            
            elif np.all(self.pending == 0):
                done = True
                next_completion_time = self.max_time
                time_duration = next_completion_time - self.current_time
                self.current_time = next_completion_time
                unit_reward = self.reward()
                reward = unit_reward * time_duration
                self.accumulated_benefit += reward
            else:
                raise ValueError("Invalid action: Skip, there are pending projects!")
        
        else:  # Launch a project
            if self.pending[action]:
                self.pending[action] = 0
                self.ongoing[action] = 1
                self.available_resources -= self.project_data['resources'][action]
                self.project_end_times[action] = self.current_time + self.project_data['duration'][action]
                self._update_pending()
            else:
                raise ValueError(f"Invalid action: {action}. Project {action} is not pending!")
 
        state = self.get_state()
        return state, reward, done, {}

# ===================== GNN MODEL =====================
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        # 定义第一层图卷积
        self.conv1 = GraphConv(input_dim, hidden_dim)
        # 定义第二层图卷积
        self.conv2 = GraphConv(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层图卷积操作，然后使用ReLU激活函数
        x = F.relu(self.conv1(x, edge_index))
        # 第二层图卷积操作
        x = self.conv2(x, edge_index)
        return x

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
        num_nodes = node_embeddings.size(0)
        
        repeated_global_info = global_information.repeat(num_nodes, 1)
        
        Q_node = self.W_Q(node_embeddings)
        K_node = self.W_K(node_embeddings)
        V_node = self.W_V(node_embeddings)
        
        Q_global = self.W_Q_global(repeated_global_info)
        K_global = self.W_K_global(repeated_global_info)
        V_global = self.W_V_global(repeated_global_info)
        
        Q = Q_node + Q_global
        K = K_node + K_global
        V = V_node + V_global
        
        attention_scores = torch.softmax((Q @ K.transpose(-2, -1)) / (self.hidden_dim ** 0.5), dim=-1)
        weighted_sum = attention_scores @ V
        
        # Aggregate over nodes (e.g., take mean)
        aggregated = weighted_sum.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
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
        self.fc1 = nn.Linear(final_state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
        # 保存边结构
        self.edge_index = edge_index


    def forward(self, node_features, global_info):
        # GNN 前向传播
        data = Data(x=node_features, edge_index=self.edge_index)
        node_embed = self.gnn(data)
        # Attention 前向传播
        state_embed = self.attention(node_embed, global_info)
        # Actor 网络
        x = F.relu(self.fc1(state_embed))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim = -1)

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
        self.fc1 = nn.Linear(final_state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # 保存边结构
        self.edge_index = edge_index

    def forward(self, node_features, global_info):
        # GNN 前向传播
        data = Data(x=node_features, edge_index=self.edge_index)
        node_embed = self.gnn(data)

        # Attention 前向传播
        state_embed = self.attention(node_embed, global_info)

        # Critic MLP 前向传播
        x = F.relu(self.fc1(state_embed))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value


# ===================== PPO TRAINING =====================
class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.node_features = []
        self.global_info = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        self.action_masks = []

    def store(self, node_feat, global_info, action, logprob, value, reward, done, action_mask):
        self.node_features.append(node_feat)
        self.global_info.append(global_info)
        self.actions.append(action)
        self.logprobs.append(logprob.detach())
        self.rewards.append(torch.FloatTensor([reward]))
        self.values.append(value.detach())
        self.is_terminals.append(torch.FloatTensor([float(done)]))
        self.action_masks.append(action_mask)

    def compute_returns_and_advantages(self, next_value, gamma=0.99, lam=0.95):
        # 拼接奖励和值
        rewards = torch.cat(self.rewards)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        values = torch.cat(self.values).squeeze(-1)
        dones = torch.cat(self.is_terminals)
        next_value = next_value.detach().squeeze(-1)  
        values = torch.cat([values, next_value])
        # values = (values - values.mean()) / (values.std() + 1e-8)
        # 优势计算
        advantages = [] 
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam  * gae
            advantages.insert(0, gae)
        # 将优势转换为张量
        advantages = torch.tensor(advantages, dtype=torch.float32)
        # 打印最终结果    
        returns = advantages + values[:-1]
        return returns, advantages
    

class PPOAgent:
    def __init__(self, actornet, criticnet, actor_lr, critic_lr, clip_range=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 gamma=0.99, lam=0.95):
        self.actor = actornet
        self.critic = criticnet
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.buffer = RolloutBuffer()

    def get_action(self, node_features, global_info, action_mask):
        epsilon = 0.5
        logits = self.actor(node_features, global_info)
        masked_logits = logits.clone()

        # 屏蔽无效动作
        mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
        masked_logits[~mask_tensor] = -1e9
        probs = F.softmax(masked_logits, dim=-1)

        if np.random.rand() < epsilon:  # epsilon-greedy策略
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)

        # 计算当前值函数估计
        value = self.critic(node_features, global_info)

        # 计算动作对应的对数概率 (用于更新)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def store_transition(self, node_features, global_info, action, logprob, value, reward, done, action_mask):
        self.buffer.store(node_features, global_info, action, logprob, value, reward, done, action_mask)

    def update(self, next_node_features, next_global_info, next_action_mask):
        with torch.no_grad():
            next_value = self.critic(next_node_features, next_global_info)
        returns, advantages = self.buffer.compute_returns_and_advantages(next_value, self.gamma, self.lam)
        # advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)  #advantage归一化
        # returns = (returns - returns.mean())/(returns.std()+1e-8)
        actor_losses = []
        critic_losses = []
        # all_losses = []
        for idx in range(len(self.buffer.actions)):
            # 获取每个存储的数据样本
            nf = self.buffer.node_features[idx]
            gi = self.buffer.global_info[idx]
            action = self.buffer.actions[idx]
            old_logprob = self.buffer.logprobs[idx]
            value = self.buffer.values[idx].squeeze(-1)
            ret = returns[idx]
            adv = advantages[idx]
            action_mask = torch.BoolTensor(self.buffer.action_masks[idx]).unsqueeze(0)

            # 前向传播
            logits = self.actor(nf, gi)
            value_pred = self.critic(nf, gi)
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e9
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logprob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            # 损失计算
            ratio = torch.exp(new_logprob - old_logprob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(value_pred.squeeze(-1), torch.tensor([ret]))
            actor_loss = policy_loss - self.entropy_coef * entropy

            # 反向传播与更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # loss = actor_loss + critic_loss
            # all_losses.append(loss.item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.buffer.reset()
        avg_loss1 = np.mean(actor_losses)
        avg_loss2 = np.mean(critic_losses)
        return avg_loss1,avg_loss2
        # avg_loss = np.mean(all_losses)
        # return avg_loss
    
def compute_node_features(state):
    durations = torch.FloatTensor(project_data['duration']).unsqueeze(1)
    resources_tensor = torch.FloatTensor(project_data['resources'])
    completed = torch.FloatTensor(state["completed"]).unsqueeze(1)
    ongoing = torch.FloatTensor(state["ongoing"]).unsqueeze(1)
    pending = torch.FloatTensor(state["pending"]).unsqueeze(1)
    realized_benefits = torch.FloatTensor(state["realized_benefits"]).unsqueeze(1)

    # Incentive: 对pending项目计算增强乘积，其他项目为1
    completed_mask = (state["completed"] == 1)
    incentives = []
    for i in range(num_projects):
        if pending[i] == 1:
            incentive_val = np.prod(enhancement_matrix[completed_mask, i])
            incentives.append(incentive_val)
        else:
            incentives.append(1.0)
    incentives = torch.FloatTensor(incentives).unsqueeze(1)

    node_features = torch.cat([
        durations,
        resources_tensor,
        incentives,
        completed,
        ongoing,
        pending,
        realized_benefits
    ], dim=1)
    return node_features

def compute_global_info(state):
    current_time = torch.FloatTensor(state["current_time"])
    available_sum = torch.FloatTensor([state["available_resources"].sum()])
    global_info = torch.cat([current_time, available_sum]).unsqueeze(0)
    return global_info

def compute_action_mask(state):
    # 根据三种case逻辑设定
    action_mask = (state["pending"] == 1)
    # skip逻辑
    if np.any(state["pending"]) == 1 and np.any(state["ongoing"]) == 0:
        # 不允许skip
        action_mask = np.append(action_mask, False)
    else:
        # 允许skip
        action_mask = np.append(action_mask, True)
    return action_mask

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# ===================== TEST EXAMPLE =====================
if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(42)
    
    num_projects = 10
    resource_capacity = [15, 15]  # 示例资源容量
    max_time = 50  # 最大时间限制
    project_data = {
        'duration': np.random.randint(2, 8, num_projects),
        'resources': np.random.randint(1, 5, (num_projects, 2)),
        'base_benefit': np.round(np.random.uniform(0.1, 0.5, num_projects), 2)
    }
    
    enhancement_matrix = np.zeros((num_projects, num_projects))
    sparse_indices = np.random.choice(np.arange(num_projects ** 2), size=int(num_projects ** 2 * 0.2), replace=False)
    for idx in sparse_indices:
        i, j = divmod(idx, num_projects)
        if i != j:  # 不允许自增强
            enhancement_matrix[i, j] = np.round(np.random.uniform(0, 0.3), 2)
    enhancement_matrix += 1
    
    precedence_constraints = {}
    for i in range(2, num_projects):
        predecessors = np.random.choice(range(i), size=np.random.randint(1, 3), replace=False).tolist()
        precedence_constraints[i] = predecessors
    
    # 创建环境实例
    env = RCPSPISEnv(num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints)
    #将前置约束变成网络图
    edge_list = []
    for proj, preds in precedence_constraints.items():
        for p in preds:
            edge_list.append([p, proj])
    if len(edge_list) == 0:
        edge_list = [[i, i] for i in range(num_projects)]
    edge_index = torch.LongTensor(edge_list).t().contiguous()  # [2, num_edges]

    # Global info dimension: let's say global info = [current_time, sum of available_resources]
    # We'll build this inside the loop from env state
    global_dim = 2
    num_resource_types = project_data['resources'].shape[1]
    node_input_dim = num_resource_types + 6 #节点特征的维度数 资源、时间、收益、激励以及状态

    # Initialize agent
    action_dim = num_projects + 1  # including skip
    actor_lr = 1e-5
    critic_lr = 1e-4
        
    actor = Actor(
        node_input_dim=node_input_dim,
        node_hidden_dim=128,
        node_output_dim=16,
        global_dim=global_dim,
        attention_hidden_dim=16,
        final_state_dim=16,
        action_dim=action_dim,
        edge_index=edge_index
    )
    
    critic = Critic(
        node_input_dim=node_input_dim,
        node_hidden_dim=128,
        node_output_dim=16,
        global_dim=global_dim,
        attention_hidden_dim=16,
        final_state_dim=16,
        edge_index=edge_index
    )
    
    
    ppo = PPOAgent(
        actornet = actor,
        criticnet = critic,
        actor_lr = actor_lr, 
        critic_lr = critic_lr,
        clip_range=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        lam=0.95
    )

    max_episodes = 1000
    steps_per_update = 50
    actor_loss_list = []
    critic_loss_list = []
    reward_list = []
  
    for i in range(10):
        # 使用tqdm显示训练进度
        with tqdm(total=int(max_episodes/10), desc='Iteration %d' % i) as pbar:
            for ep in range(int(max_episodes/10)):
                state = env.reset()
                ppo.buffer.reset()
                episode_reward = 0
                # done = False
        
                # 环境交互阶段
                # 采样阶段：环境交互与数据存储 采集数据量：steps_per_update次 game交互
                for step in range(steps_per_update):
                    done = False
                    state = env.reset()
                    while not done:
                        node_features = compute_node_features(state)
                        global_info = compute_global_info(state)
                        action_mask = compute_action_mask(state)
    
                        action, logprob, value = ppo.get_action(node_features, global_info, action_mask)
                        next_state, reward, done, _ = env.step(action)
    
                        ppo.store_transition(node_features, global_info, action, logprob, value, reward, done, action_mask)
                        episode_reward += reward
                        state = next_state
                avg_ep_reward = episode_reward/steps_per_update
                # 更新策略
                # 计算下一状态特征
                next_node_features = compute_node_features(state)
                next_global_info = compute_global_info(state)
                next_action_mask = compute_action_mask(state)
                actor_loss, critic_loss = ppo.update(next_node_features, next_global_info, next_action_mask)
        
                # 存储训练结果
                reward_list.append(episode_reward)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                # 更新进度条
                if (ep+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (max_episodes/10 * i + ep+1), 'return': '%.3f' % np.mean(reward_list[-10:])})
                pbar.update(1)
    
    # 绘制结果
    act_loss_mov = moving_average(actor_loss_list, 9)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, max_episodes+1), act_loss_mov, 'r-', label='Loss')
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Actor Loss Over Episodes")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    critic_loss_mov = moving_average(critic_loss_list, 9)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, max_episodes+1), critic_loss_mov, 'r-', label='Loss')
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Critic Loss Over Episodes")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    reward_mov = moving_average(reward_list, 9)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, max_episodes+1), reward_mov, 'b-', label='Reward')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Reward Over Episodes")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    
    
def evaluate_policy(env, ppo_model, num_projects, max_time, resource_capacity, project_data, precedence_constraints):
    """
    使用训练好的 PPO 模型完成项目调度游戏，输出策略和收益。
    
    Args:
        env: 环境实例
        ppo_model: 训练好的 PPO 模型
        num_projects: 项目数量
        max_time: 最大调度时间
        resource_capacity: 资源容量
        project_data: 项目相关数据
        precedence_constraints: 项目的先行约束关系

    Returns:
        strategy: 调度策略（完成的项目及顺序）
        total_reward: 累计收益
    """
    state = env.reset()  # 初始化环境
    strategy = []  # 调度策略
    done = False

    while not done:
        # 提取当前状态特征
        node_features = compute_node_features(state)
        global_info = compute_global_info(state)
        action_mask = compute_action_mask(state)

        # 使用模型选择动作
        action, _, _ = ppo_model.get_action(node_features, global_info, action_mask)
        # 执行动作，更新环境
        next_state, reward, done, info = env.step(action)
        strategy.append(action)
        print(f"Action: Project {action}, Reward: {reward}")
        state = next_state
    total_reward = env.accumulated_benefit

    return strategy, total_reward

strategy, total_reward = evaluate_policy(
    env, ppo, num_projects, max_time, resource_capacity, project_data, precedence_constraints
)    
print("Strategy:", strategy)
print("Total Reward:", total_reward)    
    
    
    