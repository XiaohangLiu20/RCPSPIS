#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:36:10 2024

@author: liuxiaohang
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

#User defined modules
from RCPSPISEnv import RCPSPISEnv
from network import Actor,Critic

def compute_node_features(state):
    durations = torch.FloatTensor(project_data['duration']).unsqueeze(1)
    resources_tensor = torch.FloatTensor(project_data['resources'])
    completed = torch.FloatTensor(state["completed"]).unsqueeze(1)
    ongoing = torch.FloatTensor(state["ongoing"]).unsqueeze(1)
    pending = torch.FloatTensor(state["pending"]).unsqueeze(1)
    realized_benefits = torch.FloatTensor(state["realized_benefits"]).unsqueeze(1)
    # remaining_time = torch.FloatTensor(state["remaining_time"]).unsqueeze(1)

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
        # remaining_time,
        resources_tensor,
        incentives,
        completed,
        ongoing,
        pending,
        realized_benefits
    ], dim=1)
    return node_features

def compute_global_info(state):
    current_time = torch.FloatTensor(state["current_time"])  # 当前时间
    available_resources = torch.FloatTensor(state["available_resources"])  # 保留资源的原始维度
    global_info = torch.cat([current_time, available_resources], dim=0).unsqueeze(0)
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

def compute_advantage(gamma, lmbda, td_delta, dones):
    td_delta = td_delta.detach().numpy()
    dones = dones.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta, done in zip(td_delta[::-1], dones[::-1]):
        if done:
            advantage = 0.0
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

# ===================== PPO Agent =====================
class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def store(self, state, action, logprob, reward, done, next_state):
        self.states.append(copy.deepcopy(state))
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(copy.deepcopy(next_state))


class PPOAgent:
    def __init__(self, actornet, criticnet, actor_lr, critic_lr, clip_range=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 gamma=0.99, lam=0.95, k_epoch = 1):
        self.actor = actornet
        self.critic = criticnet
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = RolloutBuffer()
        self.hyperparameters = {
            "clip_range": clip_range,
            "value_loss_coef": value_loss_coef,
            "entropy_coef": entropy_coef,
            "gamma": gamma,
            "lam": lam,
            "k_epoch": k_epoch
        }
        self.clip_range = self.hyperparameters["clip_range"]
        self.value_loss_coef = self.hyperparameters["value_loss_coef"]
        self.entropy_coef = self.hyperparameters["entropy_coef"]
        self.gamma = self.hyperparameters["gamma"]
        self.lam = self.hyperparameters["lam"]
        self.k_epoch = self.hyperparameters["k_epoch"]
        
    
    def get_action(self, node_features, global_info, action_mask):
        node_features = torch.stack([node_features])
        global_info = torch.stack([global_info])
        
        logits = self.actor(node_features, global_info)
        logits = logits.view(1, -1)
        masked_logits = logits.clone()

        # 屏蔽无效动作
        mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
        masked_logits[~mask_tensor] = -1e9
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # 计算动作对应的对数概率 (用于更新)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def select_action(self, node_features, global_info, action_mask):
        node_features = torch.stack([node_features])
        global_info = torch.stack([global_info])
        
        logits = self.actor(node_features, global_info)
        logits = logits.view(1, -1)
        masked_logits = logits.clone()

        # 屏蔽无效动作
        mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
        masked_logits[~mask_tensor] = -1e9
        probs = F.softmax(masked_logits, dim=-1)
        # print(probs)
        action = torch.argmax(probs, dim=-1)

        # 计算动作对应的对数概率 (用于更新)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def get_value(self, node_features, global_info):
        # get value of current state
        
        return self.critic(node_features, global_info)

    def store_transition(self, state, action, logprob, reward, done, next_state):
        self.buffer.store(state, action, logprob, reward, done, next_state)

    def update(self):
        
        states = self.buffer.states
        actions = torch.tensor(self.buffer.actions).view(-1, 1)
        old_logprobs = torch.tensor(self.buffer.logprobs).view(-1, 1)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float).view(-1, 1)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float).view(-1, 1)
        next_states = self.buffer.next_states
        
        # calculate cur_states values
        node_features = torch.stack([compute_node_features(state) for state in states])
        global_infos = torch.stack([compute_global_info(state) for state in states])
        masks = torch.tensor(np.array([compute_action_mask(state) for state in states]))
        
        old_values = self.get_value(node_features, global_infos).view(-1, 1)
        
        # calculate next_states values
        nxt_node_features = torch.stack([compute_node_features(next_state) for next_state in next_states])
        nxt_global_infos = torch.stack([compute_global_info(next_state) for next_state in next_states])
        nxt_values = self.get_value(nxt_node_features, nxt_global_infos).view(-1, 1)
        
        td_target = rewards + self.gamma * nxt_values * (1 - dones)
        # td_target = compute_advantage(1, 1, rewards, dones)
        td_delta = td_target - old_values
        advantages = compute_advantage(self.gamma, self.lam, td_delta, dones)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  #Normalized advantages
        
        actor_losses = []
        critic_losses = []
    
        # 对同一批样本进行多轮更新（PPO 的 K epochs 更新）
        for _ in range(self.k_epoch):
            # 前向传播
            logits = self.actor(node_features, global_infos)
            # beofre_probs = F.softmax(logits, dim=-1)
            logits = logits.squeeze(1).masked_fill(~masks, -1e9) 
            

            values = self.critic(node_features, global_infos).squeeze(-1)
    
            # 计算策略分布和损失
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            logprobs = torch.log(probs.gather(1, actions))
            entropy = dist.entropy().mean()
    
            # 计算 PPO 损失
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            actor_loss = policy_loss - self.entropy_coef * entropy
    
            # 计算 Critic 损失
            critic_loss = F.mse_loss(values, td_target.detach())
    
            # 总损失
            loss = policy_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
    
            # 反向传播和优化           
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
    
        # 清空 buffer
        self.buffer.reset()
        avg_loss1 = np.mean(actor_losses)
        avg_loss2 = np.mean(critic_losses)
        return avg_loss1,avg_loss2
        # avg_loss = np.mean(all_losses)
        # return avg_loss

    def save(self, file_path):
        """保存权重"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, file_path)
        print(f"PPOAgent saved to {file_path}")
    
    def load(self, file_path, map_location=None):
        """加载 PPOAgent，包括模型参数"""
        checkpoint = torch.load(file_path, map_location=map_location,  weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.to(map_location)
        self.critic.to(map_location)
        print(f"PPOAgent loaded from {file_path}")

    
if __name__ == "__main__":

    np.random.seed(36)
    torch.manual_seed(40)
    num_projects = 10
    resource_capacity = [15, 15]  # 示例资源容量
    max_time = 50  # 最大时间限制
    project_data = {
        'duration': np.random.randint(2, 10, num_projects),
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
     #将前置约束变成网络图
    edge_list = []
    for proj, preds in precedence_constraints.items():
        for p in preds:
            edge_list.append([p, proj])
    if len(edge_list) == 0:
        edge_list = [[i, i] for i in range(num_projects)]
    edge_index = torch.LongTensor(edge_list).t().contiguous()  # [2, num_edges]
    
    # 创建环境实例
    env = RCPSPISEnv(num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints)
    num_resource_types = project_data['resources'].shape[1]
    node_output_dim = 16
    global_dim = num_resource_types + 1 + node_output_dim
    node_input_dim = num_resource_types + 6 #节点特征的维度数 资源、时间、收益、激励以及状态

    # Initialize agent
    action_dim = num_projects + 1  # including skip
    actor_lr = 1e-4
    critic_lr = 1e-3
        
    actor = Actor(
        node_input_dim=node_input_dim,
        node_hidden_dim=64,
        node_output_dim=node_output_dim,
        global_dim=global_dim,
        attention_hidden_dim=128,
        final_state_dim=16,
        action_dim=action_dim,
        edge_index=edge_index
    )
    
    critic = Critic(
        node_input_dim=node_input_dim,
        node_hidden_dim=64,
        node_output_dim=node_output_dim,
        global_dim=global_dim,
        attention_hidden_dim=128,
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
        gamma=1,
        lam=0.95
    )

    max_episodes = 1000
    steps_per_update = 10
    actor_loss_list = []
    critic_loss_list = []
    reward_list = []
  
    for i in range(10):
        # 使用tqdm显示训练进度
        with tqdm(total=int(max_episodes/10), desc='Iteration %d' % i) as pbar:
            for ep in range(int(max_episodes/10)):
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
    
                        action, logprob = ppo.get_action(node_features, global_info, action_mask)
                        next_state, reward, done, _ = env.step(action)
    
                        ppo.store_transition(state, action, logprob, reward, done, next_state)
                        episode_reward += reward
                        state = next_state
                avg_ep_reward = episode_reward/steps_per_update
                # 更新策略
                # 计算下一状态特征
                
                actor_loss, critic_loss = ppo.update()
        
                # 存储训练结果
                reward_list.append(avg_ep_reward)
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
    
    file_path1 = 'PPO_Agent_model_output.pth'
    ppo.save(file_path1)
    
    
