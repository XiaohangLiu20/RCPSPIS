#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:13:04 2024

@author: liuxiaohang
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import pandas as pd
import seaborn as sns
import time

#User defined modules
from RCPSPISEnv import RCPSPISEnv
from network import Actor,Critic

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))



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
    def __init__(self, actornet, criticnet, actor_lr, critic_lr, device, clip_range=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 gamma=0.99, lam=0.95, k_epoch = 4):
        self.actor = actornet
        self.critic = criticnet
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = RolloutBuffer()
        self.device = device
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
        node_features = torch.stack([node_features]).to(self.device)
        global_info = torch.stack([global_info]).to(self.device)
        
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
        node_features = torch.stack([node_features]).to(self.device)
        global_info = torch.stack([global_info]).to(self.device)
        
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
    
    def compute_advantage(self, td_delta, dones):
        td_delta = td_delta.detach().numpy()
        dones = dones.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta, done in zip(td_delta[::-1], dones[::-1]):
            if done:
                advantage = 0.0
            advantage = self.gamma * self.lam * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def compute_node_features(self,state):
        durations = torch.FloatTensor(state['duration']).unsqueeze(1)
        resources_tensor = torch.FloatTensor(state['resources'])
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
                incentive_val = np.prod(state["enhancement"][completed_mask, i])
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

    def compute_global_info(self, state):
        current_time = torch.FloatTensor(state["current_time"])  # 当前时间
        available_resources = torch.FloatTensor(state["available_resources"])  # 保留资源的原始维度
        global_info = torch.cat([current_time, available_resources], dim=0).unsqueeze(0)
        return global_info

    def compute_action_mask(self, state):
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
 

    def update(self):
        
        states = self.buffer.states
        actions = torch.tensor(self.buffer.actions).view(-1, 1).to(self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs).view(-1, 1)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = self.buffer.next_states
        
        # calculate cur_states values
        node_features = torch.stack([self.compute_node_features(state) for state in states]).to(self.device)
        global_infos = torch.stack([self.compute_global_info(state) for state in states]).to(self.device)
        masks = torch.tensor(np.array([self.compute_action_mask(state) for state in states]))
        
        old_values = self.get_value(node_features, global_infos).view(-1, 1)
        
        # calculate next_states values
        nxt_node_features = torch.stack([self.compute_node_features(next_state) for next_state in next_states]).to(device)
        nxt_global_infos = torch.stack([self.compute_global_info(next_state) for next_state in next_states]).to(device)
        nxt_values = self.get_value(nxt_node_features, nxt_global_infos).view(-1, 1)
        
        td_target = rewards + self.gamma * nxt_values * (1 - dones)
        # td_target = compute_advantage(1, 1, rewards, dones)
        td_delta = td_target - old_values
        advantages = self.compute_advantage(td_delta.cpu(), dones.cpu()).to(self.device)
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
            ratio = torch.exp(logprobs.cpu() - old_logprobs).to(self.device)
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
        

def load_data_and_create_envs(file_path, num, max_time, project_num):
    """
    读取文件夹中的数据并创建环境实例。
    
    :param file_path: 文件夹路径
    :param num: 从文件夹中读取文件的数量
    :return: 生成的环境池（列表）
    """
    envs = []  # 存储生成的环境实例
    files = sorted(os.listdir(file_path))[:num]  # 读取文件夹中的文件，限制数量
    
    for file_name in files:
        if not file_name.endswith(".xlsx"):
            continue  # 跳过非 Excel 文件
        
        file_path_full = os.path.join(file_path, file_name)
        print(str(file_path_full))
        try:
            # 按照 sheet 名读取数据
            enhancement_matrix = pd.read_excel(file_path_full, sheet_name="Incentive Coefficients", header=None).to_numpy()
            enhancement_matrix = 1 + enhancement_matrix
            precedence_data = pd.read_excel(file_path_full, sheet_name="Precedence", header=0)
            durations = pd.read_excel(file_path_full, sheet_name="Durations", header=None).to_numpy().flatten()
            base_benefits = pd.read_excel(file_path_full, sheet_name="Benefits", header=None).to_numpy().flatten()
            resource_demand = pd.read_excel(file_path_full, sheet_name="Demands", header=None).to_numpy()
            resource_capacity = pd.read_excel(file_path_full, sheet_name="Capacity", header=None).to_numpy().flatten()
            
            #截取前project_num个数据
            enhancement_matrix = enhancement_matrix[:project_num, :project_num]
            durations = durations[:project_num]
            base_benefits = base_benefits[:project_num]
            resource_demand = resource_demand[:project_num]
            precedence_data = precedence_data[precedence_data['Task'] < project_num]
            
            project_data={
                'duration': durations,
                'resources': resource_demand,
                'base_benefit': base_benefits
            }
            
            # 转换 precedence_data 为字典格式
            precedence_constraints = {}
            for _, row in precedence_data.iterrows():
                task = int(row['Task'])  # 当前任务编号
                if pd.isna(row['Predecessors']):
                    precedence_constraints[task] = []
                else:
                    precedence_constraints[task] = list(map(int, str(row['Predecessors']).split(',')))

            # 创建环境实例
            env = RCPSPISEnv(
                num_projects=project_num,
                resource_capacity=resource_capacity,
                max_time=max_time,
                project_data=project_data,
                enhancement_matrix=enhancement_matrix,
                precedence_constraints=precedence_constraints
            )
            envs.append(env)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return envs


if __name__ == "__main__":
    
    ################################## set device ##################################
    print("============================================================================================")
    # set device to mps, cuda, or cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS GPU
        print("Device set to : MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")  # CPU
        print("Device set to : CPU")
    print("============================================================================================")
        
    # device = torch.device("cpu")
    # train_set_num = [60,90,120,150,180]
    file_path = '/Users/liuxiaohang/Desktop/清华/研究/DRL/PSPLIB/PSPLIB_Processed/j30/train'
    num_projects = 10
    max_time = 30
    env_num = 10
    env_list = load_data_and_create_envs(file_path, env_num, max_time, num_projects)
    
    for i in range(env_num):
        # 创建环境实例
        env = env_list[i]
        project_data = env.project_data
        precedence_constraints = env.precedence_constraints
         #将前置约束变成网络图
        edge_list = []
        for proj, preds in precedence_constraints.items():
            for p in preds:
                edge_list.append([p, proj])
        if len(edge_list) == 0:
            edge_list = [[i, i] for i in range(num_projects)]
        edge_index = torch.LongTensor(edge_list).t().contiguous().to(device)  # [2, num_edges]
    
        # Initialize agent
        num_resource_types = project_data['resources'].shape[1]
        node_output_dim = 16
        global_dim = num_resource_types + 1 + node_output_dim
        node_input_dim = num_resource_types + 6 #节点特征的维度数 资源、时间、收益、激励以及状态
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
        ).to(device)
        
        critic = Critic(
            node_input_dim=node_input_dim,
            node_hidden_dim=64,
            node_output_dim=node_output_dim,
            global_dim=global_dim,
            attention_hidden_dim=128,
            final_state_dim=16,
            edge_index=edge_index
        ).to(device)
        
        
        ppo = PPOAgent(
            actornet = actor,
            criticnet = critic,
            actor_lr = actor_lr, 
            critic_lr = critic_lr,
            device=device,
            clip_range=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            gamma=1,
            lam=0.95
        )
    
        max_episodes = 2000
        steps_per_update = 10
        actor_loss_list = []
        critic_loss_list = []
        reward_list = []
        start = time.time()
        
        for i in range(10):
            # 使用tqdm显示训练进度
            with tqdm(total=int(max_episodes/10), desc='Iteration %d' % i) as pbar:
                for ep in range(int(max_episodes/10)):
                    ppo.buffer.reset()
                    episode_reward = 0
            
                    # 环境交互阶段
                    # 采样阶段：环境交互与数据存储 采集数据量：steps_per_update次 game交互
                    for step in range(steps_per_update):
                        done = False
                        state = env.reset()
                        while not done:
                            node_features = ppo.compute_node_features(state)
                            global_info = ppo.compute_global_info(state)
                            action_mask = ppo.compute_action_mask(state)
        
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
        end = time.time()
        train_time = end-start
        print('Training time of this instance is:', train_time)
        # 绘制结果
        custom_params = {"axes.spines.right": False, "axes.spines.top": False,
                          "font.family": "Times New Roman", "font.scale": 1.5}
        sns.set_style(style="ticks", rc=custom_params) #设置绘图风格
        # palette = sns.color_palette("Set2")
        
        
        act_loss_mov = moving_average(actor_loss_list, 9)
        plt.figure(figsize=(10, 6), dpi=600, facecolor="w")
        plt.plot(np.arange(1, max_episodes+1), act_loss_mov, 'r-', linewidth=2.5)
        plt.axhline(y= 0,alpha = 0.2,linestyle = '--', color = 'black')
        plt.xlabel("训练轮次", fontsize = 18, fontproperties = 'SimSong')
        plt.ylabel("损失", fontsize = 18, fontproperties = 'SimSong')
        plt.tick_params(axis='both', labelsize=18)
        plt.grid(False)
        plt.savefig('/Users/liuxiaohang/Desktop/清华/研究/DRL/Figure/策略损失_{num_projects}_{i}.png',dpi=600)
        plt.show()
        
        critic_loss_mov = moving_average(critic_loss_list, 9)
        plt.figure(figsize=(10, 6), dpi=600, facecolor="w")
        plt.plot(np.arange(1, max_episodes+1), critic_loss_mov, 'r-', linewidth=2.5)
        plt.xlabel("训练轮次", fontsize = 18, fontproperties = 'SimSong')
        plt.ylabel("损失", fontsize = 18, fontproperties = 'SimSong')
        plt.tick_params(axis='both', labelsize=18)
        plt.grid(False)
        plt.savefig(f'/Users/liuxiaohang/Desktop/清华/研究/DRL/Figure/价值损失_{num_projects}_{i}.png',dpi=600)
        plt.show()
        
        reward_mov = moving_average(reward_list, 9)
        plt.figure(figsize=(10, 6), dpi=600, facecolor="w")
        plt.plot(np.arange(1, max_episodes+1), reward_mov, 'b-', linewidth=2.5)
        plt.xlabel("训练轮次", fontsize = 18, fontproperties = 'SimSong')
        plt.ylabel("收益", fontsize = 18, fontproperties = 'SimSong')
        plt.tick_params(axis='both', labelsize=18)
        plt.grid(False)
        plt.savefig('/Users/liuxiaohang/Desktop/清华/研究/DRL/Figure/收益曲线_{num_projects}_{i}.png',dpi=600)
        plt.show()
        
        saved_data = [act_loss_mov,critic_loss_mov,reward_mov]
        saved_data = pd.DataFrame(saved_data,columns=['Actor loss','Critic loss', 'Reward','Train Time'], index = False)
        saved_data.to_excel(f'/Users/liuxiaohang/Desktop/清华/研究/DRL/saved_data/Train_data_{num_projects}_{i}.xlsx')
        
        file_path1 = 'f/Users/liuxiaohang/Desktop/清华/研究/DRL/saved_model/PPO_Agent_model_output_{num_projects}_{i}.pth'
        ppo.save(file_path1)
        
