#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:40:45 2024

@author: liuxiaohang
"""
import numpy as np
import torch
import copy
from RCPSPISEnv import RCPSPISEnv, RCPSPISEnv_det
from PPO_training import PPOAgent
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


def evaluate_policy(env, ppo_model, num_projects, max_time, resource_capacity, project_data, precedence_constraints):
    """
    使用训练好的 PPO 模型完成项目调度任务，输出策略和收益。
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
        action, log_prob = ppo_model.select_action(node_features, global_info, action_mask)
        prob = np.exp(log_prob.detach().numpy())
        # 执行动作，更新环境
        next_state, reward, done, info = env.step(action)
        strategy.append(action.item())
        print(f"Action: Project {action.item()}, Reward: {reward}, Prob: {prob}")
        state = next_state
    total_reward = env.accumulated_benefit

    return strategy, total_reward

def dp_solve(env, state, memo, depth=0):
    """
    确定性DP算法，用于在RCPSPISEnv环境中求解最优策略。
    """
    # 创建唯一状态标识
    state_key = (
        tuple(env.completed), 
        tuple(env.ongoing), 
        tuple(env.available_resources), 
        env.current_time, 
        env.accumulated_benefit, 
        tuple(env.realized_benefits)
    )
    
    # 如果状态已经计算过，直接返回缓存的值
    if state_key in memo:
        # print(" " * depth, f"Memo Hit: State={state_key}, Reward={memo[state_key][0]}")
        return memo[state_key]
    
    backup_states = {}
    # 检查是否已完成所有项目
    if env.current_time == env.max_time:
        # print(" " * depth, f"Time Reached: {env.current_time}, Reward=0")
        return 0, []

    max_reward = 0
    best_seq = []
    
    # 遍历所有可能的动作，包括跳过
    for action in range(env.num_projects + 1):
        try:
            backup_tuple = (depth, action)
            backup_states[backup_tuple] = {
                "state": copy.deepcopy(env.get_state()),
                "accumulated_benefit": env.accumulated_benefit,
                "project_end_times": env.project_end_times.copy(),
                "current_time": env.current_time
            }
            next_state, reward, done, _ = env.step(action)
            next_state_copy = copy.deepcopy(next_state)
            # print(" " * depth, f"Action={action}, Reward={reward}, Done={done}, Time={env.current_time}")
        except ValueError:
            continue
        
        if done:
            total_reward = reward
            future_sequence = []
        else:
            future_reward, future_sequence = dp_solve(env, next_state_copy, memo, depth+2)
            total_reward = reward + future_reward
        
        # print(" " * depth, f"Action={action}, Total Reward={total_reward}, Future Sequence={future_sequence}")
        # 更新最优解
        if total_reward > max_reward:
            max_reward = total_reward
            best_seq = [action] + future_sequence
            # print(" " * depth,'Update!', max_reward,best_seq)

        # 恢复环境状态
        env.reset()
        backup_tuple = (depth, action)
        restore_state = backup_states[backup_tuple]
        env.accumulated_benefit = restore_state["accumulated_benefit"]
        env.project_end_times = restore_state["project_end_times"]
        env.completed = restore_state["state"]['completed'].copy()
        env.ongoing = restore_state["state"]['ongoing'].copy()
        env.pending = restore_state["state"]['pending'].copy()
        env.available_resources = restore_state["state"]['available_resources'].copy()
        env.current_time = restore_state["current_time"]
        env.realized_benefits = restore_state["state"]['realized_benefits'].copy()

    # 存储最优解到记忆表
    memo[state_key] = (copy.deepcopy(max_reward), copy.deepcopy(best_seq))
    return max_reward, best_seq


if __name__ == "__main__":

    np.random.seed(32)
    # torch.manual_seed(40)
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
        critic_lr = critic_lr
    )

    
    # 创建环境实例
    for _ in range(5):
        env = RCPSPISEnv(num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints)
        file_path = 'PPO_Agent_model_output.pth'
        ppo.load(file_path)
    
        strategy, total_reward = evaluate_policy(
            env, ppo, num_projects, max_time, resource_capacity, project_data, precedence_constraints
        ) 
        print("Strategy:", strategy)
        print("Total Reward:", total_reward)
        
        realized_data = env.realized_benefits
        det_env = RCPSPISEnv_det(num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints, realized_data)
        initial_state = det_env.reset()
        optimal_reward, optimal_seq = dp_solve(det_env, initial_state, {})
        print("Optimal Reward:", optimal_reward)
        print("Optimal Sequence:", optimal_seq)