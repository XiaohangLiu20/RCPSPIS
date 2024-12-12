#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:21:11 2024

@author: liuxiaohang
"""
import gym
from gym import spaces
import numpy as np
import copy

class RCPSPISEnv(gym.Env): #自定义gym环境类 后续需要注册和导入
    def __init__(self, num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints):
        """
        初始化环境
        :param num_projects: 项目数量
        :param resource_capacity: 可用资源容量
        :param max_time: 最大调度时间
        :param project_data: 包括每个项目的持续时间、资源需求、基础收益分布等
        :param enhancement_matrix: 项目间收益增强矩阵
        :param precedence_matrix: 项目前置约束要求 dict输入
        """
        super(RCPSPISEnv, self).__init__()
        
        # 项目信息
        self.num_projects = num_projects
        self.project_data = project_data  # {'duration': [], 'resources': [], 'base_benefit': []}
        self.enhancement_matrix = enhancement_matrix
        self.precedence_constraints = precedence_constraints
    
        # 资源容量和最大时间
        self.resource_capacity = np.array(resource_capacity)
        self.max_time = max_time
    
        # 状态空间: 包括已完成、正在进行、待调度的项目、资源分配、当前时间和已完成项目的实现收益
        self.observation_space = spaces.Dict({
            "completed": spaces.MultiBinary(num_projects),
            "ongoing": spaces.MultiBinary(num_projects),
            "pending": spaces.MultiBinary(num_projects),
            "available_resources": spaces.Box(0, np.inf, shape=(len(resource_capacity),), dtype=np.float32),
            "current_time": spaces.Box(0, max_time, shape=(1,), dtype=np.float32),
            "realized_benefits":spaces.Box(0, np.inf, shape=(num_projects,), dtype=np.float32),
        })
    
        # 动作空间: 当前可以启动的项目或跳过
        self.action_space = spaces.Discrete(num_projects + 1)  # +1 表示跳过动作
    
        # 初始化环境变量
        self.reset()
        
    def reset(self):
        """重置环境"""
        self.completed = np.zeros(self.num_projects, dtype=int)  # 已完成项目
        self.ongoing = np.zeros(self.num_projects, dtype=int)  # 正在进行的项目
        self.pending = np.array([1 if len(self.precedence_constraints.get(i, [])) == 0 else 0 for i in range(self.num_projects)], dtype=int)
        self.available_resources = self.resource_capacity.copy()  # 可用资源
        self.current_time = 0  # 当前时间
        self.project_end_times = np.zeros(self.num_projects, dtype=int)  # 正在进行项目的结束时间
        self.accumulated_benefit = 0  # 累积收益
        self.realized_benefits = self.project_data['base_benefit']
        return self.get_state()
    
    def get_state(self):
        """返回当前状态"""
        return {
            "completed": self.completed,
            "ongoing": self.ongoing,
            "pending": self.pending,
            "available_resources": self.available_resources,
            "current_time": np.array([self.current_time], dtype=np.float32),
            "realized_benefits":self.realized_benefits
        }
    
    def _update_pending(self):
        """严格检查前置约束、资源和时间约束，更新 pending 集合"""
        for i in range(self.num_projects):
            if self.completed[i] == 1 or self.ongoing[i] == 1:
                # 已完成或正在进行的项目不在 pending 集合中
                self.pending[i] = 0
            else:
                # 检查前置约束是否满足
                precedence_satisfied = all(self.completed[pre] == 1 for pre in self.precedence_constraints.get(i, []))
                if precedence_satisfied:
                    # 检查资源和时间约束
                    resource_feasible = np.all(self.available_resources >= self.project_data['resources'][i])
                    time_feasible = self.current_time + self.project_data['duration'][i] <= self.max_time
                    if resource_feasible and time_feasible:
                        self.pending[i] = 1
                    else:
                        self.pending[i] = 0  # 资源或时间不满足时，移出 pending
                else:
                    self.pending[i] = 0  # 前置约束不满足时，移出 pending
    
    def reward(self):
        if np.sum(self.completed) == 0:
            return 1  # 无项目完成时的默认奖励
    
        # 计算增强系数 (incentive) 返回一个长度为project的数组 为当前完成项目对其他所有项目的增强效应总和（包括未生效项目）
        incentive = np.prod(self.enhancement_matrix**self.completed[:, None], axis=0)
        # 计算增强后的收益 (enhance_benefit)
        enhance_benefit = incentive * self.realized_benefits
        # 计算总收益并确保未完成的项目不会影响乘积
        qe = (1 + enhance_benefit) * self.completed
        qe[qe == 0] = 1  # 未完成的项目其值设为 1，不影响乘积
        # 最终单位收益
        r = np.prod(qe)
        return r
    
    def step(self,action):
        """
        执行动作
        :param action: 动作索引，0-num_projects 表示调度项目，num_projects 表示跳过
        :return: (state, reward, done, info)
        """
        reward = 0
        done = False
        
        if action == self.num_projects:  # 跳过动作
            if np.any(self.ongoing):
                next_completion_time = np.min(self.project_end_times[self.ongoing > 0])
                assert next_completion_time > self.current_time, "Current time fails to update correctly!"
                time_duration = next_completion_time - self.current_time
                self.current_time = next_completion_time
                completed_projects = np.where(self.project_end_times == self.current_time)[0]
                #更新并累计收益
                unit_reward = self.reward()
                reward = unit_reward*time_duration
                self.accumulated_benefit += reward 
                #更新时间到最早有项目完成的时间节点
                for project in completed_projects:
                    self.ongoing[project] = 0
                    self.completed[project] = 1
                    self.available_resources += self.project_data['resources'][project]
                    # 更新收益
                    base_benefit = self.project_data['base_benefit'][project]
                    # 假设收益服从正态分布，均值为 base_benefit，标准差为均值的 10%
                    std_dev = base_benefit * 0.1
                    self.realized_benefits[project] = base_benefit
                
                # 更新 pending 集合
                self._update_pending()
        
            elif np.all(self.pending == 0): #如果同时也没有待执行的则说明结束
                done = True 
                next_completion_time = self.max_time
                time_duration = next_completion_time - self.current_time
                self.current_time = next_completion_time
                unit_reward = self.reward() 
                reward = unit_reward*time_duration
                self.accumulated_benefit += reward
            
            else:
                raise ValueError("Invalid action: Skip, there are pending project!")
        
        else:  # 调度项目 这里注意只有pending中的项目才是合法的action
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
    
    # 如果状态已经计算过，直接返回缓存的值
    # if state_tuple in memo:
    #     print(" " * depth, f"Memo Hit: State={state_tuple}, Reward={memo[state_tuple][0]}")
    #     return memo[state_tuple]

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
        except ValueError as e:
            # print(" " * depth, f"Invalid Action={action}: {str(e)}")
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

# 使用示例
np.random.seed(42)
# torch.manual_seed(42)

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

env = RCPSPISEnv(num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints)
initial_state = env.reset()
optimal_reward, optimal_seq = dp_solve(env, initial_state, {})
print("Optimal Reward:", optimal_reward)
print("Optimal Sequence:", optimal_seq)

# env = RCPSPISEnv(num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints)
# initial_state = env.reset()
# action1 = [2, 5, 0, 1, 5, 5, 3, 5, 4, 5, 5]

# for item in action1:
#     env.step(item)
# print(f'action1_Rewards = {env.accumulated_benefit}')

# env.reset()

# action2 = [0, 1, 5, 5, 2, 5, 3, 5, 4, 5, 5]
# for item in action2:
#     env.step(item)
# print(f'action2_Rewards = {env.accumulated_benefit}')

