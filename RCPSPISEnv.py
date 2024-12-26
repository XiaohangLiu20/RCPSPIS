#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:41:28 2024

@author: liuxiaohang
"""

import gym
from gym import spaces
import numpy as np
import copy

class RCPSPISEnv(gym.Env):
    def __init__(self, num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints):
        super(RCPSPISEnv, self).__init__()
        
        self.num_projects = num_projects
        self.project_data = copy.deepcopy(project_data)  # {'duration': [], 'resources': [], 'base_benefit': []}
        self.enhancement_matrix = copy.deepcopy(enhancement_matrix)
        self.precedence_constraints = copy.deepcopy(precedence_constraints)
    
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
        self.remaining_time = self.project_data['duration'].copy()
        self.time_duration = 0
        return self.get_state()
    
    def get_state(self):
        state = {
            "completed": self.completed,
            "ongoing": self.ongoing,
            "pending": self.pending,
            "available_resources": self.available_resources,
            "current_time": np.array([self.current_time], dtype=np.float32),
            "realized_benefits": self.realized_benefits,
            "remaining_time": self.remaining_time
        }
        
        return copy.deepcopy(state)
    
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
    
    def log_reward(self):
        if np.sum(self.completed) == 0:
            return 0
        incentive = np.prod(self.enhancement_matrix**self.completed[:, None], axis=0)
        enhance_benefit = incentive * self.realized_benefits
        log_benefit = np.log(1 + enhance_benefit)
        qe = log_benefit * self.completed
        r = np.sum(qe)
        return r
    
    def step(self, action):
        reward = 0
        done = False
        
        if action == self.num_projects:  # Skip
            if np.any(self.ongoing):
                next_completion_time = np.min(self.project_end_times[self.ongoing > 0])
                assert next_completion_time > self.current_time, "Time update error!"
                self.time_duration = next_completion_time - self.current_time
                ongoing_indices = np.where(self.ongoing == 1)[0]
                self.remaining_time[ongoing_indices] -= self.time_duration
                self.current_time = next_completion_time
                completed_projects = np.where(self.project_end_times == self.current_time)[0]
                unit_reward = self.reward() #正常reward
                # unit_reward = self.log_reward()
                reward = unit_reward * self.time_duration
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
                self.time_duration = next_completion_time - self.current_time
                self.current_time = next_completion_time
                unit_reward = self.reward() #正常reward
                # unit_reward = self.log_reward() #log reward
                reward = unit_reward * self.time_duration
                self.accumulated_benefit += reward
            else:
                raise ValueError("Invalid action: Skip, there are pending projects!")
        
        else:  # Launch a project
            if self.pending[action]:
                self.time_duration = 0
                self.pending[action] = 0
                self.ongoing[action] = 1
                self.available_resources -= self.project_data['resources'][action]
                self.project_end_times[action] = self.current_time + self.project_data['duration'][action]
                self._update_pending()
            else:
                raise ValueError(f"Invalid action: {action}. Project {action} is not pending!")
 
        state = self.get_state()
        return state, reward, done, {}
        
class RCPSPISEnv_det(gym.Env):
    def __init__(self, num_projects, resource_capacity, max_time, project_data, enhancement_matrix, precedence_constraints, realized_benefits):
        super(RCPSPISEnv_det, self).__init__()
        
        self.num_projects = num_projects
        self.project_data = copy.deepcopy(project_data)  # {'duration': [], 'resources': [], 'base_benefit': []}
        self.enhancement_matrix = copy.deepcopy(enhancement_matrix)
        self.precedence_constraints = copy.deepcopy(precedence_constraints)
    
        self.resource_capacity = np.array(resource_capacity)
        self.max_time = max_time
        self.realized_benefits = realized_benefits

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
        self.realized_benefits = self.realized_benefits
        self.remaining_time = self.project_data['duration'].copy()
        self.time_duration = 0
        return self.get_state()
    
    def get_state(self):
        state = {
            "completed": self.completed,
            "ongoing": self.ongoing,
            "pending": self.pending,
            "available_resources": self.available_resources,
            "current_time": np.array([self.current_time], dtype=np.float32),
            "realized_benefits": self.realized_benefits,
            "remaining_time": self.remaining_time,
            "duration": self.project_data['duration'].copy(),
            "resources":self.project_data['resources'].copy(),
            "enhancement":self.enhancement_matrix
        }
        
        return copy.deepcopy(state)
    
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
    
    def log_reward(self):
        if np.sum(self.completed) == 0:
            return 0
        incentive = np.prod(self.enhancement_matrix**self.completed[:, None], axis=0)
        enhance_benefit = incentive * self.realized_benefits
        log_benefit = np.log(1 + enhance_benefit)
        qe = log_benefit * self.completed
        r = np.sum(qe)
        return r
    
    def step(self, action):
        reward = 0
        done = False
        
        if action == self.num_projects:  # Skip
            if np.any(self.ongoing):
                next_completion_time = np.min(self.project_end_times[self.ongoing > 0])
                assert next_completion_time > self.current_time, "Time update error!"
                self.time_duration = next_completion_time - self.current_time
                ongoing_indices = np.where(self.ongoing == 1)[0]
                self.remaining_time[ongoing_indices] -= self.time_duration
                self.current_time = next_completion_time
                completed_projects = np.where(self.project_end_times == self.current_time)[0]
                unit_reward = self.reward() #正常reward
                # unit_reward = self.log_reward()
                reward = unit_reward * self.time_duration
                self.accumulated_benefit += reward
                for project in completed_projects:
                    self.ongoing[project] = 0
                    self.completed[project] = 1
                    self.available_resources += self.project_data['resources'][project]
                    # base_benefit = self.project_data['base_benefit'][project]
                    # self.realized_benefits[project] = base_benefit
                self._update_pending()
            
            elif np.all(self.pending == 0):
                done = True
                next_completion_time = self.max_time
                self.time_duration = next_completion_time - self.current_time
                self.current_time = next_completion_time
                unit_reward = self.reward() #正常reward
                # unit_reward = self.log_reward() #log reward
                reward = unit_reward * self.time_duration
                self.accumulated_benefit += reward
            else:
                raise ValueError("Invalid action: Skip, there are pending projects!")
        
        else:  # Launch a project
            if self.pending[action]:
                self.time_duration = 0
                self.pending[action] = 0
                self.ongoing[action] = 1
                self.available_resources -= self.project_data['resources'][action]
                self.project_end_times[action] = self.current_time + self.project_data['duration'][action]
                self._update_pending()
            else:
                raise ValueError(f"Invalid action: {action}. Project {action} is not pending!")
 
        state = self.get_state()
        return state, reward, done, {}    
