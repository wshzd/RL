# !/usr/bin/env python
# -*- coding:utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib
# inline
# 0 -> 左
# 1 -> 下
# 2 -> 右
# 3 -> 上


#       3
#       ^
#       |
# 0< —— o —— > 2
#       |
#       v
#       1
# 动作空间=4，分别是上下左右移动，同时湖面上可能刮风，使agent随机移动

# S Starting point,safe
# F Frozen surface,safe
# H Hole,end of game,bad
# G Goal,end of game,good

# 状态空间=16,4x4的矩阵
# SFFF
# FHFH
# FFFH
# HFFG

# 在左侧边缘，无法向左移动，要么原地不动，要么向下移动
# 在上侧边缘，无法向上移动，要么原地不动，要么向右移动

env = gym.make('FrozenLake-v0')
env.reset()


def value_iteration(env, gamma=1.0, no_of_iterations=2000):
    '''
    值迭代函数，目的是准确评估每一个状态的好坏
    '''
    # 状态价值函数，向量维度等于游戏状态的个数
    value_table = np.zeros(env.observation_space.n)

    # 随着迭代，记录算法的收敛性
    error = []  # 价值函数的差
    index = []  # 价值函数的第一个元素

    threshold = 1e-20

    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)

        for state in range(env.observation_space.n):  # state = 0,1,2,3,...,15
            # 在某个状态下，采取4种动作后，分别的reward，Q_value长度=4
            Q_value = []

            for action in range(env.action_space.n):  # action = 0,1,2,3
                # 采取不同action转移到不同的状态，也对应不同的reward
                next_states_rewards = []

                for next_sr in env.P[state][action]:  # 在当前state和action的情况下，把可能转移的状态遍历一遍

                    # next_sr = (0.3333333333333333, 8,       0.0           , False)
                    # next_sr = (状态转移概率,        下一个状态,得到reward的概率,游戏是否结束)
                    trans_prob, next_state, reward_prob, _ = next_sr

                    # 下一状态t的动作状态价值 = 转移到t状态的概率 × （ env反馈的reward + γ × t状态的当前价值 ）
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                # 将某一动作执行后，所有可能的t+1状态的价值加起来，就是在t状态采取a动作的价值
                Q_value.append(np.sum(next_states_rewards))

            # Q_value长度为4（上下左右）,选Q_value的最大值作为该状态的价值
            value_table[state] = max(Q_value)
        index.append(value_table[0])
        # 如果状态价值函数已经收敛
        error.append(np.sum(np.fabs(updated_value_table - value_table)))
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    # 画出价值函数的收敛曲线
    plt.figure(1)
    plt.plot(error)
    # 画出价值函数第一个元素的值的收敛情况
    plt.figure(2)
    plt.plot(index)
    return value_table


def extract_policy(value_table, gamma=1.0):
    '''
    在一个收敛的、能够对状态进行准确评估的状态值函数的基础上，推导出策略函数，即在每一个状态下应该采取什么动作最优的
    '''

    # policy代表处于状态t时应该采取的最佳动作是上/下/左/右,policy长度16
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):
        # 将价值迭代的过程再走一遍，但是不再更新value function，而是选出每个状态下对应最大价值的动作
        Q_table = np.zeros(env.action_space.n)  # len=4
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                #
                trans_prob, next_state, reward_prob, _ = next_sr
                #
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        # 选最优价值的动作
        policy[state] = np.argmax(Q_table)

    return policy


optimal_value_function = value_iteration(env=env)
print('\nthe best value function:\n', optimal_value_function, '\n')

optimal_policy = extract_policy(optimal_value_function, gamma=1.0)
print('the best policy:\n', optimal_policy)




