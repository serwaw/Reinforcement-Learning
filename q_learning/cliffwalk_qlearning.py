"""
Example in CliffWalking, using the FiniteQLearningModel.
"""
import sys
sys.path.append("..")

import utils
import gym
from q_learning.q_learn_sol import FiniteQLearningModel as QLearning
from cliff_walk.cliff_walking import CliffWalkingEnv
#from ai_safety_gridworlds.sokoban.environments.side_effects_sokoban import SideEffectsSokobanEnvironment
from safe_grid_gym.gym_interface.envs.gridworlds_env import GridworldEnv

env = GridworldEnv('side_effects_sokoban')
#env = CliffWalkingEnv()
# WARNING: If you try to set eps to a very low value,
# And you attempt to get the m.score() of m.pi, there may not
# be guarranteed convergence.
eps = 300
S = 4*12
A = 4
START_EPS = 0.7
m = QLearning(S, A, epsilon=START_EPS)

SAVE_FIG = False
history = []

for i in range(1, eps+1):
    ep = []
    prev_observation = env.reset()
    prev_action = m.choose_action(m.b, prev_observation)

    total_reward = 0
    while True:
        # Run simulation
        next_observation, reward, done, _ = env.step(prev_action)
        next_action = m.choose_action(m.b, next_observation)

        m.update_Q((prev_observation, prev_action, reward, next_observation, next_action))

        prev_observation = next_observation
        prev_action = next_action

        total_reward += reward
        if done:
            break

    history.append(total_reward)

if SAVE_FIG:
    print("Final expected returns : {}".format(m.score(env, m.pi, n_samples=10)))
    utils.render_cliffwalking(m.Q, "side_qlearning.png")