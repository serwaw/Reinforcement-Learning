import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from q_learning import q_learning_cliffwalk
from sarsa import sarsa_cliffwalk
from cliff_walk.cliff_walking import CliffWalkingEnv
from ai_safety_gridworlds.sokoban.environments.side_effects_sokoban import SideEffectsSokobanEnvironment
#from q_learning import plotting
from safe_grid_gym.gym_interface.envs.gridworlds_env import GridworldEnv
from q_learning import q_learning_sokoban
from sarsa import sarsa_sokoban



if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from cliff_walk.cliff_walking import CliffWalkingEnv

matplotlib.style.use('ggplot')

env_cliffwalk = CliffWalkingEnv()
env_sokoban = GridworldEnv('side_effects_sokoban')



def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths_sarsa_cliff=np.zeros(num_episodes)
    episode_rewards_sarsa_cliff=np.zeros(num_episodes)

    # The policy we're following
    policy = sarsa_cliffwalk.make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            episode_rewards_sarsa_cliff[i_episode] += reward
            episode_lengths_sarsa_cliff[i_episode] = t

            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state = next_state

    return Q, episode_lengths_sarsa_cliff, episode_rewards_sarsa_cliff


Q, episode_lengths_sarsa_cliff, episode_rewards_sarsa_cliff = sarsa(env_cliffwalk, 500)


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    episode_lengths_sarsa_sokoban=np.zeros(num_episodes)
    episode_rewards_sarsa_sokoban=np.zeros(num_episodes)



    # The policy we're following
    policy = sarsa_sokoban.make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        # Finds where the agent is(2), use this for sokoban
        state_find_agent = np.where(state == 2)
        # This sets concatenate the row and the column and returns a list
        state_agent_con = np.concatenate((state_find_agent)).tolist()

        # Converts it to a int number, use this one for sokoban
        state_to_int = int(''.join(map(str, state_agent_con)))

        action_probs = policy(state_to_int)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Finds where the agent is(2) for the next state, use this for sokoban
            next_obs = np.where(next_state == 2)
            # This sets concatenate the row and the column and returns a list
            next_obs_con = np.concatenate((next_obs)).tolist()

            # Converts it to a int number, use this one for sokoban
            next_obs_int = int(''.join(map(str, next_obs_con)))

            # Pick the next action
            next_action_probs = policy(next_obs_int)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            episode_rewards_sarsa_sokoban[i_episode] += reward
            episode_lengths_sarsa_sokoban[i_episode] = t

            # TD Update
            td_target = reward + discount_factor * Q[next_obs_int][next_action]
            td_delta = td_target - Q[state_to_int][action]
            Q[state_to_int][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state_to_int = next_obs_int

    return Q, episode_rewards_sarsa_sokoban, episode_lengths_sarsa_sokoban

Q, episode_rewards_sarsa_sokoban, episode_lengths_sarsa_sokoban = sarsa(env_sokoban, 500)


def plot_reward():
    # Plot the episode reward over time
    fig_reward = plt.figure(figsize=(10, 5))
    rewards_smoothed_q = pd.Series(episode_rewards_sarsa_cliff).rolling(10, min_periods=10).mean()
    rewards_smoothed_sarsa = pd.Series(episode_rewards_sarsa_sokoban).rolling(10, min_periods=10).mean()
    plt.plot(rewards_smoothed_q)
    plt.plot(rewards_smoothed_sarsa)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(10))
    plt.legend(['q-learning', 'Sarsa'])
    #plt.show()
    noshow = False
    if noshow:
        plt.close(fig_reward)
    else:
        plt.show(fig_reward)
    return fig_reward

def plot_length(smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig_length = plt.figure(figsize=(10,5))
    plt.plot(episode_lengths_sarsa_cliff)
    plt.plot(episode_lengths_sarsa_sokoban)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend(['q-learning', 'Sarsa'])
    if noshow:
        plt.close(fig_length)
    else:
        plt.show(fig_length)
    return fig_length

plot_reward()
plot_length()
