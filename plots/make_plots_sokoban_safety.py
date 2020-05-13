import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from q_learning import q_learning_sokoban
#from sarsa import sarsa_sokoban
from ai_safety_gridworlds.sokoban.environments.side_effects_sokoban import SideEffectsSokobanEnvironment
from safe_grid_gym.gym_interface.envs.gridworlds_env import GridworldEnv



if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict

matplotlib.style.use('ggplot')

env_sokoban = GridworldEnv('side_effects_sokoban')

#
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


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
    episode_lengths_sarsa=np.zeros(num_episodes)
    episode_rewards_sarsa=np.zeros(num_episodes)
    #episode_hidden_reward_sarsa = np.zeros(num_episodes)

    # The policy we're following
    #policy = sarsa_sokoban.make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        # Finds where the agent is(2)
        state_find_agent = np.where(state == 2)

        # This sets concatenate the row and the column and returns a list
        state_agent_con = np.concatenate((state_find_agent)).tolist()

        # Converts it to a int number
        state_to_int = int(''.join(map(str, state_agent_con)))

        action_probs = policy(state_to_int)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            # Finds where the agent is(2) for the next state
            next_obs = np.where(next_state == 2)
            # This sets concatenate the row and the column and returns a list
            next_obs_con = np.concatenate((next_obs)).tolist()

            # Converts it to a int number
            next_obs_int = int(''.join(map(str, next_obs_con)))
            # Pick the next action
            next_action_probs = policy(next_obs_int)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            #Finds the box
            the_box = np.where(next_state == 4)
            the_box_con = np.concatenate((the_box)).tolist()
            the_box_int = int(''.join(map(str, the_box_con)))
            #If the box is pushed to a corner
            if the_box_int == 32 and (next_obs_int == 34 or next_obs_int == 43):
                wall_penalty = -10
            #If the box pushed to a adjecent wall
            elif the_box_int == 23 and (next_obs_int == 34 or next_obs_int == 43):
                wall_penalty = -5
            elif the_box_int == 24 and (next_obs_int == 34 or next_obs_int == 43):
                wall_penalty = -5
            else:
                wall_penalty = 0

            # Update statistics
            episode_rewards_sarsa[i_episode] += (reward + wall_penalty)
            episode_lengths_sarsa[i_episode] = t

            # TD Update
            td_target = reward + wall_penalty + discount_factor * Q[next_obs_int][next_action]
            td_delta = td_target - Q[state_to_int][action]
            Q[state_to_int][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state_to_int = next_obs_int

    return Q, episode_rewards_sarsa, episode_lengths_sarsa


Q, episode_rewards_sarsa, episode_lengths_sarsa = sarsa(env_sokoban, 500)

def plot_reward():
    # Plot the episode reward over time
    fig_reward = plt.figure(figsize=(10, 5))
    #rewards_smoothed_q = pd.Series(episode_rewards_q).rolling(10, min_periods=10).mean()
    rewards_smoothed_sarsa_r = pd.Series(episode_rewards_sarsa).rolling(10, min_periods=10).mean()
    #plt.plot(rewards_smoothed_q)
    plt.plot(rewards_smoothed_sarsa_r)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(10))
    plt.legend(['Sarsa'])
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
    #plt.plot(episode_lengths_q)
    plt.plot(episode_lengths_sarsa)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend(['Sarsa'])
    if noshow:
        plt.close(fig_length)
    else:
        plt.show(fig_length)
    return fig_length


# # Find the box
            # hidden_reward = 0
            # box = 0
            # obs_box = np.where(box == 4)
            # if obs_box == 32:
            #     hidden_reward = -10
            # elif obs_box == 23 or obs_box == 24:
            #     hidden_reward = -5
            # else:
            #     hidden_reward = 0

plot_reward()
plot_length()