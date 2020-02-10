
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
#from ai_safety_gridworlds.sokoban.environment.side_effects_sokoban import SideEffectsSokobanEnvironment
from safe_grid_gym.gym_interface.envs.gridworlds_env import GridworldEnv
from ai_safety_gridworlds.distributional_shift_gym import SideEffectsSokobanEnv
from q_learning import plotting
from q_learning.q_learning_sokoban import q_learning

matplotlib.style.use('ggplot')

env = GridworldEnv('side_effects_sokoban')


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
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        print("sarsa", state)
        # Finds where the agent is(2), use this for sokoban
        state_find_agent = np.where(state == 2)
        print(state_find_agent)
        # This sets concatenate the row and the column and returns a list
        state_agent_con = np.concatenate((state_find_agent)).tolist()
        print(state_agent_con)

        # Converts it to a int number, use this one for sokoban
        state_to_int = int(''.join(map(str, state_agent_con)))
        print(state_to_int)

        action_probs = policy(state_to_int)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            print("sarsa", next_state)

            # Finds where the agent is(2) for the next state, use this for sokoban
            next_obs = np.where(next_state == 2)
            print(next_obs)
            # This sets concatenate the row and the column and returns a list
            next_obs_con = np.concatenate((next_obs)).tolist()
            print(next_obs_con)

            # Converts it to a int number, use this one for sokoban
            next_obs_int = int(''.join(map(str, next_obs_con)))
            print(next_obs_int)

            # Pick the next action
            next_action_probs = policy(next_obs_int)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            td_target = reward + discount_factor * Q[next_obs_int][next_action]
            td_delta = td_target - Q[state_to_int][action]
            Q[state_to_int][action] += alpha * td_delta

            if done:
                break

            action = next_action
            state_to_int = next_obs_int

    return Q, stats

Q, stats = sarsa(env, 500)
Q, stats = q_learning(env, 500)

plotting.plot_episode_stats(stats)