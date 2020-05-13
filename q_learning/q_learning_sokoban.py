import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
#import gym_gridworlds


if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from cliff_walk.cliff_walking import CliffWalkingEnv
#from ai_safety_gridworlds.sokoban.environment.side_effects_sokoban import SideEffectsSokobanEnvironment
from q_learning import plotting
from safe_grid_gym.gym_interface.envs.gridworlds_env import GridworldEnv
from ai_safety_gridworlds.distributional_shift_gym import SideEffectsSokobanEnv

matplotlib.style.use('ggplot')

env = GridworldEnv('side_effects_sokoban')

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        # #Finds where the agent is(2), use this for sokoban
        # obs=np.where(observation == 2)
        # print(obs)
        # #This sets concatenate the row and the column and returns a list
        # obs2 = np.concatenate((obs)).tolist()
        # print(obs2)
        #
        # #Converts it to a int number, use this one for sokoban
        # obs3 = int(''.join(map(str, obs2)))
        # print(obs3)
        # best_action = np.argmax(Q[obs3])

        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn



def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(4))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))


    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, 4)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        print(state)
        # Finds where the agent is(2), use this for sokoban
        state_find_agent = np.where(state == 2)
        print(state_find_agent)
        # This sets concatenate the row and the column and returns a list
        state_agent_con = np.concatenate((state_find_agent)).tolist()
        print(state_agent_con)

        # Converts it to a int number, use this for sokoban
        state_to_int = int(''.join(map(str, state_agent_con)))
        print(state_to_int)

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state_to_int)

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            #print(next_state)

            # Finds where the agent is(2) for the next state, use this for sokoban
            next_obs = np.where(next_state == 2)
            #print(next_obs)
            # This sets concatenate the row and the column and returns a list
            next_obs_con = np.concatenate((next_obs)).tolist()
            #print(next_obs_con)

            # Converts it to a int number, use this one for sokoban
            next_obs_int = int(''.join(map(str, next_obs_con)))
            #print(next_obs_int)


            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_obs_int])
            td_target = reward + discount_factor * Q[next_obs_int][best_next_action]
            td_delta = td_target - Q[state_to_int][action]
            Q[state_to_int][action] += alpha * td_delta

            if done:
                break

            state_to_int = next_obs_int

    return Q, stats

#Q, stats = q_learning(env, 500)
#plotting.plot_episode_stats(stats)