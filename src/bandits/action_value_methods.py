import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.bandits.bandit import Bandit


class EpsilonGreedy:
    def __init__(self, env, epsilon=0.1, step_size=0.1, initial_action_value=0, weighting_mode="sample_average"):
        self.env = env
        self.epsilon = epsilon
        if isinstance(initial_action_value, (int, float)):
            self.action_values = initial_action_value * np.ones(len(env.actions))
        else:
            assert len(initial_action_value) == len(env.actions)
            self.action_values = initial_action_value

        self.weighting_mode = weighting_mode
        if weighting_mode == "sample_average":
            self.action_counts = np.zeros(len(env.actions))
        elif weighting_mode == "constant":
            self.step_size = step_size
        self.action_values = np.zeros(len(env.actions))
        self.action_counts = np.zeros(len(env.actions))
        self.total_reward = 0
        self.total_steps = 0
        self.action = None
        self.reward = None
        self.reward_history = []
        self.action_history = []
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(self.env.actions))
        else:
            max_value = np.max(self.action_values)
            max_actions = np.where(self.action_values == max_value)[0]
            action = np.random.choice(max_actions)
        return action
    
    def update_action_values(self):
        if self.action_counts[self.action] == 0:
            return
        
        if self.weighting_mode == "sample_average":
            self.action_values[self.action] += (self.reward - self.action_values[self.action]) / self.action_counts[self.action]
        elif self.weighting_mode == "constant":
            self.action_values[self.action] += self.step_size * (self.reward - self.action_values[self.action])
    
    def step(self):
        self.action = self.select_action()
        self.reward = self.env.step(self.action)
        self.total_reward += self.reward
        self.total_steps += 1
        self.action_counts[self.action] += 1
        self.reward_history.append(self.reward)
        self.action_history.append(self.action)
        self.update_action_values()
    
    def run_episode(self, num_steps):
        for step in range(num_steps):
            self.step()
        return self.reward_history, self.action_history, self.total_reward
    
    def run(self, num_steps, num_episodes=1):
        episode_total_rewards = []
        episode_rewards = []
        episode_actions = []
        for episode in range(num_episodes):
            self.reward_history = []
            self.action_history = []
            self.total_reward = 0
            self.total_steps = 0
            self.action_values = np.zeros(len(self.env.actions))
            self.action_counts = np.zeros(len(self.env.actions))
            self.action = None
            self.reward = None
            rewards, actions, total_reward = self.run_episode(num_steps)
            episode_rewards.append(rewards)
            episode_actions.append(actions)
            episode_total_rewards.append(total_reward)
        return np.array(episode_rewards), np.array(episode_actions), np.array(episode_total_rewards)

class UCB:
    def __init__(self, env, c=2):
        self.env = env
        self.c = c
        self.action_values = np.zeros(len(env.actions))
        self.action_counts = np.zeros(len(env.actions))
        self.total_reward = 0
        self.total_steps = 0
        self.action = None
        self.reward = None
        self.reward_history = []
        self.action_history = []
        
    def select_action(self):
        if self.total_steps < len(self.env.actions):
            return self.total_steps
        else:
            ucb_values = self.action_values + self.c * np.sqrt(np.log(self.total_steps) / self.action_counts)
            return np.argmax(ucb_values)
    
    def update_action_values(self):
        if self.action_counts[self.action] == 0:
            return
        
        self.action_values[self.action] += (self.reward - self.action_values[self.action]) / self.action_counts[self.action]
    
    def step(self):
        self.action = self.select_action()
        self.reward = self.env.step(self.action)
        self.total_reward += self.reward
        self.total_steps += 1
        self.action_counts[self.action] += 1
        self.reward_history.append(self.reward)
        self.action_history.append(self.action)
        self.update_action_values()
    
    def run_episode(self, num_steps):
        for step in range(num_steps):
            self.step()
        return self.reward_history, self.action_history, self.total_reward
    
    def run(self, num_steps, num_episodes=1):
        episode_total_rewards = []
        episode_rewards = []
        episode_actions = []
        for episode in range(num_episodes):
            self.reward_history = []
            self.action_history = []
            self.total_reward = 0
            self.total_steps = 0
            self.action_values = np.zeros(len(self.env.actions))
            self.action_counts = np.zeros(len(self.env.actions))
            self.action = None
            self.reward = None
            rewards, actions, total_reward = self.run_episode(num_steps)
            episode_rewards.append(rewards)
            episode_actions.append(actions)
            episode_total_rewards.append(total_reward)
        return np.array(episode_rewards), np.array(episode_actions), np.array(episode_total_rewards)

if __name__ == "__main__":
    # Espsilon-Greedy Algorithm Usage Example
    k = 5
    actions = list(range(k))
    action_values = [0, 2, -5, -3, 5]
    num_steps = 1000
    num_episodes = 1000
    bandit = Bandit(actions, action_values, reward_var=2)
    bandit.visualize_action_values_distribution()
    
    ## Experiment with different epsilon values
    epsilons = [0.1, 0.05, 0.01]
    for epsilon in epsilons:
        epsilon_greedy = EpsilonGreedy(bandit, epsilon)
        episode_rewards, episode_actions, episode_total_rewards = epsilon_greedy.run(num_steps, num_episodes)
        # Plotting the rewards
        reward_mean = np.mean(episode_rewards, axis=0)
        reward_std = np.std(episode_rewards, axis=0)
        plt.plot(reward_mean, label=f"$\epsilon$: {epsilon}")
        plt.fill_between(range(num_steps), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"$\epsilon$-Greedy Action-value Method Applied on {k}-armed Bandit")
    plt.legend(loc="lower right")
    plt.show()
    
    ## Experimenting with constant step size
    step_sizes = [0.1, 0.01, 0.001]
    epsilon = 0.05
    for alpha in step_sizes:
        epsilon_greedy = EpsilonGreedy(bandit, epsilon, step_size=alpha, weighting_mode="constant")
        episode_rewards, episode_actions, episode_total_rewards = epsilon_greedy.run(num_steps, num_episodes)
        # Plotting the rewards
        reward_mean = np.mean(episode_rewards, axis=0)
        reward_std = np.std(episode_rewards, axis=0)
        plt.plot(reward_mean, label=f"$\\alpha$: {alpha}")
        plt.fill_between(range(num_steps), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"$\epsilon$-Greedy Action-value Method Applied on {k}-armed Bandit")
    plt.legend(loc="lower right")
    plt.show()
    
    ## Experimenting with Optimistic Initial Action Values
    initial_action_values = [0, 2, 5, 20]
    epsilon = 0.05
    alpha = 0.1
    for initial_value in initial_action_values:
        epsilon_greedy = EpsilonGreedy(bandit, epsilon, initial_action_value=initial_value)
        episode_rewards, episode_actions, episode_total_rewards = epsilon_greedy.run(num_steps, num_episodes)
        # Plotting the rewards
        reward_mean = np.mean(episode_rewards, axis=0)
        reward_std = np.std(episode_rewards, axis=0)
        plt.plot(reward_mean, label=f"Initial Action-Value: {initial_value}")
        plt.fill_between(range(num_steps), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"$\epsilon$-Greedy Action-value Method Applied on {k}-armed Bandit")
    plt.legend(loc="lower right")
    plt.show()
    
    ## Comparison between epsilon-greedy and Optimistic Initial Action Values
    initial_action_values = [0, 20] 
    epsilons = [0.01, 0]
    alpha = 0.1
    for init_val, epsilon in zip(initial_action_values, epsilons):
        epsilon_greedy = EpsilonGreedy(bandit, epsilon, initial_action_value=init_val, step_size=alpha, weighting_mode="constant")
        episode_rewards, episode_actions, episode_total_rewards = epsilon_greedy.run(num_steps, num_episodes)
        # Plotting the rewards
        reward_mean = np.mean(episode_rewards, axis=0)
        reward_std = np.std(episode_rewards, axis=0)
        plt.plot(reward_mean, label=f"Initial Action-Value: {init_val}, $\epsilon$: {epsilon}")
        plt.fill_between(range(num_steps), reward_mean - reward_std,reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Comparison of $\epsilon$-Greedy and Optimistic Initial Action Values")
    plt.legend(loc="lower right")
    plt.show()
    
    for init_val, epsilon in zip(initial_action_values, epsilons):
        epsilon_greedy = EpsilonGreedy(bandit, epsilon, initial_action_value=init_val, step_size=alpha, weighting_mode="constant")
        episode_rewards, episode_actions, episode_total_rewards = epsilon_greedy.run(num_steps, num_episodes)
        # Plotting the optimal action percentage
        optimal_action = np.array(np.sum(episode_actions[i, :] == np.argmax(action_values) for i in range(num_episodes)) / num_episodes)
        plt.plot(100 * optimal_action, label=f"$Q_1$: {init_val}, $\epsilon$: {epsilon}")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Comparison of $\epsilon$-Greedy and Optimistic Initial Action Values")
    plt.legend(loc="lower right")
    plt.show()
    
    # Experimenting with UCB
    c_values = [0, 1, 2, 3]
    for c in c_values:
        ucb = UCB(bandit, c)
        episode_rewards, episode_actions, episode_total_rewards = ucb.run(num_steps, num_episodes)
        # Plotting the rewards
        reward_mean = np.mean(episode_rewards, axis=0)
        reward_std = np.std(episode_rewards, axis=0)
        plt.plot(reward_mean, label=f"c: {c}")
        plt.fill_between(range(num_steps), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"UCB Action-value Method Applied on {k}-armed Bandit")
    plt.legend(loc="lower right")
    plt.show()


    
    