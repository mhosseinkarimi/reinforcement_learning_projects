import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Bandit:
    def __init__(self, actions, action_values, reward_var=1):
        self.actions = actions
        self.action_values = action_values
        assert len(actions) == len(action_values)
        
        if isinstance(reward_var, (int, float)):
            self.reward_var = [reward_var] * len(actions) 
        else:
            assert len(reward_var) == len(actions)
            self.reward_var = reward_var
    
    def step(self, a):
        return np.random.normal(self.action_values[a], self.reward_var[a])
    
    def get_action_values(self):
        return self.action_values
    
    def visualize_action_values_distribution(self):
        rewards = np.concatenate([np.array([self.step(a) for _ in range(100)]) for a in range(len(self.actions))])
        action_index = np.concatenate([[a] * 100 for a in range(len(self.actions))])
        sns.violinplot(x=action_index, y=rewards, inner="quartile", density_norm="width")
        plt.xticks(range(len(self.actions)), self.actions)
        plt.xlabel("Actions")
        plt.ylabel("Reward Distribution")
        plt.title("Reward Distribution for Each Action")
        plt.show()

if __name__ == "__main__":
    # Testing the Bandit class
    actions = ["A", "B", "C"]
    action_values = [0.5, 0.7, 0.9]
    bandit = Bandit(actions, action_values)
    bandit.visualize_action_values_distribution()