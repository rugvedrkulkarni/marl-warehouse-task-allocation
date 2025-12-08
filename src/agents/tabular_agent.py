"""
Tabular Q-Learning Agent (Baseline)

Implements tabular Q-learning with:
- Discretized state space
- Joint action space for multi-agent coordination
- Epsilon-greedy exploration with decay

Author: Rugved R. Kulkarni
Course: CSE 546 - Reinforcement Learning, Fall 2025
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import defaultdict


class TabularQAgent:
    """
    Tabular Q-Learning Agent for Warehouse Task Allocation.
    
    Uses discretization to handle continuous state spaces and
    maintains a Q-table for joint action-value estimation.
    """
    
    def __init__(
        self,
        state_bins: List[int],
        state_ranges: List[Tuple[float, float]],
        action_dim: int,
        learning_rate: float = 0.2,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """
        Initialize tabular Q-learning agent.
        
        Args:
            state_bins: Number of bins for each state dimension
            state_ranges: (min, max) range for each state dimension
            action_dim: Number of actions
            learning_rate: Q-learning rate (alpha)
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
        """
        self.state_bins = state_bins
        self.state_ranges = state_ranges
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state_tuple -> action -> Q-value
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_dim)
        )
        
        # Statistics
        self.update_count = 0
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Discretize continuous state to bin indices.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple of bin indices
        """
        indices = []
        for i, (val, (lo, hi), n_bins) in enumerate(
            zip(state, self.state_ranges, self.state_bins)
        ):
            # Clip to range
            val = np.clip(val, lo, hi)
            # Compute bin index
            bin_idx = int((val - lo) / (hi - lo) * (n_bins - 1))
            bin_idx = min(bin_idx, n_bins - 1)
            indices.append(bin_idx)
        
        return tuple(indices)
    
    def select_action(
        self,
        state: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            mask: Boolean mask of valid actions (True = valid)
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Discretize state
        state_key = self.discretize_state(state)
        
        # Get valid actions
        if mask is not None:
            valid_actions = np.where(mask)[0]
        else:
            valid_actions = np.arange(self.action_dim)
        
        if len(valid_actions) == 0:
            valid_actions = np.arange(self.action_dim)
        
        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Greedy action with masking
        q_values = self.q_table[state_key].copy()
        
        if mask is not None:
            q_values[~mask] = -np.inf
        
        return int(np.argmax(q_values))
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Update Q-value using TD learning.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether episode ended
            next_mask: Mask for valid actions in next state
            
        Returns:
            TD error
        """
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        # Get max Q for next state
        if done:
            max_next_q = 0
        else:
            next_q = self.q_table[next_state_key].copy()
            if next_mask is not None:
                next_q[~next_mask] = -np.inf
            max_next_q = np.max(next_q)
        
        # TD target
        target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = target - self.q_table[state_key][action]
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * td_error
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.update_count += 1
        
        return abs(td_error)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state."""
        state_key = self.discretize_state(state)
        return self.q_table[state_key].copy()
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "q_table_size": len(self.q_table),
        }
    
    def save(self, path: str):
        """Save Q-table to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'update_count': self.update_count,
            }, f)
    
    def load(self, path: str):
        """Load Q-table from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_dim), data['q_table'])
        self.epsilon = data['epsilon']
        self.update_count = data['update_count']


class SARSAAgent(TabularQAgent):
    """
    SARSA Agent (On-Policy TD Control).
    
    Updates Q-values using the actual next action taken,
    rather than the maximum Q-value.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_action = None
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool
    ) -> float:
        """
        Update Q-value using SARSA update rule.
        
        Args:
            state: Current state
            action: Current action
            reward: Received reward
            next_state: Next state
            next_action: Next action (already selected)
            done: Whether episode ended
            
        Returns:
            TD error
        """
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        # SARSA target uses actual next action
        if done:
            next_q = 0
        else:
            next_q = self.q_table[next_state_key][next_action]
        
        target = reward + self.gamma * next_q
        td_error = target - self.q_table[state_key][action]
        
        self.q_table[state_key][action] += self.learning_rate * td_error
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.update_count += 1
        
        return abs(td_error)


def create_warehouse_agent(
    n_stations: int = 2,
    queue_capacity: int = 5,
    max_processing: int = 3,
    **kwargs
) -> TabularQAgent:
    """
    Create a tabular Q-learning agent for the warehouse environment.
    
    Args:
        n_stations: Number of workstations
        queue_capacity: Maximum queue capacity
        max_processing: Maximum jobs processing per station
        **kwargs: Additional arguments for TabularQAgent
        
    Returns:
        Configured TabularQAgent
    """
    # State space: [queue_len_0, ..., proc_load_0, ..., avg_remaining_0, ..., pending]
    n_dims = 3 * n_stations + 1
    
    # Discretization settings
    state_bins = []
    state_ranges = []
    
    # Queue lengths (normalized)
    for _ in range(n_stations):
        state_bins.append(queue_capacity + 1)
        state_ranges.append((0.0, 1.0))
    
    # Processing loads (normalized)
    for _ in range(n_stations):
        state_bins.append(max_processing + 1)
        state_ranges.append((0.0, 1.0))
    
    # Average remaining times
    for _ in range(n_stations):
        state_bins.append(5)  # 5 bins for remaining time
        state_ranges.append((0.0, 1.0))
    
    # Pending job flag
    state_bins.append(2)
    state_ranges.append((0.0, 1.0))
    
    # Joint action space: (n_stations + 1) dispatcher actions * 2^n_stations acceptor combinations
    action_dim = (n_stations + 1) * (2 ** n_stations)
    
    return TabularQAgent(
        state_bins=state_bins,
        state_ranges=state_ranges,
        action_dim=action_dim,
        **kwargs
    )


if __name__ == "__main__":
    # Test tabular agent
    agent = create_warehouse_agent(
        n_stations=2,
        queue_capacity=5,
        learning_rate=0.2,
        gamma=0.99,
    )
    
    print(f"State bins: {agent.state_bins}")
    print(f"Action dim: {agent.action_dim}")
    
    # Simulate some training
    for episode in range(10):
        state = np.random.rand(7).astype(np.float32)
        total_reward = 0
        
        for step in range(50):
            mask = np.random.rand(12) > 0.3  # Random mask
            action = agent.select_action(state, mask)
            
            next_state = np.random.rand(7).astype(np.float32)
            reward = np.random.randn()
            done = step == 49
            
            td_error = agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, "
              f"Epsilon = {agent.epsilon:.4f}, Q-table size = {len(agent.q_table)}")
