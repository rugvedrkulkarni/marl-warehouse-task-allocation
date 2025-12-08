"""
Deep Q-Network (DQN) Agent with Action Masking

Implements DQN with:
- Experience replay buffer
- Target network for stability
- Action masking for constraint satisfaction

Author: Rugved R. Kulkarni
Course: CSE 546 - Reinforcement Learning, Fall 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional, List


class QNetwork(nn.Module):
    """Q-value neural network."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [128, 128]
    ):
        """
        Initialize Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        mask: Optional[np.ndarray] = None
    ):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done, mask))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        masks = [t[5] for t in batch]
        
        return states, actions, rewards, next_states, dones, masks
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with Action Masking."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "auto",
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            device: Device to use ("auto", "cpu", or "cuda")
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training counters
        self.train_steps = 0
    
    def select_action(
        self, 
        state: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy with action masking.
        
        Args:
            state: Current state
            mask: Boolean mask of valid actions (True = valid)
            training: Whether in training mode (uses exploration)
            
        Returns:
            Selected action
        """
        # Get valid actions
        if mask is not None:
            valid_actions = np.where(mask)[0]
        else:
            valid_actions = np.arange(self.action_dim)
        
        if len(valid_actions) == 0:
            # Fallback: all actions valid
            valid_actions = np.arange(self.action_dim)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Greedy action selection with masking
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions with large negative value
        if mask is not None:
            q_values[~mask] = -float('inf')
        
        return int(np.argmax(q_values))
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        mask: Optional[np.ndarray] = None
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done, mask)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, masks = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values with masking
        with torch.no_grad():
            next_q = self.target_network(next_states)
            
            # Apply masks to next Q values
            for i, mask in enumerate(masks):
                if mask is not None:
                    invalid_mask = torch.FloatTensor(~mask).to(self.device)
                    next_q[i] = next_q[i] - invalid_mask * 1e9
            
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, path)
    
    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent - reduces overestimation bias."""
    
    def train_step(self) -> Optional[float]:
        """Perform one training step with Double DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, masks = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            # Select actions using online network
            online_q = self.q_network(next_states)
            
            # Apply masks
            for i, mask in enumerate(masks):
                if mask is not None:
                    invalid_mask = torch.FloatTensor(~mask).to(self.device)
                    online_q[i] = online_q[i] - invalid_mask * 1e9
            
            best_actions = online_q.argmax(1)
            
            # Evaluate using target network
            target_q_values = self.target_network(next_states)
            max_next_q = target_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()


if __name__ == "__main__":
    # Test DQN agent
    agent = DQNAgent(
        state_dim=7,
        action_dim=12,
        hidden_dims=[64, 64],
    )
    
    # Simulate some training
    for _ in range(100):
        state = np.random.randn(7).astype(np.float32)
        mask = np.random.rand(12) > 0.3
        action = agent.select_action(state, mask)
        next_state = np.random.randn(7).astype(np.float32)
        reward = np.random.randn()
        done = False
        
        agent.store_transition(state, action, reward, next_state, done, mask)
        loss = agent.train_step()
        
        if loss is not None:
            print(f"Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}")
