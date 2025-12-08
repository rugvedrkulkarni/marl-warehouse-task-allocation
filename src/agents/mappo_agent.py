"""
Multi-Agent Proximal Policy Optimization (MAPPO) with CTDE Architecture

Implements Multi-Agent PPO with:
- Centralized Training with Decentralized Execution (CTDE)
- Shared critic network across agents
- Independent actor policies per agent
- Action masking for constraint satisfaction
- Sequential decision-making for multi-agent coordination

Author: Rugved R. Kulkarni
Course: CSE 546 - Reinforcement Learning, Fall 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ActorNetwork(nn.Module):
    """Actor network for policy learning."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [64, 64]
    ):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action logits.
        
        Args:
            state: State tensor
            
        Returns:
            Action logits (before softmax)
        """
        return self.network(state)
    
    def get_distribution(
        self, 
        state: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Categorical:
        """
        Get action distribution with optional masking.
        
        Args:
            state: State tensor
            mask: Boolean mask tensor (True = valid action)
            
        Returns:
            Categorical distribution over actions
        """
        logits = self.forward(state)
        
        if mask is not None:
            # Mask invalid actions with large negative value
            logits = logits.masked_fill(~mask, float('-inf'))
        
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """Critic network for value estimation (shared across agents)."""
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dims: List[int] = [64, 64]
    ):
        """
        Initialize critic network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning state value.
        
        Args:
            state: State tensor
            
        Returns:
            State value estimate
        """
        return self.network(state).squeeze(-1)


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear all stored data."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.masks = []
        self.agent_ids = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        mask: Optional[np.ndarray] = None,
        agent_id: str = "default"
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.masks.append(mask)
        self.agent_ids.append(agent_id)
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all stored data as tensors."""
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'log_probs': torch.FloatTensor(self.log_probs),
            'dones': torch.FloatTensor(self.dones),
        }
    
    def __len__(self) -> int:
        return len(self.states)


class MAPPOAgent:
    """
    Multi-Agent PPO Agent with CTDE Architecture.
    
    Implements Centralized Training with Decentralized Execution:
    - Each agent has its own actor network (policy)
    - All agents share a common critic network (value function)
    - Cooperative reward structure for multi-agent coordination
    """
    
    def __init__(
        self,
        state_dim: int,
        agent_configs: Dict[str, int],  # agent_name -> action_dim
        hidden_dims: List[int] = [64, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "auto",
    ):
        """
        Initialize MAPPO agent.
        
        Args:
            state_dim: Dimension of state space
            agent_configs: Dictionary mapping agent names to their action dimensions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of optimization epochs per update
            batch_size: Minibatch size for training
            device: Device to use ("auto", "cpu", or "cuda")
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.agent_configs = agent_configs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Create actor networks (one per agent)
        self.actors: Dict[str, ActorNetwork] = {}
        self.actor_optimizers: Dict[str, optim.Adam] = {}
        
        for agent_name, action_dim in agent_configs.items():
            self.actors[agent_name] = ActorNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)
            
            self.actor_optimizers[agent_name] = optim.Adam(
                self.actors[agent_name].parameters(), lr=learning_rate
            )
        
        # Shared critic network
        self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Rollout buffer (per agent)
        self.buffers: Dict[str, RolloutBuffer] = {
            name: RolloutBuffer() for name in agent_configs.keys()
        }
    
    def select_action(
        self,
        agent_name: str,
        state: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action for a specific agent.
        
        Args:
            agent_name: Name of the agent
            state: Current state
            mask: Boolean mask of valid actions
            training: Whether in training mode
            
        Returns:
            (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get mask tensor
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get action distribution
            dist = self.actors[agent_name].get_distribution(state_tensor, mask_tensor)
            
            # Sample action
            if training:
                action = dist.sample()
            else:
                action = dist.probs.argmax(dim=-1)
            
            log_prob = dist.log_prob(action)
            
            # Get value estimate
            value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        agent_name: str,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        mask: Optional[np.ndarray] = None
    ):
        """Store transition for a specific agent."""
        self.buffers[agent_name].add(
            state, action, reward, value, log_prob, done, mask, agent_name
        )
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward tensor
            values: Value estimates tensor
            dones: Done flags tensor
            next_value: Value estimate for the state after the last step
            
        Returns:
            (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        
        last_gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        Update all agents using collected rollouts.
        
        Args:
            next_state: State after the last step (for bootstrapping)
            
        Returns:
            Dictionary of losses for each agent
        """
        losses = {}
        
        # Get next value for bootstrapping
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).item()
        
        # Update each agent
        for agent_name in self.agent_configs.keys():
            buffer = self.buffers[agent_name]
            
            if len(buffer) == 0:
                continue
            
            data = buffer.get()
            
            states = data['states'].to(self.device)
            actions = data['actions'].to(self.device)
            old_log_probs = data['log_probs'].to(self.device)
            rewards = data['rewards']
            values = data['values']
            dones = data['dones']
            
            # Compute advantages
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            advantages = advantages.to(self.device)
            returns = returns.to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            
            for _ in range(self.n_epochs):
                # Get current policy distribution
                dist = self.actors[agent_name].get_distribution(states)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                new_values = self.critic(states)
                critic_loss = nn.MSELoss()(new_values, returns)
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update actor
                self.actor_optimizers[agent_name].zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(
                    self.actors[agent_name].parameters(), self.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                
                self.actor_optimizers[agent_name].step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
            
            losses[f"{agent_name}_actor_loss"] = total_actor_loss / self.n_epochs
            losses[f"{agent_name}_critic_loss"] = total_critic_loss / self.n_epochs
            losses[f"{agent_name}_entropy"] = total_entropy / self.n_epochs
            
            # Clear buffer
            buffer.reset()
        
        return losses
    
    def save(self, path: str):
        """Save all networks to file."""
        save_dict = {
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        
        for name in self.agent_configs.keys():
            save_dict[f'{name}_actor'] = self.actors[name].state_dict()
            save_dict[f'{name}_optimizer'] = self.actor_optimizers[name].state_dict()
        
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load all networks from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        for name in self.agent_configs.keys():
            self.actors[name].load_state_dict(checkpoint[f'{name}_actor'])
            self.actor_optimizers[name].load_state_dict(checkpoint[f'{name}_optimizer'])


class SequentialMAPPO(MAPPOAgent):
    """
    Sequential Multi-Agent PPO for warehouse task allocation.
    
    Implements sequential decision-making:
    1. Dispatcher decides routing
    2. Only the target station's acceptor decides acceptance
    
    This prevents conflicts between acceptor agents.
    """
    
    def __init__(
        self,
        state_dim: int,
        n_stations: int,
        hidden_dims: List[int] = [64, 64],
        **kwargs
    ):
        """
        Initialize Sequential MAPPO.
        
        Args:
            state_dim: Dimension of state space
            n_stations: Number of workstations
            hidden_dims: Hidden layer dimensions
            **kwargs: Additional arguments for MAPPOAgent
        """
        # Create agent configs
        agent_configs = {
            'dispatcher': n_stations + 1,  # Route to each station or reject
        }
        for i in range(n_stations):
            agent_configs[f'acceptor_{i}'] = 2  # Accept or reject
        
        super().__init__(state_dim, agent_configs, hidden_dims, **kwargs)
        
        self.n_stations = n_stations
    
    def get_dispatcher_mask(self, queue_lengths: List[int], queue_capacity: int) -> np.ndarray:
        """
        Get action mask for dispatcher.
        
        Args:
            queue_lengths: Current queue lengths for each station
            queue_capacity: Maximum queue capacity
            
        Returns:
            Boolean mask (True = valid action)
        """
        mask = np.ones(self.n_stations + 1, dtype=bool)
        
        # Mask stations with full queues
        for i, length in enumerate(queue_lengths):
            if length >= queue_capacity:
                mask[i] = False
        
        return mask


if __name__ == "__main__":
    # Test MAPPO agent
    agent = SequentialMAPPO(
        state_dim=7,
        n_stations=2,
        hidden_dims=[64, 64],
    )
    
    # Simulate episode
    state = np.random.randn(7).astype(np.float32)
    
    for step in range(10):
        # Dispatcher action
        disp_mask = np.array([True, True, True])  # All valid
        disp_action, disp_log_prob, disp_value = agent.select_action(
            'dispatcher', state, disp_mask
        )
        
        print(f"Step {step}: Dispatcher -> {disp_action}")
        
        if disp_action < 2:  # Routed to a station
            # Acceptor action
            acc_action, acc_log_prob, acc_value = agent.select_action(
                f'acceptor_{disp_action}', state
            )
            print(f"         Acceptor {disp_action} -> {'Accept' if acc_action else 'Reject'}")
            
            # Store transitions
            reward = 1.0 if acc_action else -2.0
            agent.store_transition('dispatcher', state, disp_action, reward/2, 
                                   disp_value, disp_log_prob, False)
            agent.store_transition(f'acceptor_{disp_action}', state, acc_action, reward/2,
                                   acc_value, acc_log_prob, False)
        else:
            # Dispatcher rejected
            reward = -5.0
            agent.store_transition('dispatcher', state, disp_action, reward,
                                   disp_value, disp_log_prob, False)
        
        state = np.random.randn(7).astype(np.float32)
    
    # Update
    losses = agent.update(state)
    print(f"\nLosses: {losses}")
