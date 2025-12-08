"""
Training Script for Multi-Agent PPO (MAPPO)

Trains the MAPPO agent on the warehouse task allocation environment.

Usage:
    python train_mappo.py --episodes 2000 --n_stations 2

Author: Rugved R. Kulkarni
Course: CSE 546 - Reinforcement Learning, Fall 2025
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.warehouse_env import MultiAgentWarehouseEnv
from agents.mappo_agent import SequentialMAPPO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MAPPO on Warehouse Environment")
    
    # Environment
    parser.add_argument("--n_stations", type=int, default=2, help="Number of workstations")
    parser.add_argument("--queue_capacity", type=int, default=5, help="Queue capacity per station")
    parser.add_argument("--max_processing", type=int, default=3, help="Max processing per station")
    
    # Training
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--rollout_length", type=int, default=128, help="Rollout length before update")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # MAPPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--n_epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64], help="Hidden dims")
    
    # Logging
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=500, help="Model save frequency")
    parser.add_argument("--save_dir", type=str, default="results/models", help="Save directory")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, n_episodes: int = 10) -> dict:
    """
    Evaluate the agent.
    
    Args:
        env: Environment
        agent: MAPPO agent
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_rewards = []
    total_completed = []
    total_rejected = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        while True:
            current_agent = env.get_current_agent()
            
            if current_agent == "dispatcher":
                # Get dispatcher mask
                mask = agent.get_dispatcher_mask(
                    [len(q) for q in env.queues],
                    env.queue_capacity
                )
                action, _, _ = agent.select_action("dispatcher", obs, mask, training=False)
                obs, reward, terminated, truncated, info = env.step_dispatcher(action)
            else:
                # Acceptor
                station_idx = int(current_agent.split("_")[1])
                action, _, _ = agent.select_action(current_agent, obs, training=False)
                obs, reward, terminated, truncated, info = env.step_acceptor(station_idx, action)
            
            episode_reward += reward
            
            if info["step_count"] >= 50:  # Max steps
                break
        
        total_rewards.append(episode_reward)
        total_completed.append(info["jobs_completed"])
        total_rejected.append(info["jobs_rejected"])
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_completed": np.mean(total_completed),
        "mean_rejected": np.mean(total_rejected),
    }


def train(args):
    """Main training loop."""
    # Set seed
    set_seed(args.seed)
    
    # Create experiment name
    if args.exp_name is None:
        args.exp_name = f"mappo_{args.n_stations}stations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create save directory
    save_path = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Create environment
    env = MultiAgentWarehouseEnv(
        n_stations=args.n_stations,
        queue_capacity=args.queue_capacity,
        max_processing=args.max_processing,
    )
    
    # Calculate state dimension
    state_dim = 3 * args.n_stations + 1
    
    # Create agent
    agent = SequentialMAPPO(
        state_dim=state_dim,
        n_stations=args.n_stations,
        hidden_dims=args.hidden_dims,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        n_epochs=args.n_epochs,
    )
    
    print(f"Training MAPPO with {args.n_stations} stations")
    print(f"State dim: {state_dim}, Device: {agent.device}")
    print(f"Save path: {save_path}")
    print("-" * 50)
    
    # Training metrics
    episode_rewards = []
    episode_completed = []
    best_eval_reward = -float('inf')
    
    # Training loop
    pbar = tqdm(range(args.episodes), desc="Training")
    
    for episode in pbar:
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while step_count < args.max_steps:
            current_agent = env.get_current_agent()
            
            if current_agent == "dispatcher":
                # Dispatcher action
                mask = agent.get_dispatcher_mask(
                    [len(q) for q in env.queues],
                    env.queue_capacity
                )
                action, log_prob, value = agent.select_action("dispatcher", obs, mask)
                
                new_obs, reward, terminated, truncated, info = env.step_dispatcher(action)
                
                # Store transition (reward will be shared)
                agent.store_transition(
                    "dispatcher", obs, action, reward,
                    value, log_prob, terminated or truncated, mask
                )
                
            else:
                # Acceptor action
                station_idx = int(current_agent.split("_")[1])
                action, log_prob, value = agent.select_action(current_agent, obs)
                
                new_obs, reward, terminated, truncated, info = env.step_acceptor(station_idx, action)
                
                # Store transition
                agent.store_transition(
                    current_agent, obs, action, reward,
                    value, log_prob, terminated or truncated
                )
            
            episode_reward += reward
            obs = new_obs
            
            if env.current_agent == "dispatcher":
                step_count += 1
        
        # Update agent
        losses = agent.update(obs)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_completed.append(info["jobs_completed"])
        
        # Update progress bar
        pbar.set_postfix({
            "reward": f"{episode_reward:.1f}",
            "completed": info["jobs_completed"],
        })
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_metrics = evaluate(env, agent, n_episodes=10)
            
            print(f"\nEpisode {episode + 1}:")
            print(f"  Eval Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Jobs Completed: {eval_metrics['mean_completed']:.1f}")
            print(f"  Jobs Rejected: {eval_metrics['mean_rejected']:.1f}")
            
            # Save best model
            if eval_metrics['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['mean_reward']
                agent.save(os.path.join(save_path, "best_model.pt"))
                print(f"  New best model saved!")
        
        # Regular save
        if (episode + 1) % args.save_freq == 0:
            agent.save(os.path.join(save_path, f"model_ep{episode + 1}.pt"))
    
    # Final save
    agent.save(os.path.join(save_path, "final_model.pt"))
    
    # Save training curves
    np.savez(
        os.path.join(save_path, "training_curves.npz"),
        rewards=np.array(episode_rewards),
        completed=np.array(episode_completed),
    )
    
    print("\nTraining complete!")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final avg completed (last 100): {np.mean(episode_completed[-100:]):.1f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
