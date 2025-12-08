"""
Multi-Agent Warehouse Task Allocation Environment

A custom Gymnasium environment simulating a warehouse with:
- 1 Dispatcher agent that routes incoming jobs
- N Acceptor agents (one per workstation) that accept/reject jobs

Author: Rugved R. Kulkarni
Course: CSE 546 - Reinforcement Learning, Fall 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class WarehouseTaskAllocationEnv(gym.Env):
    """
    Multi-Agent Warehouse Task Allocation Environment.
    
    This environment simulates a warehouse sortation system where:
    - Jobs arrive at a dispatcher
    - The dispatcher routes jobs to workstations or rejects them
    - Each workstation has an acceptor that can accept or reject jobs
    - Jobs are processed over time and completed
    
    The goal is to maximize throughput while respecting capacity constraints.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        n_stations: int = 2,
        queue_capacity: int = 5,
        max_processing: int = 3,
        job_arrival_prob: float = 0.8,
        processing_time_range: Tuple[int, int] = (2, 5),
        congestion_threshold: float = 0.7,
        congestion_penalty: float = 0.8,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the warehouse environment.
        
        Args:
            n_stations: Number of workstations
            queue_capacity: Maximum jobs in each station's queue
            max_processing: Maximum jobs being processed simultaneously per station
            job_arrival_prob: Probability of a new job arriving each step
            processing_time_range: (min, max) processing time for jobs
            congestion_threshold: Queue fill ratio triggering congestion penalty
            congestion_penalty: Penalty coefficient for congestion
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        self.n_stations = n_stations
        self.queue_capacity = queue_capacity
        self.max_processing = max_processing
        self.job_arrival_prob = job_arrival_prob
        self.processing_time_range = processing_time_range
        self.congestion_threshold = congestion_threshold
        self.congestion_penalty = congestion_penalty
        self.render_mode = render_mode
        
        # Observation space: 
        # [queue_length_0, queue_length_1, ..., proc_load_0, proc_load_1, ..., 
        #  avg_remaining_0, avg_remaining_1, ..., pending_job_flag]
        obs_dim = 3 * n_stations + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action spaces
        # Dispatcher: route to station 0, station 1, ..., or reject
        self.dispatcher_action_space = spaces.Discrete(n_stations + 1)
        
        # Acceptors: accept (1) or reject (0)
        self.acceptor_action_spaces = [spaces.Discrete(2) for _ in range(n_stations)]
        
        # For compatibility with single-agent algorithms (joint action space)
        # Joint action = dispatcher_action * (2^n_stations) + acceptor_actions_binary
        self.joint_action_space = spaces.Discrete(
            (n_stations + 1) * (2 ** n_stations)
        )
        
        # Internal state
        self.reset()
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Station queues: list of remaining processing times
        self.queues: List[List[int]] = [[] for _ in range(self.n_stations)]
        
        # Jobs being processed: list of remaining times
        self.processing: List[List[int]] = [[] for _ in range(self.n_stations)]
        
        # Pending job (job waiting to be dispatched)
        self.pending_job = False
        self.pending_job_time = 0
        
        # Statistics
        self.jobs_completed = 0
        self.jobs_rejected = 0
        self.total_jobs_arrived = 0
        self.step_count = 0
        
        # Generate first job
        self._maybe_generate_job()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Joint action (dispatcher + acceptors encoded)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        reward = 0.0
        
        # Decode joint action
        dispatcher_action, acceptor_actions = self._decode_action(action)
        
        # Process dispatcher action (only if there's a pending job)
        if self.pending_job:
            if dispatcher_action < self.n_stations:
                # Route to a station
                target_station = dispatcher_action
                
                # Check if acceptor accepts and queue has space
                if (acceptor_actions[target_station] == 1 and 
                    len(self.queues[target_station]) < self.queue_capacity):
                    # Successfully enqueue
                    self.queues[target_station].append(self.pending_job_time)
                    reward += 1.0  # Reward for successful enqueue
                    self.pending_job = False
                else:
                    # Rejected by acceptor or queue full
                    reward -= 2.0
                    self.jobs_rejected += 1
                    self.pending_job = False
            else:
                # Dispatcher rejects
                reward -= 5.0
                self.jobs_rejected += 1
                self.pending_job = False
        
        # Process jobs at each station
        for i in range(self.n_stations):
            # Move jobs from queue to processing if capacity available
            while (len(self.processing[i]) < self.max_processing and 
                   len(self.queues[i]) > 0):
                self.processing[i].append(self.queues[i].pop(0))
            
            # Process jobs (decrease remaining time)
            completed = []
            for j, remaining in enumerate(self.processing[i]):
                self.processing[i][j] = remaining - 1
                if self.processing[i][j] <= 0:
                    completed.append(j)
            
            # Remove completed jobs
            for j in sorted(completed, reverse=True):
                self.processing[i].pop(j)
                self.jobs_completed += 1
                reward += 10.0  # Reward for job completion
        
        # Apply congestion penalty
        for i in range(self.n_stations):
            queue_fill = len(self.queues[i]) / self.queue_capacity
            if queue_fill > self.congestion_threshold:
                excess = queue_fill - self.congestion_threshold
                reward -= self.congestion_penalty * excess
        
        # Generate new job
        self._maybe_generate_job()
        
        # Get observation
        observation = self._get_observation()
        
        # Episode never terminates (use truncation for time limits)
        terminated = False
        truncated = False
        
        return observation, reward, terminated, truncated, self._get_info()
    
    def _decode_action(self, action: int) -> Tuple[int, List[int]]:
        """
        Decode joint action into dispatcher and acceptor actions.
        
        Args:
            action: Joint action integer
            
        Returns:
            (dispatcher_action, acceptor_actions)
        """
        acceptor_base = 2 ** self.n_stations
        dispatcher_action = action // acceptor_base
        acceptor_code = action % acceptor_base
        
        acceptor_actions = []
        for i in range(self.n_stations):
            acceptor_actions.append(acceptor_code % 2)
            acceptor_code //= 2
        
        return dispatcher_action, acceptor_actions
    
    def encode_action(
        self, 
        dispatcher_action: int, 
        acceptor_actions: List[int]
    ) -> int:
        """
        Encode dispatcher and acceptor actions into joint action.
        
        Args:
            dispatcher_action: Dispatcher's action
            acceptor_actions: List of acceptor actions
            
        Returns:
            Joint action integer
        """
        acceptor_code = 0
        for i, a in enumerate(acceptor_actions):
            acceptor_code += a * (2 ** i)
        
        return dispatcher_action * (2 ** self.n_stations) + acceptor_code
    
    def _maybe_generate_job(self):
        """Generate a new job with some probability."""
        if not self.pending_job and self.np_random.random() < self.job_arrival_prob:
            self.pending_job = True
            self.pending_job_time = self.np_random.integers(
                self.processing_time_range[0], 
                self.processing_time_range[1] + 1
            )
            self.total_jobs_arrived += 1
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        obs = []
        
        # Normalized queue lengths
        for i in range(self.n_stations):
            obs.append(len(self.queues[i]) / self.queue_capacity)
        
        # Normalized processing loads
        for i in range(self.n_stations):
            obs.append(len(self.processing[i]) / self.max_processing)
        
        # Average remaining processing time (normalized)
        max_time = self.processing_time_range[1]
        for i in range(self.n_stations):
            if len(self.processing[i]) > 0:
                avg_remaining = np.mean(self.processing[i]) / max_time
            else:
                avg_remaining = 0.0
            obs.append(avg_remaining)
        
        # Pending job flag
        obs.append(1.0 if self.pending_job else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {
            "jobs_completed": self.jobs_completed,
            "jobs_rejected": self.jobs_rejected,
            "total_jobs_arrived": self.total_jobs_arrived,
            "step_count": self.step_count,
            "queue_lengths": [len(q) for q in self.queues],
            "processing_counts": [len(p) for p in self.processing],
            "pending_job": self.pending_job,
        }
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get the action mask for valid actions.
        
        Returns:
            Boolean array where True indicates a valid action
        """
        n_actions = (self.n_stations + 1) * (2 ** self.n_stations)
        mask = np.ones(n_actions, dtype=bool)
        
        if not self.pending_job:
            # If no pending job, only "reject" dispatcher actions make sense
            # But we allow all actions (they just won't do anything)
            pass
        else:
            # Mask out routing to full queues
            for disp_action in range(self.n_stations):
                if len(self.queues[disp_action]) >= self.queue_capacity:
                    # Mask all joint actions that route to this station
                    for acc_code in range(2 ** self.n_stations):
                        joint_action = disp_action * (2 ** self.n_stations) + acc_code
                        mask[joint_action] = False
        
        return mask
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_text(self):
        """Render as text output."""
        print(f"\n=== Step {self.step_count} ===")
        print(f"Pending Job: {'Yes' if self.pending_job else 'No'}")
        for i in range(self.n_stations):
            print(f"Station {i}: Queue={len(self.queues[i])}/{self.queue_capacity}, "
                  f"Processing={len(self.processing[i])}/{self.max_processing}")
        print(f"Completed: {self.jobs_completed}, Rejected: {self.jobs_rejected}")
    
    def _render_frame(self) -> np.ndarray:
        """Render as RGB array."""
        # Simple visualization (can be enhanced)
        width, height = 400, 200
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw stations
        station_width = width // (self.n_stations + 1)
        for i in range(self.n_stations):
            x = (i + 1) * station_width
            # Queue bar
            queue_height = int(150 * len(self.queues[i]) / self.queue_capacity)
            frame[150-queue_height:150, x-20:x+20] = [0, 0, 255]  # Blue
            # Processing bar
            proc_height = int(150 * len(self.processing[i]) / self.max_processing)
            frame[150-proc_height:150, x+25:x+45] = [0, 255, 0]  # Green
        
        return frame
    
    def close(self):
        """Clean up resources."""
        pass


class MultiAgentWarehouseEnv(WarehouseTaskAllocationEnv):
    """
    Multi-Agent version with separate action handling for each agent.
    
    This version is designed for CTDE (Centralized Training, Decentralized Execution)
    where agents take actions sequentially:
    1. Dispatcher decides routing
    2. Only the target station's acceptor decides acceptance
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Track which agent should act next
        self.current_agent = "dispatcher"
        self.dispatcher_decision = None
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_agent = "dispatcher"
        self.dispatcher_decision = None
        return obs, info
    
    def get_current_agent(self) -> str:
        """Get the name of the agent that should act next."""
        return self.current_agent
    
    def step_dispatcher(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute dispatcher action.
        
        Args:
            action: 0 to n_stations-1 for routing, n_stations for reject
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        
        if not self.pending_job:
            # No job to dispatch
            self._process_time_step()
            return self._get_observation(), reward, False, False, self._get_info()
        
        if action >= self.n_stations:
            # Dispatcher rejects
            reward -= 5.0
            self.jobs_rejected += 1
            self.pending_job = False
            self._process_time_step()
            self.current_agent = "dispatcher"
            return self._get_observation(), reward, False, False, self._get_info()
        
        # Store dispatcher decision and switch to acceptor
        self.dispatcher_decision = action
        self.current_agent = f"acceptor_{action}"
        
        return self._get_observation(), 0.0, False, False, self._get_info()
    
    def step_acceptor(self, station: int, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute acceptor action.
        
        Args:
            station: Which station's acceptor is acting
            action: 0 for reject, 1 for accept
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        
        if action == 1 and len(self.queues[station]) < self.queue_capacity:
            # Accept and enqueue
            self.queues[station].append(self.pending_job_time)
            reward += 1.0
        else:
            # Reject
            reward -= 2.0
            self.jobs_rejected += 1
        
        self.pending_job = False
        self._process_time_step()
        self.current_agent = "dispatcher"
        self.dispatcher_decision = None
        
        return self._get_observation(), reward, False, False, self._get_info()
    
    def _process_time_step(self):
        """Process one time step: job processing and new job arrival."""
        self.step_count += 1
        
        # Process jobs at each station
        for i in range(self.n_stations):
            # Move jobs from queue to processing
            while (len(self.processing[i]) < self.max_processing and 
                   len(self.queues[i]) > 0):
                self.processing[i].append(self.queues[i].pop(0))
            
            # Process jobs
            completed = []
            for j in range(len(self.processing[i])):
                self.processing[i][j] -= 1
                if self.processing[i][j] <= 0:
                    completed.append(j)
            
            # Remove completed
            for j in sorted(completed, reverse=True):
                self.processing[i].pop(j)
                self.jobs_completed += 1
        
        # Generate new job
        self._maybe_generate_job()


# Register environments with Gymnasium
gym.register(
    id="WarehouseTaskAllocation-v0",
    entry_point="src.environment.warehouse_env:WarehouseTaskAllocationEnv",
)

gym.register(
    id="MultiAgentWarehouse-v0",
    entry_point="src.environment.warehouse_env:MultiAgentWarehouseEnv",
)


if __name__ == "__main__":
    # Test the environment
    env = WarehouseTaskAllocationEnv(n_stations=2, render_mode="human")
    obs, info = env.reset()
    
    print("Initial observation:", obs)
    print("Action space:", env.joint_action_space)
    
    for _ in range(10):
        action = env.joint_action_space.sample()
        mask = env.get_action_mask()
        # Sample valid action
        valid_actions = np.where(mask)[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward:.2f}")
    
    env.close()
