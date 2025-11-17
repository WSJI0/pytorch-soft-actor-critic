import random
import numpy as np
import os
import pickle

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Memory
    """
    def __init__(self, capacity, seed, alpha=0.6, beta_start=0.4, beta_frames=100000):
        random.seed(seed)
        np.random.seed(seed)
        
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Initial importance sampling weight
        self.beta_frames = beta_frames  # Frames over which beta is annealed to 1
        self.frame = 1  # Current frame number
        
        self.tree = SumTree(capacity)
        self.epsilon = 1e-6  # Small constant to prevent zero priorities
        self.max_priority = 1.0

    def _get_beta(self):
        """Get current beta value with annealing"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        """Add experience to memory with maximum priority"""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """Sample batch with priorities and return experiences, indices, and weights"""
        batch = []
        indices = []
        weights = []
        priorities = []
        
        beta = self._get_beta()
        self.frame += 1
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, experience = self.tree.get(s)
            
            batch.append(experience)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = np.power(len(self.tree.data) * sampling_probs, -beta)
        weights /= weights.max()  # Normalize weights
        
        # Convert batch to arrays
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (state, action, reward, next_state, done), indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities of experiences"""
        for idx, priority in zip(indices, priorities):
            # Add small epsilon and apply alpha
            priority = (np.abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/per_buffer_{}_{}".format(env_name, suffix)
        print('Saving PER buffer to {}'.format(save_path))

        buffer_data = {
            'tree_data': self.tree.data,
            'tree_tree': self.tree.tree,
            'write': self.tree.write,
            'n_entries': self.tree.n_entries,
            'max_priority': self.max_priority,
            'frame': self.frame
        }

        with open(save_path, 'wb') as f:
            pickle.dump(buffer_data, f)

    def load_buffer(self, save_path):
        print('Loading PER buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            buffer_data = pickle.load(f)
            
        self.tree.data = buffer_data['tree_data']
        self.tree.tree = buffer_data['tree_tree']
        self.tree.write = buffer_data['write']
        self.tree.n_entries = buffer_data['n_entries']
        self.max_priority = buffer_data['max_priority']
        self.frame = buffer_data['frame']
