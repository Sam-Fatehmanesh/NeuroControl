import torch
import random  # This import was missing in the original code

class ReplayBuffer:
    """
    A circular buffer to store and sample experiences for reinforcement learning.
    """

    def __init__(self, capacity):
        """
        Initialize the ReplayBuffer.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.capacity = capacity
        self.buffer = []  # List to store experiences
        self.position = 0  # Current position in the buffer

    def push(self, state, action, reward, next_state):
        """
        Add a new experience to the buffer.

        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The resulting state after taking the action
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity  # Circular buffer implementation

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: Batches of states, actions, rewards, and next_states as PyTorch tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(torch.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: The number of experiences currently in the buffer.
        """
        return len(self.buffer)
