import os
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

## Replay Buffer ##
# Store transitions (state, action, reward, new_state, done)
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


## Critic Network ##
# Estimates the expected return of a state-action pair Q(s,a)
# Provides feedback to the Actor on which actions are expected to yield higher returns.
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dim, n_actions, fc1_dims=256, fc2_dims=256, name='critic', checkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(checkpt_dir, name + '_sac')

        self.fc1 = nn.Linear(input_dim + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

## Value Network ##
# Estimates the expected return of being in a state V(s)
# Needed to compute the policy's expected value, crucial for the entropy-augmented objective function used in SAC
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dim, fc1_dims=256, fc2_dims=256, name='value', checkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.checkpoint_file = os.path.join(checkpt_dir, name + '_sac')

        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

## Actor Network ##
# Represents the policy function, which maps states to a probability distribution over actions
# During training, the Actor generates actions according to its policy, which are then evaluated by the Critic network
# The goal of the Actor is to improve its policy based on feedback from the Critic to maximize the expected return
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dim, max_action, fc1_dims=256, fc2_dims=256, n_actions=1, name='actor', checkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(checkpt_dir, name + '_sac')
        self.max_action = T.tensor(max_action)
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    #     self.init_weights()
    #
    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.fc1.weight)
    #     nn.init.xavier_uniform_(self.fc2.weight)
    #     nn.init.xavier_uniform_(self.mu.weight)
    #     nn.init.xavier_uniform_(self.sigma.weight)

    def forward(self, state):
        state = state.float()
        prob = self.fc1(state)
        if T.any(T.isnan(prob)):
            print(state)
            print(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparametrize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparametrize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # Split the actions
        action_1 = actions[:, 0]
        action_2 = actions[:, 1]
        action_3 = actions[:, 2]

        # Scale actions to their respective ranges
        action_1 = T.sigmoid(action_1) * self.max_action[0]
        action_2 = T.sigmoid(action_2) * self.max_action[1]
        action_3 = T.tanh(action_3) * self.max_action[2]
        action = T.cat((action_1.unsqueeze(1), action_2.unsqueeze(1), action_3.unsqueeze(1)), dim=1)

        # Compute log probabilities for each action
        log_probs = probabilities.log_prob(actions)
        # Adjust log_probs for the sigmoid actions
        log_probs[:, 0] -= T.log(action_1 / self.max_action[0] * (1 - action_1 / self.max_action[0]) + self.reparam_noise)
        log_probs[:, 1] -= T.log(action_2 / self.max_action[1] * (1 - action_2 / self.max_action[1]) + self.reparam_noise)
        # Adjust log_probs for the tanh action
        log_probs[:, 2] -= T.log(1 - (action_3 / self.max_action[2]).pow(2) + self.reparam_noise)

        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

## Agent Class ##
# Includes the Critic, Actor and ValueNetwork and implements the learn and
class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, tau=0.005, input_dim=3, env=None, gamma=0.99, n_actions=1, max_size=1_000_000,
                 layer1_size=256, layer2_size=256, batch_size=512, reward_scale=2):
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_dim, n_actions)
        self.actor = ActorNetwork(alpha, input_dim, n_actions=n_actions, name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dim, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dim, n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(beta, input_dim, name='value')
        self.target_value = ValueNetwork(beta, input_dim, name='target_value')
        self.scale = reward_scale

        self.update_network_parameters(tau=1)

    def choose_action(self, observation_array):
        if len(observation_array.shape) > 1:
            actions_array = []
            for observation in observation_array:
                state = T.tensor(np.array([observation])).to(self.actor.device)
                actions, _ = self.actor.sample_normal(state, reparametrize=False)
                actions = actions.cpu().detach().numpy()[0]
                actions_array.append(actions)
            return np.array(actions_array)
        else:
            observation = observation_array
            state = T.tensor(np.array([observation])).to(self.actor.device)
            actions, _ = self.actor.sample_normal(state, reparametrize=False)
            return actions.cpu().detach().numpy()[0]
    def rememeber(self, state_array, action_array, reward_array, new_state_array, done_array):
        for state, action, reward, new_state, done in zip(state_array, action_array, reward_array, new_state_array, done_array):
            self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """
        This functions performs a soft-update on the parameters of the target_value network
        using the parameters of the value network. This process helps stabilize training by
        smoothing the updates to the target network, preventing it from changing too quickly,
        which is crucial for the stability and performance of SAC.
        """
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample batch of memory from memory buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Convert into tensors
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        # Estimate values of current and next states
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        # Value Network
        # Sample actions
        actions, log_probs = self.actor.sample_normal(state, reparametrize=False)
        log_probs = log_probs.view(-1)
        # Evaluation of actions using critic network
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        # Take element-wise min to be more conservative and address overestimation bias
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        # Subtract entropy to encourage exploration
        value_target = critic_value - log_probs
        # Compute loss and backprop
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Actor Network
        # Sample actions, reparametrize=True to allow backprop through sampling action
        actions, log_probs = self.actor.sample_normal(state, reparametrize=True)
        log_probs = log_probs.view(-1)
        # Evaluation of actions with 2 critic networks to avoid overestimation bias
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Encourage actions with high Q-values, penalize actions with high entropy
        actor_loss = log_probs - critic_value
        # Average across all values
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Critic Network
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # Computation of TD-error
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Soft update to the value networks
        self.update_network_parameters()
