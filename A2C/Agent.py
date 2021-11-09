import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims,1)

        self.optimizer = optim.RMSprop( self.parameters(),lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)


class Agent():
    def __init__(self,lr, input_dims, n_actions, fc1_dims, fc2_dims,
                 gamma = 0.99):
        self.gamma = gamma
        self.lr =lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.net = ActorCriticNetwork(lr, input_dims, n_actions,
                                      fc1_dims=fc1_dims, fc2_dims = fc2_dims)

        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype = T.float).to(self.net.device)
        probabilities, _ = self.net.forward(state)
        probabilities = F.softmax(probabilities, dim =1)
        actions_probs = T.distributions.Categorical(probabilities)
        action = actions_probs.sample()
        log_prob = actions_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.net.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.net.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.net.device)
        reward = T.tensor([reward], dtype=T.float).to(self.net.device)

        _, critic_value = self.net.forward(state)
        _, critic_value_ = self.net.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_los = -self.log_prob*delta
        critic_loss = delta**2

        (actor_los + critic_loss).backward()
        self.net.optimizer.step()

