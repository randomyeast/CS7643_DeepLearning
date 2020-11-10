import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        H,W,C = env.observation_space.shape 
        
        self.conv1 = nn.Conv2d(C*config.state_history,16,kernel_size = 8,stride = 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.relu2 = nn.ReLU()
        
        h = (H-8)//4 + 1
        out = (h-4)//2 + 1
        self.fc = nn.Linear(out*out*32,env.action_space.n)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        batch_size = len(state)

        state = state.transpose(1,3)
        
        h = self.conv1(state)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = h.reshape(batch_size,-1)
        h = self.fc(h)

        return h
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################
