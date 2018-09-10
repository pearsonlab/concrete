import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO
from pyro.optim import Adam, Adagrad, RMSprop

from tensorboardX import SummaryWriter


class Transition_gen(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Transition_gen, self).__init__()
        self.l1 = nn.Linear(z_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t_1):
        h1 = self.relu(self.l1(z_t_1))
        h2 = self.relu(self.l2(h1))
        logits_t = self.l3(h2)

        return logits_t


class Transition_rec(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super(Transition_rec, self).__init__()
        self.l1 = nn.Linear(z_dim + x_dim * 2, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x_t_1, x_t, z_t_1):
        h1_q = self.relu(self.l1(torch.cat((x_t_1, x_t, z_t_1), 0)))
        h2_q = self.relu(self.l2(h1_q))
        logits_t_q = self.l3(h2_q)

        return logits_t_q


class Conditional_gen(nn.Module):
    def __init__(self, z1_dim, z2_dim, hidden_dim):
        super(Conditional_gen, self).__init__()
        self.l1 = nn.Linear(z1_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, z2_dim)
        self.relu = nn.ReLU()

    def forward(self, z1):
        r1 = self.relu(self.l1(z1))
        r2 = self.relu(self.l2(r1))
        logits_z2 = self.l3(r2)

        return logits_z2


class Exit_gen(nn.Module):
    def __init__(self, z1_dim, z2_dim, hidden_dim):
        super(Exit_gen, self).__init__()
        self.l1 = nn.Linear(z1_dim + z2_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z1, z2=None):
        if z2 is None:
            e1 = self.relu(self.l1(z1))
        else:
            e1 = self.relu(self.l1(torch.cat((z1, z2), 0)))
        e2 = self.relu(self.l2(h1))
        e = self.softmax(self.l3(h2))

        return e
