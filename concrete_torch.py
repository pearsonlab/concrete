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


class HMM(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, temperature):
        super(HMM, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        # self.gen = Transition_gen(z_dim, hidden_dim)
        self.unc_A = nn.Parameter(torch.randn(z_dim, z_dim))
        self.rec = Transition_rec(x_dim, z_dim, hidden_dim)

        self.z0_logits_gen = nn.Parameter(torch.randn(z_dim))
        self.z0_logits_rec = nn.Parameter(torch.randn(z_dim))

        self.loc = nn.Parameter(torch.randn(z_dim, x_dim))
        # self.loc = torch.tensor([[5., -5.], [-5., 5.]])
        self.vec_scale_tril = nn.Parameter(
            torch.eye(x_dim)[np.tril_indices(x_dim)].repeat(z_dim, 1))
        # self.cov = torch.tensor(
        #     [[[4., 3.6], [3.6, 4.]], [[4., 3.6], [3.6, 4.]]])

        self.temp = temperature
        self.softmax = nn.Softmax(dim=0)
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.softplus = nn.Softplus()

    def model(self, obs):
        T = obs.size(0)
        self.gen_dist = []

        pyro.module("hmm", self)

        z_prev = self.softmax(self.z0_logits_gen)

        # self.A = self.softmax(self.unc_A)
        self.A = self.softplus(self.unc_A)

        scale_tril = torch.zeros(self.z_dim, self.x_dim, self.x_dim)
        for i in range(self.z_dim):
            scale_tril[i][np.tril_indices(self.x_dim)] = self.vec_scale_tril[i]
        self.cov = scale_tril.matmul(scale_tril.transpose(1, 2))

        for t in range(1, T + 1):
            # logits_t = self.gen(z_prev)
            # z_t = pyro.sample("z_{}".format(t), dist.RelaxedOneHotCategorical(
            #     self.temp, logits=logits_t))
            # probs_t = self.A.matmul(z_prev.view(self.z_dim, 1)).view(self.z_dim)
            # z_t = pyro.sample("z_{}".format(t), dist.RelaxedOneHotCategorical(
            #     self.temp, probs=probs_t))
            logits_t = torch.log(self.A.matmul(z_prev.view(self.z_dim, 1)).view(self.z_dim))
            dist_t = dist.RelaxedOneHotCategorical(self.temp, logits=logits_t)
            self.gen_dist.append(dist_t)
            z_t = pyro.sample("z_{}".format(t), dist_t)

            loc_t = torch.sum(self.loc * z_t.view(self.z_dim, 1), 0)
            cov_t = torch.sum(self.cov * z_t.pow(2).view(self.z_dim, 1, 1), 0)
            pyro.sample("x_{}".format(t), dist.MultivariateNormal(
                loc_t, covariance_matrix=cov_t), obs=obs[t - 1])

            z_prev = z_t

    def guide(self, obs):
        T = obs.size(0)
        self.rec_dist = []
        self.z_q = torch.empty(T, self.z_dim)

        pyro.module("hmm", self)

        z_prev = self.softmax(self.z0_logits_rec)
        x_prev = obs[0]

        for t in range(1, T + 1):
            logits_t = self.rec(x_prev, obs[t - 1], z_prev)
            dist_t = dist.RelaxedOneHotCategorical(self.temp, logits=logits_t)
            self.rec_dist.append(dist_t)
            z_t = pyro.sample("z_{}".format(t), dist_t)
            self.z_q[t - 1] = z_t

            z_prev = z_t
            x_prev = obs[t - 1]

    # def elbo(self, obs):
    #     T = obs.size(0)
    #     elbo = torch.zeros([])

    #     for t in range(T):
    #         elbo += self.gen_dist[t].log_prob(self.z_q[t])
    #         elbo += dist.MultivariateNormal(
    #             torch.sum(self.loc * self.z_q[t].view(self.z_dim, 1), 0),
    #             torch.sum(self.cov * self.z_q[t].pow(2).view(self.z_dim, 1, 1), 0)).log_prob(obs[t])
    #         elbo -= self.rec_dist[t].log_prob(self.z_q[t])

    #     return elbo

    # def klqp(self):
    #     T = self.z_q.size(0)
    #     kl = torch.zeros([])

    #     for t in range(T):
    #         kl += self.gen_dist[t].log_prob(self.z_q[t])
    #         kl -= self.rec_dist[t].log_prob(self.z_q[t])

    #     return kl

    def log_prob(self):
        T = self.z_q.size(0)
        gen_log_probs = torch.empty(T)
        rec_log_probs = torch.empty(T)

        for t in range(T):
            gen_log_probs[t] = self.gen_dist[t].log_prob(self.z_q[t])
            rec_log_probs[t] = self.rec_dist[t].log_prob(self.z_q[t])

        return torch.stack([gen_log_probs, rec_log_probs], -1)

    # def klqp_A(self):
    #     A = torch.tensor([[.9, .1], [.1, .9]])
    #     T = self.z_q.size(0)
    #     kl = torch.zeros([])

    #     z_prev = self.softmax(self.z0_logits_rec)
    #     for t in range(T):
    #         logits_t = torch.log(A.matmul(z_prev.view(self.z_dim, 1)).view(self.z_dim))
    #         dist_t = dist.RelaxedOneHotCategorical(self.temp, logits=logits_t)
    #         kl += dist_t.log_prob(self.z_q[t])
    #         kl -= self.rec_dist[t].log_prob(self.z_q[t])
    #         z_prev = self.z_q[t]

    #     return kl

    # def log_lik(self, obs):
    #     T = obs.size(0)
    #     log_lik = torch.zeros([])

    #     for t in range(T):
    #         log_lik += dist.MultivariateNormal(
    #             torch.sum(self.loc * self.z_q[t].view(self.z_dim, 1), 0),
    #             torch.sum(self.cov * self.z_q[t].pow(2).view(self.z_dim, 1, 1), 0)).log_prob(obs[t])

    #     return log_lik

    # def elbo_A(self, obs):
    #     # A = torch.tensor([[.9, .1], [.1, .9]]) * 5.
    #     z_prev = self.softmax(self.z0_logits_rec)
    #     T = obs.size(0)
    #     elbo = torch.zeros([])

    #     for t in range(T):
    #         logits_t = torch.log(self.A.matmul(z_prev.view(self.z_dim, 1)).view(self.z_dim))
    #         dist_t = dist.RelaxedOneHotCategorical(self.temp, logits=logits_t)
    #         elbo += dist_t.log_prob(self.z_q[t])
    #         z_t = pyro.sample("z_{}".format(t), dist_t)
    #         z_prev = self.z_q[t]

    #         elbo += dist.MultivariateNormal(
    #             torch.sum(self.loc * z_t.view(self.z_dim, 1), 0),
    #             torch.sum(self.cov * z_t.pow(2).view(self.z_dim, 1, 1), 0)).log_prob(obs[t])

    #         elbo -= self.rec_dist[t].log_prob(self.z_q[t])

    #     return elbo


def main(args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = SummaryWriter(log_dir=args.model_dir + "/log")

    temperature = torch.tensor(args.temperature)
    hmm = HMM(x_dim=args.obs_dim, z_dim=args.state_dim,
              hidden_dim=args.hidden_dim, temperature=temperature)
    data = torch.from_numpy(np.load(args.data_dir))

    optimizer = Adam({"lr": args.learning_rate})
    svi = SVI(hmm.model, hmm.guide, optimizer, loss=Trace_ELBO(num_particles=10))

    # for name, param in hmm.named_parameters():
    #     print(name, param)

    for epoch in range(1, args.num_epochs + 1):
        hmm.temp = torch.max(
            args.temperature * torch.pow(
                torch.tensor(0.9), torch.tensor(np.floor(epoch / 100))),
            torch.tensor(0.3))

        loss = svi.step(obs=data)

        writer.add_scalar('elbo', -loss, epoch)
         # writer.add_scalars(
        #     'elbo', {'Pyro': -loss, 'true_A': hmm.elbo_A(data)}, epoch)
        # writer.add_scalar('klqp', hmm.klqp(), epoch)
        # writer.add_scalars(
        #     'klqp', {'klqp': hmm.klqp(), 'true_A': hmm.klqp_A()}, epoch)
        # writer.add_scalar('log_likelihood', hmm.log_lik(data), epoch)
        writer.add_scalar('temperature', hmm.temp, epoch)
        writer.add_scalars(
            'A', {'A_00': hmm.A[0, 0], 'A_10': hmm.A[1, 0],
                  'A_01': hmm.A[0, 1], 'A_11': hmm.A[1, 1]}, epoch)
        for d in range(args.state_dim):
            writer.add_scalars(
                'component_{}/mean'.format(d + 1),
                {'dim_1': hmm.loc[d, 0], 'dim_2': hmm.loc[d, 1]}, epoch)
            writer.add_scalars(
                'component_{}/covariance'.format(d + 1),
                {'var_1': hmm.cov[d, 0, 0], 'var_2': hmm.cov[d, 1, 1],
                 'cov_12': hmm.cov[d, 0, 1]}, epoch)

        if epoch % 100 == 0:
            print("loss after {0} epochs: {1}".format(epoch, loss))
            # print("mean:\n{}".format(hmm.loc.detach().numpy()))
            # print("covariance:\n{}".format(hmm.cov.detach().numpy()))
            np.savez(args.model_dir + "/epoch_{}".format(epoch),
                     z0_logits=hmm.z0_logits_gen.detach().numpy(),
                     A=hmm.A.detach().numpy(),
                     loc=hmm.loc.detach().numpy(),
                     cov=hmm.cov.detach().numpy(),
                     z_q=hmm.z_q.detach().numpy(),
                     log_probs=hmm.log_prob().detach().numpy(),
                     temp=hmm.temp.detach().numpy())

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse_arguments")
    parser.add_argument('-x_dim', '--obs_dim', type=int, default=2)
    parser.add_argument('-z_dim', '--state_dim', type=int, default=2)
    parser.add_argument('-h_dim', '--hidden_dim', type=int, default=32)
    parser.add_argument('-t', '--temperature', type=float, default=0.5)
    parser.add_argument('-n', '--num-epochs', type=int, default=4000)
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.05)
    args = parser.parse_args()

    main(args)
