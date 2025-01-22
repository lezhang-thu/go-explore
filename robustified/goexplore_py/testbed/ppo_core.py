__all__ = ['MLPActorCritic']

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActor(nn.Module):

    def __init__(self, obs_dim, num_actions, hidden_sizes, activation):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                 nn.Linear(512, num_actions))
        #self.im = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
        #                        nn.Linear(512, num_actions))
        #self.gate = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
        #                          nn.Linear(512, 2))

    def forward(self, obs, deterministic=False, with_logprob=True):
        #expert_probs = torch.stack(
        #    [F.softmax(expert(obs), -1) for expert in (self.net, self.im)],
        #    dim=1)
        #gate_probs = F.softmax(self.gate(obs), dim=-1)
        #logit = torch.log(
        #    torch.sum(gate_probs.unsqueeze(-1) * expert_probs, dim=1))

        logit = self.net(obs)
        if deterministic:
            actions = logit.argmax(1)
        else:
            actions = torch.distributions.categorical.Categorical(
                logits=logit).sample()

        return actions, logit, None


class MLPActorCritic(nn.Module):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 alpha=1e-2):
        super().__init__()

        self.conv = ConvOnly(4, obs_dim)
        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.v1 = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                nn.Linear(512, 1))
        self.adv1 = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                  nn.Linear(512, act_dim))
        self.v2 = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                nn.Linear(512, 1))
        self.adv2 = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                  nn.Linear(512, act_dim))
        self._alpha = alpha

    def Q_values(self, obs, logit, first_flag=True):
        if first_flag:
            v_function = self.v1
            adv_function = self.adv1
        else:
            v_function = self.v2
            adv_function = self.adv2

        v = v_function(obs)
        adv = adv_function(obs)
        x = torch.distributions.categorical.Categorical(logits=logit)
        baseline = (adv * F.softmax(logit, -1).detach()).sum(
            -1, True) + self._alpha * x.entropy().unsqueeze(-1).detach()
        adv = adv - baseline
        q = v + adv
        v_loss = F.huber_loss(v, baseline)

        return q, adv, v_loss, v

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs = self.conv(obs)
            a, logit, _ = self.pi(obs, deterministic, False)
            return a.squeeze(0).detach().cpu().numpy(), F.softmax(
                logit, -1).gather(
                    -1,
                    a.unsqueeze(-1),
                ).squeeze().detach().cpu().numpy()


class ConvOnly(nn.Module):

    def __init__(self, frame_stack, obs_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, obs):
        obs = obs.to(torch.float32) / 255.0
        # (B, C, H, W)
        obs = obs.permute(0, 3, 1, 2)
        return self.cnn(obs)
