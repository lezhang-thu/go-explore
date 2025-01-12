import os
import sys
import random

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from goexplore_py.sac.ppo_core import MLPActorCritic
from goexplore_py.sac.zeta_atari_env import make_env


class TransitionDataset(Dataset):

    def __init__(self, transitions):
        """
        Args:
            transitions (list): List of tuples (s0, a, r, s1, d).
        """
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        s0, a, r, s1, d = self.transitions[idx]
        return {
            's0': torch.tensor(s0, dtype=torch.float32),
            'a': torch.tensor(a, dtype=torch.long),
            'r': torch.tensor(r, dtype=torch.float32),
            's1': torch.tensor(s1, dtype=torch.float32),
            'd': torch.tensor(d, dtype=torch.bool)
        }


def sac(train_queue,
        val_env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(),
        gamma=0.99,
        polyak=0.995,
        lr=1e-4,
        batch_size=32,
        num_val_episodes=100):

    val_env = val_env_fn()
    obs_dim = val_env.observation_space.shape
    act_dim = val_env.action_space.n
    print('val_env act_dim: {}'.format(act_dim))

    # Create actor-critic module and target networks
    alpha = 1e-2
    ac_kwargs['alpha'] = alpha
    ac = actor_critic(376, act_dim, **ac_kwargs).cpu()
    ac_targ = deepcopy(ac)
    del ac_targ.conv
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for x in ac_targ.parameters():
        x.requires_grad = False
    full_opt = torch.optim.Adam(ac.parameters(), lr)

    def compute_pi_loss(batch):
        o = ac.conv(batch['s0'].cpu())
        _, logit_a, _ = ac.pi(o)
        action_samples = torch.swapaxes(
            torch.distributions.categorical.Categorical(logits=logit_a).sample(
                (1, )), 0, 1)

        with torch.no_grad():
            q1, adv1, _, _ = ac.Q_values(o, logit_a, True)
            q2, adv2, _, _ = ac.Q_values(o, logit_a, False)
            y_q1 = torch.gather(q1, -1, action_samples)
            y_q2 = torch.gather(q2, -1, action_samples)
            adv1 = torch.gather(adv1, -1, action_samples)
            adv2 = torch.gather(adv2, -1, action_samples)
            mask = y_q1 > y_q2
            adv = mask * adv2 + (~mask) * adv1

        logp_pi = F.log_softmax(logit_a, -1).gather(-1, action_samples)
        loss_pi = -(((adv - alpha * logp_pi).detach() * logp_pi).mean())
        return loss_pi

    def compute_offpolicy_loss(batch):
        o, a, r, o2, d = batch['s0'], batch['a'], batch['r'], batch[
            's1'], batch['d']
        r = r.cpu()
        d = d.cpu()
        a = a.cpu().unsqueeze(-1)

        o = ac.conv(o.cpu())
        with torch.no_grad():
            _, logit_a, _ = ac.pi(o)

        q1, _, v_loss1, _ = ac.Q_values(o, logit_a, True)
        q2, _, v_loss2, _ = ac.Q_values(o, logit_a, False)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            o2 = ac.conv(o2.cpu())
            a2, logit_a2, _ = ac.pi(o2)
            x = ac_targ.pi(o2)[1]
            a2 = a2.unsqueeze(-1)
            targ_q1 = ac_targ.Q_values(o2, x, True)[0]
            targ_q2 = ac_targ.Q_values(o2, x, False)[0]
            targ_q1 = torch.gather(targ_q1, -1, a2).squeeze(-1)
            targ_q2 = torch.gather(targ_q2, -1, a2).squeeze(-1)
            targ_q = torch.min(targ_q1, targ_q2)
            backup = r + gamma * (1 - d.float()) * (-alpha * F.log_softmax(
                logit_a2, -1).gather(-1, a2).squeeze(-1) + targ_q)

        x_q1 = torch.gather(q1, -1, a).squeeze(-1)
        x_q2 = torch.gather(q2, -1, a).squeeze(-1)
        return F.huber_loss(x_q1, backup) + F.huber_loss(
            x_q2, backup) + v_loss1 + v_loss2

    def update(x):
        full_opt.zero_grad()
        (compute_offpolicy_loss(x) + compute_pi_loss(x)).backward()
        full_opt.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for x, x_targ in zip([ac.v1, ac.adv1, ac.v2, ac.adv2, ac.pi], [
                    ac_targ.v1, ac_targ.adv1, ac_targ.v2, ac_targ.adv2,
                    ac_targ.pi
            ]):
                for p, p_targ in zip(x.parameters(), x_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    @torch.no_grad()
    def get_action(o, deterministic=False):
        o = torch.from_numpy(np.asarray(o)).unsqueeze(0).cpu()
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def val_agent(m):
        ep_ret, ep_len = 0, 0
        for j in range(m):
            o, d = val_env.reset(), False
            while not d:
                # Take deterministic actions at test time
                o, r, d, _ = val_env.step(get_action(o, True)[0])
                ep_ret += r
                ep_len += 1
        return ep_ret / m, ep_len / m

    training_steps = 0
    while True:
        transitions = []
        num = 0
        #while not train_queue.empty() and num < int(100e3):
        while num < int(256):
            transitions.append(train_queue.get())
            num += 1
        if num == 0: break
        dataset = TransitionDataset(transitions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print('sac training_steps: {}'.format(training_steps))
        for x in dataloader:
            update(x)
            training_steps += 1
            #if training_steps % int(1e6):
            if training_steps % int(1e4) == 0:
                val_ep_ret, val_ep_len = val_agent(10)
                print('test score@training_steps {}: {}, ep_len: {}'.format(
                    training_steps, val_ep_ret, val_ep_len))


def main(env, train_queue):
    env = '{}NoFrameskip-v4'.format(env)
    hid = 256
    l = 2
    gamma = 0.99
    y = lambda: make_env(env,
                         wrapper_kwargs={
                             'frame_stack': True,
                             'clip_rewards': False,
                             'episode_life': False,
                         })
    sac(
        train_queue,
        y,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[hid] * l),
        gamma=gamma,
    )
