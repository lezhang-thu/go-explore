import numpy as np
import random

import torch


class ReplayBuffer(object):

    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, pi_old, adv):
        data = (obs_t, action, reward, obs_tp1, done, pi_old, adv)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    @torch.no_grad()
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, pi_olds, advs = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, pi_old, adv = data

            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
            pi_olds.append(pi_old)
            advs.append(adv)

        batch = dict(obs=np.asarray(obses_t),
                     obs2=np.asarray(obses_tp1),
                     rew=np.clip(np.asarray(rewards), -1, 1),
                     done=np.asarray(dones),
                     pi_old=np.asarray(pi_olds),
                     adv=np.asarray(advs))
        x = {
            k: torch.as_tensor(v, dtype=torch.float32).cuda()
            for k, v in batch.items()
        }
        x['act'] = torch.as_tensor(np.asarray(actions), dtype=torch.int64)
        del obses_t[:]
        del obses_tp1[:]
        del batch
        return x

    def sample(self, batch_size, recent=False):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        x = 0
        if recent:
            x = max(len(self._storage) - int(1e4), 0)
        idxes = [
            random.randint(x,
                           len(self._storage) - 1) for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)
