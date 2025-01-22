import os
import sys

sys.path.insert(
    0,
    '/home/ubuntu/lezhang.thu/sac-go-explore/im-go-explore/go-explore/robustified'
)

import time
import random
import typing
import logging
import shutil
from copy import deepcopy
import itertools
from collections import deque

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from goexplore_py.testbed.ppo_core import MLPActorCritic

#from zeta_atari_env import make_env
from goexplore_py.testbed.generic_atari_env import MyAtari
from goexplore_py.testbed.replay_buffer_recent import ReplayBuffer


class ExpertReplay:

    def __init__(self):
        self.episodes = []
        self.best_10 = []
        self.m = 0
        self.significant_thres = 1.0

    def append(self, episode: dict) -> None:
        assert 'return' in episode
        assert 'exp' in episode
        assert 'timestamp' in episode

        ## -1 for default unset state
        #episode['life'] = -1
        #self.m += len(episode['exp'])
        #self.episodes.append(episode)
        self.best_10.append(episode)
        self._filter()

    def _filter(self) -> None:
        # best 10
        x = [(episode['timestamp'], episode['return'])
             for episode in self.best_10]
        x.sort(key=lambda v: (v[1], v[0]), reverse=True)
        kept = set([_[0] for _ in x[:10]])
        #print(x)
        #print(kept)
        #exit(0)

        counter = 0
        y = []
        #self.m = 0
        for episode in self.best_10:
            if episode['timestamp'] in kept:
                y.append(episode)
                #self.m += len(episode['exp'])
                counter += 1
                if counter > 10:
                    break
        self.best_10 = y

    def collate(self, scores_recent: list) -> None:
        y = []
        self.m = 0

        x_stamp = [_[0] for _ in scores_recent]
        for episode in self.best_10:
            x_score = []
            flag_in = False
            for j, t in enumerate(x_stamp):
                if t != episode['timestamp']:
                    #if True:
                    x_score.append(scores_recent[j][1])
                else:
                    assert episode['return'] == scores_recent[j][1]
                    flag_in = True
            if not flag_in:
                x_score = x_score[1:]

            outlier = stats.zscore(
                np.asarray(x_score + [episode['return']],
                           dtype=np.float32))[-1]
            #if outlier > self.significant_thres or episode['life'] > 0:
            if outlier > self.significant_thres:
                #if True:
                #if episode['life'] > 0:
                #    episode['life'] = max(episode['life'] - 1, 0)
                ## -1
                #else:
                #    episode['life'] = 5
                y.append(episode)
                self.m += len(episode['exp'])
        self.episodes = y

    def sample(self, batch_size: int) -> tuple:
        # debug - start
        #if random.uniform(0, 1) < 1e-2:
        #    for x in self.episodes:
        #        print('{}: {}'.format(x['timestamp'], x['return']), end='**')
        #    print()

        idxes = [random.randint(0, self.m - 1) for _ in range(batch_size)]
        idxes.sort()
        states, actions, rewards, states2, dones, pi_olds = [], [], [], [], [], []
        y = [states, actions, rewards, states2, dones, pi_olds]

        x = iter(self.episodes)
        start_episode = next(x)['exp']
        start_idx = 0
        for k in idxes:
            while k >= start_idx + len(start_episode):
                start_idx += len(start_episode)
                start_episode = next(x)['exp']
            for j in range(6):
                y[j].append(start_episode[k - start_idx][j])

        keys = ('obs', 'act', 'rew', 'obs2', 'done', 'pi_old')
        dtypes = (np.float32, np.int64, np.float32, np.float32, np.float32,
                  np.float32)
        return {
            key:
            torch.as_tensor(np.asarray(_, dtype=dtype),
                            device=torch.device('cuda'))
            for (key, dtype, _) in zip(keys, dtypes, y)
        }


def setup_logging(save_dir, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    #fh = logging.FileHandler(os.path.join(save_dir, '{}'.format(logger_name)))
    #fh.setLevel(logging.INFO)
    #fh.setFormatter(formatter)
    #logger.addHandler(fh)
    return logger


def sac(  #env_fn,
        #val_env_fn,
        communicate_queue,
        env_name,
        actor_critic=MLPActorCritic,
        #ac_kwargs=dict(),
        ac_kwargs=dict(hidden_sizes=[256] * 2),
        seed=0,
        steps_per_epoch=int(1e4),
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        lr=1e-4,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        num_val_episodes=100):

    #env = env_fn()
    #val_env = val_env_fn()
    env = MyAtari(env_name)
    val_env = MyAtari(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    print('act_dim: {}'.format(act_dim))

    # Create actor-critic module and target networks
    alpha = 1e-2
    ac_kwargs['alpha'] = alpha
    ac = actor_critic(376, act_dim, **ac_kwargs).cuda()
    ac_targ = deepcopy(ac)
    del ac_targ.conv
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for x in ac_targ.parameters():
        x.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(replay_size)
    output_threshold = 1e-4

    # debug - start
    expert_replay = ExpertReplay()
    history_len = 10
    full_opt = torch.optim.Adam(ac.parameters(), lr)
    # go-explore - start
    go_explore_expert_replay = ExpertReplay()

    # go-explore - end

    def compute_pi_loss(data):
        o = ac.conv(data['obs'].cuda())
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

    def compute_offpolicy_loss(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data[
            'obs2'], data['done']
        r = r.cuda()
        d = d.cuda()
        a = a.cuda().unsqueeze(-1)

        o = ac.conv(o.cuda())
        with torch.no_grad():
            _, logit_a, _ = ac.pi(o)

        q1, _, v_loss1, _ = ac.Q_values(o, logit_a, True)
        q2, _, v_loss2, _ = ac.Q_values(o, logit_a, False)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            o2 = ac.conv(o2.cuda())
            a2, logit_a2, _ = ac.pi(o2)
            x = ac_targ.pi(o2)[1]
            a2 = a2.unsqueeze(-1)
            targ_q1 = ac_targ.Q_values(o2, x, True)[0]
            targ_q2 = ac_targ.Q_values(o2, x, False)[0]
            targ_q1 = torch.gather(targ_q1, -1, a2).squeeze(-1)
            targ_q2 = torch.gather(targ_q2, -1, a2).squeeze(-1)
            targ_q = torch.min(targ_q1, targ_q2)
            backup = r + gamma * (1 - d) * (-alpha * F.log_softmax(
                logit_a2, -1).gather(-1, a2).squeeze(-1) + targ_q)

        x_q1 = torch.gather(q1, -1, a).squeeze(-1)
        x_q2 = torch.gather(q2, -1, a).squeeze(-1)
        return F.huber_loss(x_q1, backup) + F.huber_loss(
            x_q2, backup) + v_loss1 + v_loss2

    #def im_loss(data):
    #    a = data['act'].cuda().unsqueeze(-1)
    #    pi_old = data['pi_old'].cuda().unsqueeze(-1)

    #    o = ac.conv(data['obs'].cuda())
    #    logit_a = ac.pi(o)[1]

    #    with torch.no_grad():
    #        q1, adv1, _, _ = ac.Q_values(o, logit_a, True)
    #        q2, adv2, _, _ = ac.Q_values(o, logit_a, False)
    #        y_q1 = torch.gather(q1, -1, a)
    #        y_q2 = torch.gather(q2, -1, a)
    #        adv1 = torch.gather(adv1, -1, a)
    #        adv2 = torch.gather(adv2, -1, a)
    #        mask = y_q1 > y_q2
    #        adv = mask * adv2 + (~mask) * adv1

    #    ratio = F.softmax(logit_a, -1).gather(-1, a) / pi_old
    #    epsilon_low = 0.2
    #    epsilon_high = 0.2

    #    logp_pi = F.log_softmax(logit_a, -1).gather(-1, a)
    #    x = (adv - alpha * logp_pi).detach()
    #    # debug - start
    #    spec = (ratio < 1 - epsilon_low) & (x < 0)
    #    x[spec] = torch.abs(x).mean()

    #    loss_ppo = torch.min(
    #        ratio * x,
    #        torch.clip(ratio, 1 - epsilon_low, 1 + epsilon_high) * x).mean()

    #    loss_ppo = -loss_ppo
    #    return loss_ppo

    def imitation_game(data):
        a = data['act'].cuda().unsqueeze(-1)
        o = ac.conv(data['obs'].cuda())
        logit_a = ac.pi(o)[1]
        #logit_a = ac.pi.im(o)
        #gate = F.log_softmax(ac.pi.gate(o), -1)[:, 1]
        logp_pi = F.log_softmax(logit_a, -1).gather(-1, a)
        return -(logp_pi.mean()) #- (gate.mean())

    def update():
        x = replay_buffer.sample(batch_size)
        full_opt.zero_grad()
        compute_offpolicy_loss(x).backward()
        full_opt.step()

        x = replay_buffer.sample(batch_size, True)
        full_opt.zero_grad()
        compute_pi_loss(x).backward()
        full_opt.step()

        full_opt.zero_grad()
        compute_offpolicy_loss(x).backward()
        full_opt.step()

        #if False:
        #    #if expert_replay.m >= batch_size:  # and random.uniform(0, 1) < 0.25:
        #    x = expert_replay.sample(batch_size)
        #    full_opt.zero_grad()
        #    #im_loss(x).backward()
        #    imitation_game(x).backward()
        #    full_opt.step()

        #    #full_opt.zero_grad()
        #    #compute_offpolicy_loss(x).backward()
        #    #full_opt.step()

        #    #full_opt.zero_grad()
        #    #compute_offpolicy_loss(expert_replay.sample(batch_size)).backward()
        #    #full_opt.step()
        if go_explore_expert_replay.m >= batch_size:  # and random.uniform(0, 1) < 0.25:
            x = go_explore_expert_replay.sample(batch_size)
            full_opt.zero_grad()
            #im_loss(x).backward()
            imitation_game(x).backward()
            full_opt.step()
            #full_opt.zero_grad()
            #compute_offpolicy_loss(x).backward()
            #full_opt.step()
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
        o = torch.from_numpy(np.asarray(o)).unsqueeze(0).cuda()
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    @torch.no_grad()
    def calc_adv(o, a, r, o2, d):
        return 0.0
        #o_o2 = ac.conv(torch.from_numpy(np.stack([o, o2])).cuda())
        #logit_a = ac.pi(o_o2)[1]
        #q1, _, _, v1 = ac.Q_values(o_o2, logit_a, True)
        #q2, _, _, v2 = ac.Q_values(o_o2, logit_a, False)

        #q1 = q1[0, a]
        #q2 = q2[0, a]
        #x = F.log_softmax(logit_a[0, :], 0)[a]
        #r = np.clip(np.asarray(r), -1, 1)
        #if q1 > q2:
        #    return (r + (1 - d) * gamma * v2[1, :] - v2[0, :] - alpha * x).item()
        #else:
        #    return (r + (1 - d) * gamma * v1[1, :] - v1[0, :] - alpha * x).item()

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

    # debug - start
    episodic_returns = deque(maxlen=history_len)
    episode_s_a = []
    # debug - end

    logger = setup_logging('output', '{}.txt'.format('sac'))
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    ep_ret_best = -1e5
    # Main loop: collect experience in env and update/log each epoch
    for t in range(1, total_steps + 1):
        if t > start_steps:
            #if True:
            a, pi_old = get_action(o)
        else:
            a = env.action_space.sample()
            pi_old = np.asarray(1.0 / act_dim, dtype=np.float32)

        # Step the env
        o2, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1

        # debug - start
        adv_s1_s2 = calc_adv(o, a, r, o2, d)
        # debug - end

        # Store experience to replay buffer
        replay_buffer.add(o, a, np.clip(np.asarray(r), -1, 1), o2, float(d),
                          pi_old, adv_s1_s2)
        # debug - start
        episode_s_a.append((o, a, np.clip(np.asarray(r), -1,
                                          1), o2, float(d), pi_old))
        # debug - end

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d:
            logger.info('t: {}, ep_ret: {}, ep_len: {}'.format(
                t, ep_ret, ep_len))

            # debug - start
            expert_replay.append({
                'timestamp': t,
                'return': ep_ret,
                'exp': episode_s_a
            })
            # reset (important!!!)
            episode_s_a = []

            episodic_returns.append((t, ep_ret))
            if len(episodic_returns) >= history_len:
                expert_replay.collate(list(episodic_returns))
            # debug - end

            o, ep_ret, ep_len = env.reset(), 0, 0
        # go-explore - start
        if communicate_queue is not None and not communicate_queue.empty():
            list_of_actions, cumulative_reward, timestamp = communicate_queue.get(
            )
            go_explore_episode_s_a = []
            go_explore_env = MyAtari(env_name)
            score = 0
            s_0 = go_explore_env.reset()
            for a in list_of_actions:
                s_1, reward, done, _ = go_explore_env.step(a)
                score += reward
                go_explore_episode_s_a.append(
                    (s_0, a, np.clip(np.asarray(reward), -1,
                                     1), s_1, float(done), 1.0))
                s_0 = s_1
                if done:
                    assert score == cumulative_reward, 'recovering the exact trajectory error! score: {} cumulative_reward: {}'.format(
                        score, cumulative_reward)
                    logger.info('GOOD!!!')
                    logger.info(
                        'go-explore t: {}, cumulative_reward: {}, #transitions: {}'
                        .format(timestamp, cumulative_reward,
                                len(go_explore_episode_s_a)))
                    go_explore_expert_replay.append({
                        'timestamp':
                        timestamp,
                        'return':
                        score,
                        'exp':
                        go_explore_episode_s_a,
                    })
        if len(episodic_returns) >= history_len:
            go_explore_expert_replay.collate(list(episodic_returns))
        # go-explore - end
        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                update()

        if t % int(1e5) == 0:
            import gc
            gc.collect()

        # End of epoch handling
        if t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            #if epoch in {10, epochs}:
            #if True:
            if epoch % 2 == 0:
                #val_ep_ret, val_ep_len = val_agent(100)
                val_ep_ret, val_ep_len = val_agent(1)
                logger.info('test score@epoch {}: {}, ep_len: {}'.format(
                    epoch, val_ep_ret, val_ep_len))

                ##if ep_ret > ep_ret_best:
                #if val_ep_ret > ep_ret_best:
                #    #ep_ret_best = ep_ret
                #    ep_ret_best = val_ep_ret
                #    torch.save(
                #        ac.state_dict(),
                #        os.path.join('output',
                #                     'model-{}.pth'.format(logger_args)))
                #if epoch in [10, 50, 100]:
                #    shutil.copy2(
                #        os.path.join('output',
                #                     'model-{}.pth'.format(logger_args)),
                #        os.path.join(
                #            'output',
                #            'model-{}-epoch_{}.pth'.format(logger_args,
                #                                           epoch)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    print("env: {}, seed: {}, epochs: {}".format(args.env, args.seed,
                                                 args.epochs))
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed + 1)
    np.random.seed(seed + 2)

    #x = lambda: make_env(args.env,
    #                     seed=args.seed + 3,
    #                     wrapper_kwargs={
    #                         'frame_stack': True,
    #                         'clip_rewards': False,
    #                         'episode_life': False,
    #                     })
    #y = lambda: make_env(args.env,
    #                     seed=args.seed + 4,
    #                     wrapper_kwargs={
    #                         'frame_stack': True,
    #                         'clip_rewards': False,
    #                         'episode_life': False,
    #                     })
    sac(  #x,
        #y,
        communicate_queue=None,
        env_name='Pong',
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs)
