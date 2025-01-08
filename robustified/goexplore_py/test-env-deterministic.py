import gym
import numpy as np
import random

if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    random.seed(3)
    np.random.seed(2)
    env = gym.make('MontezumaRevengeDeterministic-v4')

    env.seed(11)
    x1 = env.reset()
    x2 = env.step(rng.integers(0, env.action_space.n))[0]
    x3 = env.step(rng.integers(0, env.action_space.n))[0]

    rng = np.random.default_rng(seed=42)
    random.seed(5)
    np.random.seed(7)

    env.seed(11)
    y1 = env.reset()
    y2 = env.step(rng.integers(0, env.action_space.n))[0]
    y3 = env.step(rng.integers(0, env.action_space.n))[0]

    assert (x1 == y1).all() and (x2 == y2).all() and (x3 == y3).all()
