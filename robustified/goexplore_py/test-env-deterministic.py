import gym
import numpy as np
import random

if __name__ == '__main__':
    random.seed(3)
    np.random.seed(2)
    env = gym.make('MontezumaRevengeDeterministic-v4')

    env.seed(11)
    x1 = env.reset()
    x2 = env.step(0)[0]

    random.seed(5)
    np.random.seed(7)
    env.seed(11)
    y1 = env.reset()
    y2 = env.step(0)[0]

    assert (x1 == y1).all() and (x2 == y2).all()
