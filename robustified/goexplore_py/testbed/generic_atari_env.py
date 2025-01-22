from collections import deque
import numpy as np
import typing
import gym
from gym.wrappers import TimeLimit 
import cv2


class MyAtari:

    def __init__(self, name, end_on_death=False):
        self.name = name
        self.env = TimeLimit(gym.make('{}Deterministic-v4'.format(self.name)), max_episode_steps=10_000)
        self.unwrapped.seed(0)
        self.env.reset()
        self.unprocessed_state = None
        self.end_on_death = end_on_death
        self.prev_lives = 0

        # sac
        # FrameStack
        self._k = 4
        self._w, self._h = (84, 84)
        self._frames_sac = deque([], maxlen=self._k)

    def __getattr__(self, e):
        return getattr(self.env, e)

    # sac
    def _process(self, t):
        t = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
        t = cv2.resize(t, (self._w, self._h), interpolation=cv2.INTER_AREA)
        return np.expand_dims(t, -1)

    # sac
    def _stack_frame(self):
        return np.concatenate(list(self._frames_sac), axis=-1)

    def reset(self) -> np.ndarray:
        self.env = TimeLimit(gym.make('{}Deterministic-v4'.format(self.name)), max_episode_steps=10_000)
        self.unwrapped.seed(0)
        self.unprocessed_state = self.env.reset()

        # sac
        t = self._process(self.unprocessed_state)
        for _ in range(self._k):
            self._frames_sac.append(t)
        return self._stack_frame()

    def step(self, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        self.unprocessed_state, reward, done, lol = self.env.step(action)
        # sac
        self._frames_sac.append(self._process(self.unprocessed_state))
        state_sac = self._stack_frame()

        cur_lives = self.env.unwrapped.ale.lives()
        if self.end_on_death and cur_lives < self.prev_lives:
            done = True
        self.prev_lives = cur_lives
        return state_sac, reward, done, None
