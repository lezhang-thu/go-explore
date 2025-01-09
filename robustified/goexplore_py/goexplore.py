# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

from .generic_atari_env import *
from .utils import *
import loky
import bz2
from multiprocessing import shared_memory
import pickle

compress = bz2
compress_suffix = '.bz2'
compress_kwargs = {}
n_digits = 20
DONE = None


class LPool:

    def __init__(self, n_cpus, maxtasksperchild=100):
        self.pool = loky.get_reusable_executor(n_cpus, timeout=100)

    def map(self, f, r):
        return self.pool.map(f, r)


def run_f_seeded(args):
    f, seed, args = args
    with use_seed(seed):
        return f(args)


class SeedPoolWrap:

    def __init__(self, pool):
        self.pool = pool

    def map(self, f, r):
        return self.pool.map(run_f_seeded,
                             [(f, random.randint(0, 2**32 - 10), e)
                              for e in r])


def seed_pool_wrapper(pool_class):

    def f(*args, **kwargs):
        return SeedPoolWrap(pool_class(*args, **kwargs))

    return f


class Cell:

    def __init__(self,
                 score=-infinity,
                 seen_times=0,
                 trajectory_len=infinity,
                 cell_frame=None):
        self.score = score
        self._seen_times = seen_times

        self.trajectory_len = trajectory_len
        self.cell_frame = cell_frame

        # sac
        self.action_seq = np.zeros(0, dtype=np.uint8)

    @property
    def seen_times(self):
        return self._seen_times

    def inc_seen_times(self, value):
        self._seen_times += value

    def set_seen_times(self, value):
        self._seen_times = value


@dataclass
class PosInfo:
    __slots__ = ['cell', 'frame']
    cell: tuple
    frame: typing.Any


@dataclass
class TrajectoryElement:
    __slots__ = ['to', 'action', 'reward', 'done']
    to: PosInfo
    action: int
    reward: float
    done: bool


Experience = tuple

# ### Main


class RotatingSet:

    def __init__(self, M):
        self.max_size = M
        self.clear()

    def clear(self):
        self.set = set()
        self.list = collections.deque(maxlen=self.max_size)

    def add(self, e):
        if e in self.set:
            return
        if len(self.list) == self.max_size:
            self.set.remove(self.list[0])
        self.list.append(e)
        self.set.add(e)
        assert len(self.list) == len(self.set)

    def __iter__(self):
        for e in self.list:
            yield e

    def __len__(self):
        return len(self.list)


POOL = None
ENV = None


def get_env():
    return ENV


def get_downscale(args):
    f, cur_shape, cur_pix_val = args
    return imdownscale(f, cur_shape, cur_pix_val).flatten().tobytes()


@functools.lru_cache(maxsize=1)
def get_saved_grid(file):
    return pickle.load(compress.open(file, 'rb'))


class FormerGrids:

    def __init__(self, args):
        self.args = args
        self.cur_length = 0

    def _getfilename(self, i):
        return f'{self.args.base_path}/__grid_{i}.pickle{compress_suffix}'

    def append(self, elem):
        filename = self._getfilename(self.cur_length)
        assert not os.path.exists(filename)
        fastdump(elem, compress.open(filename, 'wb'))
        self.cur_length += 1

    def pop(self):
        assert self.cur_length >= 1
        filename = self._getfilename(self.cur_length - 1)
        res = get_saved_grid(filename)
        os.remove(filename)
        self.cur_length -= 1
        return res

    def __getitem__(self, item):
        if item < 0:
            item = self.cur_length + item
        return get_saved_grid(self._getfilename(item))

    def __len__(self):
        return self.cur_length


class Explore:

    def __init__(
        self,
        explorer_policy,
        cell_selector,
        env,
        pool_class,
        args,
    ):
        global POOL, ENV
        self.args = args

        self.env_info = env
        self.make_env()
        POOL = pool_class(multiprocessing.cpu_count() * 2,
                          maxtasksperchild=100)

        self.explorer = explorer_policy
        self.selector = cell_selector
        self.grid = defaultdict(Cell)
        self.frames_true = 0
        self.frames_compute = 0
        self.start = None
        self.cycles = 0
        self.dynamic_state_split_rules = (None, None)
        self.random_recent_frames = RotatingSet(self.args.max_recent_frames)
        self.last_recompute_dynamic_state = -self.args.recompute_dynamic_state_every + self.args.first_compute_dynamic_state

        self.max_score = 0
        self.prev_len_grid = 0

        self.reset()

        self.normal_frame_shape = (160, 210)
        cell_key = self.get_cell()
        self.grid[cell_key] = Cell()
        self.grid[cell_key].trajectory_len = 0
        self.grid[cell_key].score = 0
        self.grid[cell_key].cell_frame = self.get_frame(True)
        # Create the DONE cell
        self.grid[DONE] = Cell()
        self.selector.cell_update(cell_key, self.grid[cell_key])
        self.selector.cell_update(DONE, self.grid[DONE])
        # summary: cell_key - reset (init) cell. DONE - done cell
        self.former_grids = FormerGrids(args)

    def make_env(self):
        global ENV
        if ENV is None:
            ENV = self.env_info[0](**self.env_info[1])
            ENV.reset()

    def reset(self):
        self.make_env()
        return ENV.reset()

    def step(self, action):
        return ENV.step(action)

    @staticmethod
    def get_dynamic_repr(orig_state,
                         target_size,
                         max_pix_val,
                         first_compute_dynamic_state=None):
        if isinstance(orig_state, bytes):
            orig_state = RLEArray.frombytes(orig_state, dtype=np.uint8)
        orig_state = orig_state.to_np()
        dynamic_repr = []

        if target_size is None:
            dynamic_repr.append(random.randint(1, first_compute_dynamic_state))
        else:
            state = imdownscale(orig_state, target_size, max_pix_val)
            dynamic_repr.append(state.tobytes())
        return tuple(dynamic_repr)

    def try_split_frames(self, frames):
        tqdm.write('Decoding frames')
        frames = [
            RLEArray.frombytes(f, dtype=np.uint8).to_np() for f in frames
        ]

        frames_np = np.array(frames)
        shm = shared_memory.SharedMemory(create=True, size=frames_np.nbytes)
        shared_frames = np.ndarray(frames_np.shape,
                                   dtype=frames_np.dtype,
                                   buffer=shm.buf)
        np.copyto(shared_frames, frames_np)

        tqdm.write('Frames decoded')
        unif_ent_cache = {}

        def get_dist_score(dist):
            if len(dist) == 1:
                return 0.0
            from math import log

            def ent(dist):
                return -sum(log(e) * e for e in dist)

            def unif_ent(l):
                if l not in unif_ent_cache:
                    return ent([1 / l] * l)
                return unif_ent_cache[l]

            def norment(dist):
                return ent(dist) / unif_ent(len(dist))

            target_len = len(frames) * self.args.cell_split_factor
            return norment(dist) / np.sqrt(
                abs(len(dist) - target_len) / target_len + 1)

        best_shape = (random.randint(1, self.normal_frame_shape[0] - 1),
                      random.randint(1, self.normal_frame_shape[1] - 1))
        best_pix_val = random.randint(2, 255)
        best_score = -infinity
        best_n = 0
        seen = set()

        n_processes = multiprocessing.cpu_count()
        # Intuition: we want our batch size to be such that it will be processed in two passes
        # len(frames): f, n_process: n
        # the following gives: ceil( f / (2n)). ideally, into 2n segments.
        BATCH_SIZE = (len(frames) + 2 * n_processes - 1) // (2 * n_processes)

        def proc_downscale(to_process, returns, batch_size, shm_name, shape,
                           dtype):
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            shared_frames = np.ndarray(shape,
                                       dtype=dtype,
                                       buffer=existing_shm.buf)

            while True:
                start_batch, cur_shape, cur_pix_val = to_process.get()
                if start_batch == -1:
                    existing_shm.close()
                    return
                results = []
                for i in range(
                        start_batch,
                        min(len(shared_frames), start_batch + batch_size)):
                    results.append(
                        imdownscale(shared_frames[i], cur_shape,
                                    cur_pix_val).tobytes())
                returns.put(results)

        tqdm.write('Creating processes')
        to_process = multiprocessing.Queue()
        returns = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(target=proc_downscale,
                                    args=(to_process, returns, BATCH_SIZE,
                                          shm.name, shared_frames.shape,
                                          shared_frames.dtype))
            for _ in range(n_processes)
        ]
        for p in processes:
            p.start()
        tqdm.write('Processes created')

        for _ in tqdm(range(self.args.split_iterations),
                      desc='New representation'):
            cur_shape = best_shape
            cur_pix_val = best_pix_val
            while (cur_shape, cur_pix_val) in seen:
                cur_shape = list(best_shape)
                for idx in range(2):
                    while True:
                        cur_shape[idx] = np.random.geometric(
                            min(1 / (best_shape[idx] + 1),
                                20 / self.normal_frame_shape[idx]))
                        if cur_shape[idx] >= 1 and cur_shape[
                                idx] <= self.normal_frame_shape[idx] - 1:
                            break
                cur_shape = tuple(cur_shape)
                while True:
                    cur_pix_val = np.random.geometric(
                        min(1 / best_pix_val, 1 / 12))
                    if cur_pix_val >= 2 and cur_pix_val <= 255:
                        break
            seen.add((cur_shape, cur_pix_val))

            for i in range(0, len(frames), BATCH_SIZE):
                to_process.put((i, cur_shape, cur_pix_val))
            downscaled = []
            for _ in range(0, len(frames), BATCH_SIZE):
                downscaled += returns.get()

            dist = np.array(list(Counter(downscaled).values())) / len(frames)
            cur_score = get_dist_score(dist)

            if cur_score >= best_score:
                if cur_score > best_score:
                    tqdm.write(
                        f'NEW BEST score: {cur_score} n: {len(dist)} shape:{cur_shape} {cur_pix_val}'
                    )
                best_score = cur_score
                best_shape = cur_shape
                best_n = len(dist)
                best_pix_val = cur_pix_val

        for i in range(n_processes):
            to_process.put((-1, None, None))
        for p in processes:
            p.join()
        # Clean up shared memory
        shm.close()
        shm.unlink()
        return best_shape, best_pix_val, best_n

    def maybe_split_dynamic_state(self):
        if not (self.frames_compute - self.last_recompute_dynamic_state
                > self.args.recompute_dynamic_state_every
                or len(self.grid) > self.args.max_archive_size):
            return
        if True: return

        if len(self.grid) > self.args.max_archive_size:
            tqdm.write(
                'Recomputing representation because of archive size (should not happen too often)'
            )
        self.last_recompute_dynamic_state = self.frames_compute

        tqdm.write('Recomputing state representation')
        best_shape, best_pix_val, best_n = self.try_split_frames(
            self.random_recent_frames)
        tqdm.write(
            f'Switching representation to {best_shape} with {best_pix_val} pixels ({best_n} / {len(self.random_recent_frames)})'
        )
        self.dynamic_state_split_rules = (best_shape, best_pix_val)

        self.random_recent_frames.clear()
        self.selector.clear_all_cache()
        self.former_grids.append(self.grid)
        self.grid = defaultdict(Cell)

        start = time.time()
        for grid_idx in tqdm(reversed(range(len(self.former_grids))),
                             desc='recompute_grid'):
            tqdm.write('Loading grid')
            old_grid = self.former_grids[grid_idx]
            tqdm.write('Creating queues')
            to_process = multiprocessing.Queue()
            returns = multiprocessing.Queue()

            def get_repr(cell_key, cell_frame):
                if cell_key == DONE:
                    return ((cell_key, cell_key))
                else:
                    return ((cell_key,
                             Explore.get_dynamic_repr(cell_frame, best_shape,
                                                      best_pix_val)))

            def redo_repr():
                while True:
                    f, cell_key, cell_frame = to_process.get()
                    if not f:
                        return
                    returns.put(get_repr(cell_key, cell_frame))

            for cell_key, cell in tqdm(old_grid.items(), desc='add_to_grid'):
                to_process.put((True, cell_key, cell.cell_frame))
            tqdm.write('Creating processes')
            n_processes = multiprocessing.cpu_count()
            processes = [
                multiprocessing.Process(target=redo_repr)
                for _ in range(n_processes)
            ]
            tqdm.write('Starting processes')
            for p in processes:
                p.start()
            tqdm.write('Processes started')

            def iter_grid(num):
                tqdm.write('Iter grid')
                for _ in range(num):
                    old_key, new_key = returns.get()
                    yield old_grid[old_key], new_key
                tqdm.write('Done iter grid')

            for cell, new_key in iter_grid(len(old_grid)):
                if new_key not in self.grid or self.should_accept_cell(
                        self.grid[new_key], cell.score, cell.trajectory_len):
                    t = self.grid[new_key]
                    t.score = cell.score
                    t.trajectory_len = cell.trajectory_len
                    t.action_seq = cell.action_seq
                    t.cell_frame = cell.cell_frame

                    if self.args.reset_cell_on_update:
                        self.grid[new_key].set_seen_times(cell.seen_times)
                    # TODO verify. else: in self.grid but worse. then, cell.seen_times is the same
                    self.selector.cell_update(new_key, self.grid[new_key])
            tqdm.write('Clearing processes')
            for _ in range(n_processes):
                to_process.put((False, None, None))
            for p in tqdm(processes, desc='processes_clear'):
                p.join()
            tqdm.write('Processes cleared')
        tqdm.write(f'Recomputing the grid took {time.time() - start} seconds')
        self.prev_len_grid = len(self.grid)
        tqdm.write(
            f'New size: {len(self.grid)}. Old size: {len(self.former_grids[-1])}'
        )

    def get_frame(self, asbytes):
        frame = ENV.state[-1]
        return frame.tobytes() if asbytes else frame

    def get_pos_info(self):
        return PosInfo(
            self.get_cell(),
            self.get_frame(True) if self.args.dynamic_state else None)

    def get_cell(self):
        return self.get_dynamic_repr(self.get_frame(False),
                                     *self.dynamic_state_split_rules,
                                     self.args.first_compute_dynamic_state)

    def run_explorer(self, explorer, state_sac_0=None, max_steps=-1):
        trajectory = []
        while True:
            if ((max_steps > 0 and len(trajectory) >= max_steps)):
                break
            action = explorer.get_action(ENV)
            _, reward, done, state_sac_1 = self.step(action)
            # TODO
            # (state_sac_0, action, reward, state_sac_1, done)
            state_sac_0 = state_sac_1

            self.frames_true += 1
            self.frames_compute += 1
            trajectory.append(
                TrajectoryElement(
                    # initial_pos_info,
                    self.get_pos_info(),
                    action,
                    reward,
                    done,
                ))
            if done:
                break
        return trajectory

    def run_seed(self, seed, start_sac=None, max_steps=-1):
        with use_seed(seed):
            self.explorer.init_seed()
            return self.run_explorer(self.explorer, start_sac, max_steps)

    def process_cell(self, info):
        # This function runs in a SUBPROCESS, and processes a single cell.
        cell_key, cell, seed, target_shape, max_pix = info.data
        assert cell_key != DONE
        self.env_info[0].TARGET_SHAPE = target_shape
        self.env_info[0].MAX_PIX_VALUE = max_pix
        self.frames_true = 0
        self.frames_compute = 0

        # go-step
        _, state_sac_0 = self.reset()
        for action in cell.action_seq:
            _, reward, done, state_sac_1 = self.step(action)
            # TODO
            # (state_sac_0, action, reward, state_sac_1, done)
            state_sac_0 = state_sac_1
        assert np.all(
            self.get_frame(True) == cell.cell_frame), '\n{}\n{}\n{}\n'.format(
                self.get_frame(True), cell.cell_frame, cell.action_seq)
        if len(cell.action_seq) > 0:
            print('#' * 20)
            print('hit once! {}'.format(len(cell.action_seq)))

        self.frames_true += len(cell.action_seq)
        # explore-step
        end_trajectory = self.run_seed(seed,
                                       state_sac_0,
                                       max_steps=self.args.explore_steps)
        return TimedPickle(
            (cell_key, end_trajectory, self.frames_true, self.frames_compute),
            'ret',
            enabled=info.enabled)

    def run_cycle(self):
        # Choose a bunch of cells, send them to the workers for processing, then combine the results.
        # A lot of what this function does is only aimed at minimizing the amount of data that needs
        # to be pickled to the workers, which is why it sets a lot of variables to None only to restore
        # them later.
        global POOL
        if self.start is None:
            self.start = time.time()

        if self.args.dynamic_state:
            self.maybe_split_dynamic_state()

        self.cycles += 1
        chosen_cells = []
        cell_keys = self.selector.choose_cell(self.grid,
                                              size=self.args.batch_size)
        for i, cell_key in enumerate(cell_keys):
            cell_copy = self.grid[cell_key]
            seed = random.randint(0, 2**31)
            chosen_cells.append(
                TimedPickle(
                    (cell_key, cell_copy, seed, self.env_info[0].TARGET_SHAPE,
                     self.env_info[0].MAX_PIX_VALUE),
                    'args',
                    enabled=False))

        # NB: save some of the attrs that won't be necessary but are very large, and set them to none instead,
        #     this way they won't be pickled.
        cache = {}
        to_save = [
            'grid',
            'former_grids',
            'selector',
            'random_recent_frames',
        ]
        for attr in to_save:
            cache[attr] = getattr(self, attr)
            setattr(self, attr, None)

        trajectories = [
            e.data for e in POOL.map(self.process_cell, chosen_cells)
        ]
        chosen_cells = [e.data for e in chosen_cells]

        for attr, v in cache.items():
            setattr(self, attr, v)

        # Note: we do this now because starting here we're going to be concatenating the trajectories
        # of these cells, and they need to remain the same!
        chosen_cells = [(k, copy.copy(c), s, shape, pix)
                        for k, c, s, shape, pix in chosen_cells]
        cells_to_reset = set()

        for ((cell_key, cell_copy, _, _, _),
             (_, end_trajectory, ft, fc)) in zip(chosen_cells, trajectories):
            self.frames_true += ft
            self.frames_compute += fc

            seen_cells = {cell_key}
            start_cell = self.grid[cell_key]
            start_cell.inc_seen_times(1)
            self.selector.cell_update(cell_key, start_cell)

            cur_score = cell_copy.score
            act_seq = list()
            for k, elem in enumerate(end_trajectory):
                act_seq.append(elem.action)
                potential_cell_key = DONE if elem.done else elem.to.cell
                if potential_cell_key != DONE and (
                        random.random() < self.args.recent_frame_add_prob):
                    self.random_recent_frames.add(elem.to.frame)
                was_in_grid = potential_cell_key in self.grid
                # type(self.grid): defaultdict(Cell)
                # was_in_grid should run beforehand
                # potential_cell might be a new cell, or old. it depends!
                potential_cell = self.grid[potential_cell_key]
                if potential_cell_key not in seen_cells:
                    seen_cells.add(potential_cell_key)
                    potential_cell.inc_seen_times(1)
                    if was_in_grid:
                        self.selector.cell_update(potential_cell_key,
                                                  potential_cell)
                full_traj_len = cell_copy.trajectory_len + k + 1
                cur_score += elem.reward
                
                assert was_in_grid or self.should_accept_cell(potential_cell, cur_score,
                                           full_traj_len), '\n{}\n{}'.format(potential_cell.score, cur_score)
                if self.should_accept_cell(potential_cell, cur_score,
                                           full_traj_len):
                    cells_to_reset.add(potential_cell_key)
                    potential_cell.trajectory_len = full_traj_len
                    potential_cell.action_seq = np.concatenate(
                        (start_cell.action_seq,
                         np.asarray(act_seq, dtype=np.uint8)))
                    potential_cell.score = cur_score
                    if cur_score > self.max_score:
                        self.max_score = cur_score
                    potential_cell.cell_frame = elem.to.frame

                    # debug - check consistency
                    # go-step
                    if len(start_cell.action_seq) > 20:
                        _, state_sac_0 = self.reset()
                        for action in potential_cell.action_seq:
                            _, reward, done, state_sac_1 = self.step(action)
                            # TODO
                            # (state_sac_0, action, reward, state_sac_1, done)
                            state_sac_0 = state_sac_1
                        if not np.all(
                            self.get_frame(True) == potential_cell.cell_frame):
                            print(start_cell.action_seq)
                            print(potential_cell.action_seq)
                            exit(0)
                        #assert np.all(
                        #    self.get_frame(True) == potential_cell.cell_frame), '\n{}\n{}\n{}\n'.format(
                        #        self.get_frame(True), potential_cell.cell_frame, potential_cell.action_seq)


                    self.selector.cell_update(potential_cell_key,
                                              potential_cell)
        if self.args.reset_cell_on_update:
            for cell_key in cells_to_reset:
                self.grid[cell_key].set_seen_times(0)
        return [(k) for k, c, s, shape, pix in chosen_cells], trajectories

    def should_accept_cell(self, potential_cell, cur_score, full_traj_len):
        if self.args.optimize_score:
            return (cur_score > potential_cell.score
                    or (full_traj_len < potential_cell.trajectory_len
                        and cur_score == potential_cell.score))
        return full_traj_len < potential_cell.trajectory_len
