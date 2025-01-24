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
DONE = None


class LPool:

    def __init__(self, n_cpus):
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
                 restore=None,
                 cell_frame=None):
        self.score = score
        self._seen_times = seen_times

        self.trajectory_len = trajectory_len
        self.restore = restore
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
    __slots__ = ['cell', 'restore', 'frame']
    cell: tuple
    restore: typing.Any
    frame: typing.Any


@dataclass
class TrajectoryElement:
    __slots__ = ['to', 'action', 'reward', 'done']
    to: PosInfo
    action: int
    reward: float
    done: bool


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
        POOL = pool_class(multiprocessing.cpu_count(), )

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
        self.former_grids.append(copy.deepcopy(self.grid))

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

    def get_dynamic_repr(self, orig_state):
        if isinstance(orig_state, bytes):
            orig_state = RLEArray.frombytes(orig_state, dtype=np.uint8)
        orig_state = orig_state.to_np()
        dynamic_repr = []

        target_size, max_pix_val = self.dynamic_state_split_rules
        if target_size is None:
            dynamic_repr.append(1)
        else:
            state = imdownscale(orig_state, target_size, max_pix_val)
            dynamic_repr.append(state.tobytes())
        return tuple(dynamic_repr)

    def try_split_frames(self, frames):
        n_processes = multiprocessing.cpu_count()
        frames = [RLEArray.frombytes(f, dtype=np.uint8) for f in frames]
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

        # Intuition: we want our batch size to be such that it will be processed in two passes
        BATCH_SIZE = len(frames) // (n_processes // 2 + 1) + 1

        def proc_downscale(to_process, returns):
            while True:
                start_batch, cur_shape, cur_pix_val = to_process.get()
                if start_batch == -1:
                    return
                results = []
                for i in range(start_batch,
                               min(len(frames), start_batch + BATCH_SIZE)):
                    results.append(
                        imdownscale(frames[i].to_np(), cur_shape,
                                    cur_pix_val).tobytes())
                returns.put(results)

        to_process = multiprocessing.Queue()
        returns = multiprocessing.Queue()
        processes = [
            multiprocessing.Process(target=proc_downscale,
                                    args=(to_process, returns))
            for _ in range(n_processes)
        ]
        for p in processes:
            p.start()

        for _ in range(self.args.split_iterations):
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
                    print(
                        'NEW BEST score: {:.4f} n: {: <8} shape: ({: <3}, {: <3}) {: <3}'
                        .format(cur_score, len(dist), *cur_shape, cur_pix_val))
                best_score = cur_score
                best_shape = cur_shape
                best_n = len(dist)
                best_pix_val = cur_pix_val

        for i in range(n_processes):
            to_process.put((-1, None, None))
        for p in processes:
            try:
                p.join(1)
            except Exception:
                p.terminate()

        return best_shape, best_pix_val, best_n

    def maybe_split_dynamic_state(self):
        if (self.frames_compute - self.last_recompute_dynamic_state
                > self.args.recompute_dynamic_state_every
                or len(self.grid) > self.args.max_archive_size):
            if len(self.grid) > self.args.max_archive_size:
                print(
                    'Recomputing representation because of archive size (should not happen too often)'
                )
            self.last_recompute_dynamic_state = self.frames_compute

            print('Recomputing state representation')
            best_shape, best_pix_val, best_n = self.try_split_frames(
                self.random_recent_frames)
            print(
                f'Switching representation to {best_shape} with {best_pix_val} pixels ({best_n} / {len(self.random_recent_frames)})'
            )
            if self.dynamic_state_split_rules[0] is None:
                self.grid = self.former_grids.pop()
            self.dynamic_state_split_rules = (best_shape, best_pix_val)

            self.random_recent_frames.clear()
            self.selector.clear_all_cache()
            self.former_grids.append(self.grid)
            self.grid = defaultdict(Cell)

            start = time.time()
            for grid_idx in reversed(range(len(self.former_grids))):
                old_grid = self.former_grids[grid_idx]
                n_processes = multiprocessing.cpu_count()
                to_process = multiprocessing.Queue()
                returns = multiprocessing.Queue()

                def iter_grid(grid_idx, old_grid):
                    in_queue = set()
                    has_had_timeout = [False]

                    def queue_process_min(min_size):
                        while len(in_queue) > min_size:
                            if has_had_timeout[0]:
                                cur = in_queue.pop()
                                _, old_key, new_key = get_repr(grid_idx, cur)
                                yield old_key, old_grid[old_key], new_key
                            else:
                                import queue
                                try:
                                    _, old_key, new_key = returns.get(
                                        timeout=5 * 60)
                                    if old_key in in_queue:
                                        in_queue.remove(old_key)
                                        yield old_key, old_grid[
                                            old_key], new_key
                                    else:
                                        print(
                                            f'Warning: saw duplicate key: {old_key}'
                                        )
                                except queue.Empty:
                                    has_had_timeout[0] = True
                                    print(
                                        'Warning: timeout in receiving from queue. Switching to 100% single threaded'
                                    )

                    for k in old_grid:
                        for to_yield in queue_process_min(n_processes):
                            yield to_yield
                        if not has_had_timeout[0]:
                            to_process.put((grid_idx, k), timeout=60)
                        in_queue.add(k)
                    for to_yield in queue_process_min(0):
                        yield to_yield

                def get_repr(i_grid, key):
                    frame = self.former_grids[i_grid][key].cell_frame
                    if frame is None or key is None:
                        return ((i_grid, key, key))
                    else:
                        return ((i_grid, key, self.get_dynamic_repr(frame)))

                def redo_repr():
                    while True:
                        i_grid, key = to_process.get()
                        if i_grid is None:
                            return
                        returns.put(get_repr(i_grid, key))

                processes = [
                    multiprocessing.Process(target=redo_repr)
                    for _ in range(n_processes)
                ]
                for p in processes:
                    p.start()
                for cell_key, cell, new_key in iter_grid(grid_idx, old_grid):
                    if new_key not in self.grid or self.should_accept_cell(
                            self.grid[new_key], cell.score,
                            cell.trajectory_len):
                        if new_key not in self.grid:
                            self.grid[new_key] = Cell()
                        self.grid[new_key].score = cell.score
                        self.grid[new_key].trajectory_len = cell.trajectory_len
                        self.grid[new_key].restore = cell.restore
                        self.grid[new_key].action_seq = cell.action_seq
                        self.grid[new_key].cell_frame = cell.cell_frame
                        if self.args.reset_cell_on_update:
                            self.grid[new_key].set_seen_times(cell.seen_times)
                    self.selector.cell_update(new_key, self.grid[new_key])
                for _ in range(n_processes):
                    to_process.put((None, None), block=False)
                for p in processes:
                    try:
                        p.join(timeout=1)
                    except Exception:
                        p.terminate()
                        p.join()
            self.prev_len_grid = len(self.grid)
            print(
                f'New size: {len(self.grid)}. Old size: {len(self.former_grids[-1])}'
            )

    def get_frame(self, asbytes):
        frame = ENV.state[-1]
        return frame.tobytes() if asbytes else frame

    def get_pos_info(self, include_restore=True):
        return PosInfo(
            self.get_cell(),
            self.get_restore() if include_restore else None,
            self.get_frame(True) if self.args.dynamic_state else None)

    def get_restore(self):
        return ENV.get_restore()

    def restore(self, val):
        self.make_env()
        ENV.restore(val)

    def get_cell(self):
        return self.get_dynamic_repr(self.get_frame(False))

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
        #_, state_sac_0 = self.reset()
        #for action in cell.action_seq:
        #    _, reward, done, state_sac_1 = self.step(action)
        #    # TODO
        #    # (state_sac_0, action, reward, state_sac_1, done)
        #    state_sac_0 = state_sac_1
        #assert np.all(self.get_frame(True) == cell.cell_frame)

        #self.frames_true += len(cell.action_seq)
        state_sac_0 = None
        if cell.restore is not None:
            self.restore(cell.restore)
            self.frames_true += cell.trajectory_len
        else:
            assert cell.trajectory_len == 0, 'Cells must have a restore unless they are the initial state'
            self.reset()
        # explore-step
        end_trajectory = self.run_seed(seed,
                                       state_sac_0,
                                       max_steps=self.args.explore_steps)
        return TimedPickle(
            (cell_key, end_trajectory, self.frames_true, self.frames_compute),
            'ret',
            enabled=info.enabled)

    def run_cycle(self, communicate_queue):
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
            # to alter within self.grid, should index by self.grid[cell_key]
            # cell_copy is only a copy, not within self.grid
            start_cell = self.grid[cell_key]
            start_cell.inc_seen_times(1)
            self.selector.cell_update(cell_key, start_cell)

            cur_score = cell_copy.score
            potential_cell = start_cell
            old_potential_cell_key = cell_key
            act_seq = list()
            for k, elem in enumerate(end_trajectory):
                act_seq.append(elem.action)
                potential_cell_key = DONE if elem.done else elem.to.cell
                if potential_cell_key != DONE and (
                        random.random() < self.args.recent_frame_add_prob):
                    self.random_recent_frames.add(elem.to.frame)
                if potential_cell_key != old_potential_cell_key:
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
                old_potential_cell_key = potential_cell_key
                full_traj_len = cell_copy.trajectory_len + k + 1
                cur_score += elem.reward

                if self.should_accept_cell(potential_cell, cur_score,
                                           full_traj_len):
                    cells_to_reset.add(potential_cell_key)
                    potential_cell.trajectory_len = full_traj_len
                    potential_cell.restore = elem.to.restore
                    potential_cell.action_seq = np.concatenate(
                        (cell_copy.action_seq, np.array(act_seq,
                                                        dtype=np.uint8)))
                    potential_cell.score = cur_score
                    if cur_score > self.max_score:
                        self.max_score = cur_score
                    potential_cell.cell_frame = elem.to.frame
                    self.selector.cell_update(potential_cell_key,
                                              potential_cell)
                    if potential_cell_key == DONE:
                        print('DONE enqueue...')
                        communicate_queue.put(
                            (copy.deepcopy(potential_cell.action_seq),
                             potential_cell.score, self.frames_compute))
                        if ((self.env_info[1]['name'] == 'Pong'
                             and self.grid[DONE].score > 0) or
                            (self.env_info[1]['name'] == 'MontezumaRevenge'
                             #and self.grid[DONE].score > 600)):
                             and self.grid[DONE].score >= 100)):
                            print('sleep for 24 hours...')
                            time.sleep(24 * 60 * 60)
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
