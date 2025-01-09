# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.insert(0, '/home/ubuntu/lezhang.thu/go-explore/robustified')
print(sys.path)
import argparse
import copy
import glob
import json
import shutil
import psutil
import time
import uuid

import numpy as np
from tqdm import tqdm

from goexplore_py.randselectors import Weight, WeightedSelector
from goexplore_py.explorers import RepeatedRandomExplorer
from goexplore_py.goexplore import Explore, LPool, seed_pool_wrapper, DONE
import goexplore_py.generic_atari_env as generic_atari_env
from goexplore_py.utils import get_code_hash

VERSION = 1

THRESH_TRUE = 20_000_000_000
THRESH_COMPUTE = 1_000_000
MAX_FRAMES = None
MAX_FRAMES_COMPUTE = None
MAX_ITERATIONS = None
MAX_TIME = 12 * 60 * 60
MAX_CELLS = None
MAX_SCORE = None


def _run(base_path, args):
    explorer = RepeatedRandomExplorer(args.repeat_action)
    if args.dynamic_state:
        args.target_shape = (-1, -1)
        args.max_pix_value = -1
    if 'generic' in args.game:
        game_class = generic_atari_env.MyAtari
        game_class.TARGET_SHAPE = args.target_shape
        game_class.MAX_PIX_VALUE = args.max_pix_value
        game_args = dict(name=args.game.split('_')[1])
    selector = WeightedSelector(
        game_class,
        seen=Weight(args.seen_weight, args.seen_power),
    )
    pool_cls = seed_pool_wrapper(LPool)

    expl = Explore(
        explorer,
        selector,
        (game_class, game_args),
        pool_class=pool_cls,
        args=args,
    )
    #with tqdm(desc='Time (seconds)', smoothing=0, total=MAX_TIME) as t_time, \
    #        tqdm(desc='Iterations', total=MAX_ITERATIONS) as t_iter, \
    #        tqdm(desc='Compute steps', total=MAX_FRAMES_COMPUTE) as t_compute, \
    #        tqdm(desc='Game step', total=MAX_FRAMES) as t, \
    #        tqdm(desc='Max score', total=MAX_SCORE) as t_score, \
    #        tqdm(desc='Done score', total=MAX_SCORE) as t_done_score, \
    #        tqdm(desc='Cells', total=MAX_CELLS) as t_cells:
    #    t_compute.update(expl.frames_compute)
    #    t.update(expl.frames_true)
    #    start_time = time.time()
    #    last_time = np.round(start_time)
    #    n_iters = 0
    #    prev_checkpoint = None
    if True:
        def should_continue():
            #if MAX_TIME is not None and time.time() - start_time >= MAX_TIME:
            #    return False
            if MAX_FRAMES is not None and expl.frames_true >= MAX_FRAMES:
                return False
            if MAX_FRAMES_COMPUTE is not None and expl.frames_compute >= MAX_FRAMES_COMPUTE:
                return False
            if MAX_ITERATIONS is not None and n_iters >= MAX_ITERATIONS:
                return False
            if MAX_CELLS is not None and len(expl.grid) >= MAX_CELLS:
                return False
            if MAX_SCORE is not None and expl.max_score >= MAX_SCORE:
                return False
            return True

        while should_continue():
            # Run one iteration
            old = expl.frames_true
            old_compute = expl.frames_compute
            old_len_grid = len(expl.grid)
            old_max_score = expl.max_score

            expl.run_cycle()

            #t.update(expl.frames_true - old)
            #t_score.update(expl.max_score - old_max_score)
            #t_done_score.n = expl.grid[DONE].score
            #t_done_score.refresh()
            #t_compute.update(expl.frames_compute - old_compute)
            #t_iter.update(1)
            ## Note: due to the archive compression that can happen with dynamic cell representation,
            ## we need to do this so that tqdm doesn't complain about negative updates.
            #t_cells.n = len(expl.grid)
            #t_cells.refresh()

            #cur_time = np.round(time.time())
            #t_time.update(int(cur_time - last_time))
            #last_time = cur_time
            #n_iters += 1

            ## In some circumstances (see comments), save a checkpoint and some pictures
            #if (old == 0 or  # It is the first iteration
            #        old // THRESH_TRUE != expl.frames_true // THRESH_TRUE
            #        or  # We just passed the THRESH_TRUE threshold
            #        old_compute // THRESH_COMPUTE
            #        != expl.frames_compute // THRESH_COMPUTE
            #        or  # We just passed the THRESH_COMPUTE threshold
            #        not should_continue()):  # This is the last iteration
            #    pass


class Tee(object):

    def __init__(self, name, output):
        self.file = open(name, 'w')
        self.stdout = getattr(sys, output)
        self.output = output
        setattr(sys, self.output, self)

    def __del__(self):
        setattr(sys, self.output, self.stdout)
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def run(base_path, args):
    cur_id = 0
    if os.path.exists(base_path):
        current = glob.glob(base_path + '/*')
        for c in current:
            try:
                idx, _ = c.split('/')[-1].split('_')
                idx = int(idx)
                if idx >= cur_id:
                    cur_id = idx + 1

                if args.seed is not None:
                    if os.path.exists(c + '/has_died'):
                        continue
                    other_kwargs = json.load(open(c + '/kwargs.json'))
                    is_same_kwargs = True
                    for k, v in vars(args).items():

                        def my_neq(a, b):
                            if isinstance(a, tuple):
                                a = list(a)
                            if isinstance(b, tuple):
                                b = list(b)
                            return a != b

                        if k != 'base_path' and my_neq(other_kwargs[k], v):
                            is_same_kwargs = False
                            break
                    if is_same_kwargs:
                        try:
                            last_exp = sorted([
                                e for e in glob.glob(c + '/*_experience.bz2')
                                if 'thisisfake' not in e
                            ])[-1]
                        except IndexError:
                            continue
                        mod_time = os.path.getmtime(last_exp)
                        if time.time() - mod_time < 3600:
                            print('Another run is already running at', c,
                                  'exiting.')
                            return
                        compute = int(last_exp.split('/')[-1].split('_')[1])
                        if compute >= args.max_compute_steps:
                            print(
                                'A completed equivalent run already exists at',
                                c, 'exiting.')
                            return
            except Exception:
                pass

    base_path = f'{base_path}/{cur_id:04d}_{uuid.uuid4().hex}/'
    args.base_path = base_path
    os.makedirs(base_path, exist_ok=True)
    open(f'{base_path}/thisisfake_{args.max_compute_steps}_experience.bz2',
         'w')
    info = copy.copy(vars(args))
    info['version'] = VERSION
    info['code_hash'] = get_code_hash()
    print('Code hash:', info['code_hash'])
    del info['base_path']
    json.dump(info,
              open(base_path + '/kwargs.json', 'w'),
              sort_keys=True,
              indent=2)

    code_path = base_path + '/code'
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    shutil.copytree(cur_dir,
                    code_path,
                    ignore=shutil.ignore_patterns('*.png', '*.stl', '*.JPG',
                                                  '__pycache__', 'LICENSE*',
                                                  'README*'))

    teeout = Tee(args.base_path + '/log.out', 'stdout')
    teeerr = Tee(args.base_path + '/log.err', 'stderr')

    print('Experiment running in', base_path)

    try:
        _run(base_path, args)
    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()
        import signal
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            os.kill(child.pid, signal.SIGTERM)
        open(base_path + 'has_died', 'w')
        os._exit(1)


if __name__ == '__main__':
    if os.path.exists('/home/udocker/deeplearning_goexplore_adrienle/'):
        os.makedirs('/mnt/phx4', exist_ok=True)
        os.system(
            '/opt/michelangelo/mount.nfs -n -o nolock qstore1-phx4:/share /mnt/phx4'
        )
    parser = argparse.ArgumentParser()
    current_group = parser

    def boolarg(arg, *args, default=False, help='', neg=None, dest=None):

        def extract_name(a):
            dashes = ''
            while a[0] == '-':
                dashes += '-'
                a = a[1:]
            return dashes, a

        if dest is None:
            _, dest = extract_name(arg)

        group = current_group.add_mutually_exclusive_group()
        group.add_argument(arg,
                           *args,
                           dest=dest,
                           action='store_true',
                           help=help + (' (DEFAULT)' if default else ''),
                           default=default)
        not_args = []
        for a in [arg] + list(args):
            dashes, name = extract_name(a)
            not_args.append(f'{dashes}no_{name}')
        if isinstance(neg, str):
            not_args[0] = neg
        if isinstance(neg, list):
            not_args = neg
        group.add_argument(*not_args,
                           dest=dest,
                           action='store_false',
                           help=f'Opposite of {arg}' +
                           (' (DEFAULT)' if not default else ''),
                           default=default)

    def add_argument(*args, **kwargs):
        if 'help' in kwargs and kwargs.get('default') is not None:
            kwargs['help'] += f' (default: {kwargs.get("default")})'
        current_group.add_argument(*args, **kwargs)

    current_group = parser.add_argument_group('General Go-Explore')
    add_argument('--game',
                 '-g',
                 type=str,
                 default='montezuma',
                 help='Determines the game to which apply goexplore.')
    add_argument(
        '--repeat_action',
        '--ra',
        type=float,
        default=20,
        help=
        'The average number of times that actions will be repeated in the exploration phase.'
    )
    add_argument('--explore_steps',
                 type=int,
                 default=100,
                 help='Maximum number of steps in the explore phase.')
    boolarg(
        '--optimize_score',
        default=True,
        help=
        'Optimize for score (only speed). Will use fewer "game frames" and come up with faster trajectories with lower scores. If not combined with --remember_rooms and --objects_from_ram is not enabled, things should run much slower.'
    )
    add_argument('--batch_size',
                 type=int,
                 default=100,
                 help='Number of worker threads to spawn')
    boolarg(
        '--reset_cell_on_update',
        '--rcou',
        help=
        'Reset the times-chosen and times-chosen-since when a cell is updated.'
    )
    add_argument('--explorer_type',
                 type=str,
                 default='repeated',
                 help='The type of explorer. repeated, drift or random.')
    add_argument('--seed', type=int, default=None, help='The random seed.')

    current_group = parser.add_argument_group('Checkpointing')
    add_argument('--base_path',
                 '-p',
                 type=str,
                 default='./results/',
                 help='Folder in which to store results')
    add_argument('--path_postfix',
                 '--pf',
                 type=str,
                 default='',
                 help='String appended to the base path.')
    add_argument(
        '--checkpoint_game',
        type=int,
        default=20_000_000_000_000_000_000,
        help=
        'Save a checkpoint every this many GAME frames (note: recommmended to ignore, since this grows very fast at the end).'
    )
    add_argument('--checkpoint_compute',
                 type=int,
                 default=1_000_000,
                 help='Save a checkpoint every this many COMPUTE frames.')
    boolarg(
        '--clear_old_checkpoints',
        neg='--keep_checkpoints',
        default=True,
        help=
        'Clear large format checkpoints. Checkpoints aren\'t necessary for view folder to work. They use a lot of space.'
    )

    current_group = parser.add_argument_group('Runtime')
    add_argument('--max_game_steps',
                 type=int,
                 default=None,
                 help='Maximum number of GAME frames.')
    add_argument('--max_compute_steps',
                 '--mcs',
                 type=int,
                 default=None,
                 help='Maximum number of COMPUTE frames.')
    add_argument('--max_iterations',
                 type=int,
                 default=None,
                 help='Maximum number of iterations.')
    add_argument('--max_hours',
                 '--mh',
                 type=float,
                 default=12,
                 help='Maximum number of hours to run this for.')
    add_argument('--max_cells',
                 type=int,
                 default=None,
                 help='The maximum number of cells before stopping.')
    add_argument(
        '--max_score',
        type=float,
        default=None,
        help='Stop when this score (or more) has been reached in the archive.')

    current_group = parser.add_argument_group('General Selection Probability')
    add_argument('--seen_weight',
                 '--sw',
                 type=float,
                 default=0.0,
                 help='The weight of the "seen" attribute in cell selection.')
    add_argument('--seen_power',
                 '--sp',
                 type=float,
                 default=0.5,
                 help='The power of the "seen" attribute in cell selection.')

    boolarg(
        '--dynamic_state',
        help=
        'Dynamic downscaling of states. Ignores --resize_x, --resize_y, --max_pix_value and --resize_shape.'
    )

    add_argument(
        '--first_compute_dynamic_state',
        type=int,
        default=100_000,
        help=
        'Number of steps before recomputing the dynamic state representation (ignored if negative).'
    )
    add_argument(
        '--first_compute_archive_size',
        type=int,
        default=10_000,
        help=
        'Number of steps before recomputing the dynamic state representation (ignored if negative).'
    )
    add_argument(
        '--recompute_dynamic_state_every',
        type=int,
        default=5_000_000,
        help=
        'Number of steps before recomputing the dynamic state representation (ignored if negative).'
    )
    add_argument(
        '--max_archive_size',
        type=int,
        default=1_000_000_000,
        help=
        'Number of steps before recomputing the dynamic state representation (ignored if negative).'
    )

    add_argument(
        '--cell_split_factor',
        type=float,
        default=0.03,
        help=
        'The factor by which we try to split frames when recomputing the representation. 1 -> each frame is its own cell. 0 -> all frames are in the same cell.'
    )
    add_argument(
        '--split_iterations',
        type=int,
        default=100,
        help=
        'The number of iterations when recomputing the representation. A higher number means a more accurate (but less stochastic) results, and a lower number means a more stochastic and less accurate result. Note that stochasticity can be a good thing here as it makes it harder to get stuck.'
    )
    add_argument(
        '--max_recent_frames',
        type=int,
        default=5_000,
        help=
        'The number of recent frames to use in recomputing the representation. A higher number means slower recomputation but more accuracy, a lower number is faster and more stochastic.'
    )
    add_argument(
        '--recent_frame_add_prob',
        type=float,
        default=0.1,
        help=
        'The probability for a frame to be added to the list of recent frames.'
    )

    current_group = parser.add_argument_group('Performance')
    add_argument('--n_cpus',
                 type=int,
                 default=None,
                 help='Number of worker threads to spawn')
    add_argument('--pool_class',
                 type=str,
                 default='loky',
                 help='The multiprocessing pool class (py or torch or loky).')
    add_argument('--start_method',
                 type=str,
                 default='fork',
                 help='The process start method.')
    boolarg('--reset_pool',
            help='The pool should be reset every 100 iterations.')
    boolarg('--profile', help='Whether or not to enable a profiler.')

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed + 1)

    THRESH_TRUE = args.checkpoint_game
    THRESH_COMPUTE = args.checkpoint_compute
    MAX_FRAMES = args.max_game_steps
    MAX_FRAMES_COMPUTE = args.max_compute_steps
    MAX_TIME = args.max_hours * 3600
    MAX_ITERATIONS = args.max_iterations
    MAX_CELLS = args.max_cells
    MAX_SCORE = args.max_score

    try:
        run(args.base_path, args)
    finally:
        pass
