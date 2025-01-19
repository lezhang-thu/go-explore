# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.insert(0, '/home/ubuntu/lezhang.thu/sac-go-explore/im-go-explore/go-explore/robustified')
print(sys.path)
import argparse
import copy
import glob
import json
import shutil
import psutil
import time
import uuid
import logging
import multiprocessing

import numpy as np
from tqdm import tqdm

from goexplore_py.randselectors import Weight, WeightedSelector
from goexplore_py.explorers import RepeatedRandomExplorer
from goexplore_py.goexplore import Explore, LPool, seed_pool_wrapper, DONE
import goexplore_py.generic_atari_env as generic_atari_env
from goexplore_py.utils import get_code_hash
from goexplore_py.testbed.normal_ppo import sac

VERSION = 1

MAX_FRAMES = None
MAX_FRAMES_COMPUTE = None
MAX_CELLS = None
MAX_SCORE = None


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
    logger = setup_logging('output', '{}.txt'.format('None'))
    communicate_queue = multiprocessing.Queue(100)
    sac_process = multiprocessing.Process(target=sac, args=(communicate_queue, args.game.split('_')[1])) 
    sac_process.start()

    def should_continue():
        if ((MAX_FRAMES is not None and expl.frames_true >= MAX_FRAMES)
                or (MAX_FRAMES_COMPUTE is not None
                    and expl.frames_compute >= MAX_FRAMES_COMPUTE)
                or (MAX_CELLS is not None and len(expl.grid) >= MAX_CELLS)
                or (MAX_SCORE is not None and expl.max_score >= MAX_SCORE)):
            return False
        return True

    t_compute = 0
    while should_continue():
        # Run one iteration
        expl.run_cycle(communicate_queue)

        if expl.frames_compute - t_compute > int(
                1e6) or expl.frames_compute >= MAX_FRAMES_COMPUTE:
            t_compute = expl.frames_compute
            logger.info('Compute steps: {}'.format(expl.frames_compute))
            logger.info('Game step: {}'.format(expl.frames_true))
            logger.info('Max score {}'.format(expl.max_score))
            logger.info('Done score: {}'.format(expl.grid[DONE].score))
            logger.info('Cells: {}'.format(len(expl.grid)))


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

    MAX_FRAMES = args.max_game_steps
    MAX_FRAMES_COMPUTE = args.max_compute_steps
    MAX_CELLS = args.max_cells
    MAX_SCORE = args.max_score

    try:
        run(args.base_path, args)
    finally:
        pass
