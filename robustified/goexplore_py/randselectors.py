# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

from .import_ai import *
from goexplore_py.goexplore import DONE


@dataclass()
class Weight:
    weight: float = 1.0
    power: float = 1.0

    def __repr__(self):
        return f'w={self.weight:.2f}=p={self.power:.2f}'


def numberOfSetBits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


def convert_score(e):
    # TODO: this doesn't work when actual score is used!! Fix?
    if isinstance(e, tuple):
        return len(e)
    return numberOfSetBits(e)


class WeightedSelector:

    def __init__(self, game, seen=Weight(0.1)):
        self.seen: Weight = seen
        self.game = game

        self.clear_all_cache()

    def clear_all_cache(self):
        self.all_weights = []
        self.to_choose_idxs = []
        self.cells = []
        self.all_weights_nparray = None
        self.cell_pos = {}
        self.cell_score = {}
        self.possible_scores = defaultdict(int)
        self.to_update = set()
        self.update_all = False

    def get_score(self, cell_key, cell):
        if cell_key == DONE:
            return 0.0
        else:
            return cell.score

    def cell_update(self, cell_key, cell):
        prev_possible_scores = len(self.possible_scores)
        is_new = cell_key not in self.cell_pos
        if is_new:
            self.cell_pos[cell_key] = len(self.all_weights)
            self.all_weights.append(0.0)
            self.cells.append(cell_key)
            self.to_choose_idxs.append(len(self.to_choose_idxs))
            self.all_weights_nparray = None

            if cell_key != DONE:
                self.cell_score[cell_key] = self.get_score(cell_key, cell)
                self.possible_scores[self.get_score(cell_key, cell)] += 1
        elif cell_key != DONE:
            score = self.get_score(cell_key, cell)
            old_score = self.cell_score[cell_key]
            self.possible_scores[score] += 1
            self.possible_scores[old_score] -= 1
            self.cell_score[cell_key] = score
            if self.possible_scores[old_score] == 0:
                del self.possible_scores[old_score]
        self.to_update.add(cell_key)

    def compute_weight(self, value, weight):
        return weight.weight * 1 / (value + 1)**weight.power

    def get_seen_weight(self, cell):
        return self.compute_weight(cell.seen_times, self.seen)

    def get_weight(self, cell_key, cell, possible_scores, known_cells):
        if cell_key == DONE:
            return 0.0
        return self.get_seen_weight(cell)

    def update_weights(self, known_cells):
        if len(known_cells) == 0:
            return

        if self.update_all:
            to_update = self.cells
        else:
            to_update = self.to_update

        for example_key in known_cells:
            if example_key is not None:
                break

        possible_scores = sorted(self.possible_scores,
                                 key=((lambda x: x) if isinstance(
                                     example_key, tuple) else convert_score))
        for cell in to_update:
            idx = self.cell_pos[cell]
            self.all_weights[idx] = self.get_weight(cell, known_cells[cell],
                                                    possible_scores,
                                                    known_cells)
            if self.all_weights_nparray is not None:
                self.all_weights_nparray[idx] = self.all_weights[idx]

        self.update_all = False
        self.to_update = set()

    def choose_cell(self, known_cells, size=1):
        self.update_weights(known_cells)
        if len(known_cells) != len(self.all_weights):
            print(
                'ERROR, known_cells has a different number of cells than all_weights'
            )
            print(
                f'Cell numbers: known_cells {len(known_cells)}, all_weights {len(self.all_weights)}, to_choose_idx {len(self.to_choose_idxs)}, cell_pos {len(self.cell_pos)}'
            )
            for c in known_cells:
                if c not in self.cell_pos:
                    print(f'Tracked but unknown cell: {c}')
            for c in self.cell_pos:
                if c not in known_cells:
                    print(f'Untracked cell: {c}')
            assert False, 'Incorrect length stuff'

        if len(self.cells) == 1:
            return [self.cells[0]] * size
        if self.all_weights_nparray is None:
            self.all_weights_nparray = np.array(self.all_weights)
        weights = self.all_weights_nparray
        to_choose = self.cells
        total = np.sum(weights)
        idxs = np.random.choice(self.to_choose_idxs,
                                size=size,
                                p=weights / total)
        # TODO: in extremely rare cases, we do select the DONE cell. Not sure why. We filter it out here but should
        # try to fix the underlying bug eventually.
        return [to_choose[i] for i in idxs if to_choose[i] != DONE]

    def __repr__(self):
        return f'weight-seen-{self.seen}-chosen-{self.chosen}-chosen-since-new-{self.chosen_since_new_weight}-action-{self.action}-room-{self.room_cells}-dir-{self.dir_weights}'
