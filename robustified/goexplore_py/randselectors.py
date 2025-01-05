# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.

# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from dataclasses import dataclass
from goexplore_py.goexplore import DONE


@dataclass()
class Weight:
    weight: float = 1.0
    power: float = 1.0

    def __repr__(self):
        return f'w={self.weight:.2f} p={self.power:.2f}'


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
        self.to_update = set()

    def cell_update(self, cell_key, cell):
        if cell_key not in self.cell_pos:
            self.cell_pos[cell_key] = len(self.all_weights)
            self.all_weights.append(0.0)
            self.cells.append(cell_key)
            self.to_choose_idxs.append(len(self.to_choose_idxs))
            self.all_weights_nparray = None
        self.to_update.add(cell_key)

    def get_weight(self, cell_key, cell):
        return 0.0 if cell_key == DONE else self.seen.weight * 1 / (
            cell.seen_times + 1)**self.seen.power

    def update_weights(self, known_cells):
        if len(known_cells) == 0:
            return
        for cell in self.to_update:
            idx = self.cell_pos[cell]
            self.all_weights[idx] = self.get_weight(cell, known_cells[cell])
            if self.all_weights_nparray is not None:
                self.all_weights_nparray[idx] = self.all_weights[idx]
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
        return f'weight-seen-{self.seen}'
