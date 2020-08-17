import numpy as np
import re, itertools, copy
import scipy.ndimage
from src.params import *
from fractions import Fraction

class Board:
    def __init__(self, size=[0, 0]):
        self.names = ['', '', '']
        self.params = {'R': DEF_R, 'T': 10, 'b': [1, 0, 0], 'm': 0.1, 's': 0.01, 'kn': 1, 'gn': 1}
        self.param_P = 0
        self.cells = np.zeros(size)

    @classmethod
    def from_values(cls, cells, params=None, names=None):
        self = cls()
        self.names = names.copy() if names is not None else None
        self.params = params.copy() if params is not None else None
        self.cells = cells.copy() if cells is not None else None
        return self

    @classmethod
    def from_data(cls, data):
        self = cls()
        self.names = [data.get('code', ''), data.get('name', ''), data.get('cname', '')]
        self.params = data.get('params')
        if self.params:
            self.params = self.params.copy()
            self.params['b'] = Board.st2fracs(self.params['b'])
        self.cells = data.get('cells')
        if self.cells:
            if type(self.cells) in [tuple, list]:
                self.cells = ''.join(self.cells)
            self.cells = Board.rle2arr(self.cells)
        return self

    def to_data(self, is_shorten=True):
        rle_st = Board.arr2rle(self.cells, is_shorten)
        params2 = self.params.copy()
        params2['b'] = Board.fracs2st(params2['b'])
        data = {'code': self.names[0], 'name': self.names[1], 'cname': self.names[2], 'params': params2, 'cells': rle_st}
        return data

    def params2st(self):
        params2 = self.params.copy()
        params2['b'] = '[' + Board.fracs2st(params2['b']) + ']'
        return ','.join(['{}={}'.format(k, str(v)) for (k, v) in params2.items()])

    def long_name(self):
        # return ' | '.join(filter(None, self.names))
        return '{0} - {1} {2}'.format(*self.names)

    @staticmethod
    def arr2rle(A, is_shorten=True):
        ''' RLE = Run-length encoding:
            http://www.conwaylife.com/w/index.php?title=Run_Length_Encoded
            http://golly.sourceforge.net/Help/formats.html#rle
            https://www.rosettacode.org/wiki/Run-length_encoding#src
            0=b=.  1=o=A  1-24=A-X  25-48=pA-pX  49-72=qA-qX  241-255=yA-yO '''
        V = np.rint(A * 255).astype(int).tolist()  # [[255 255] [255 0]]
        code_arr = [[' .' if v == 0 else ' ' + chr(ord('A') + v - 1) if v < 25 else chr(ord('p') + (v - 25) // 24) + chr(ord('A') + (v - 25) % 24) for v in row] for row in
                    V]  # [[yO yO] [yO .]]
        if is_shorten:
            rle_groups = [[(len(list(g)), c.strip()) for c, g in itertools.groupby(row)] for row in code_arr]  # [[(2 yO)] [(1 yO) (1 .)]]
            for row in rle_groups:
                if row[-1][1] == '.': row.pop()  # [[(2 yO)] [(1 yO)]]
            st = '$'.join(''.join([(str(n) if n > 1 else '') + c for n, c in row]) for row in rle_groups) + '!'  # "2 yO $ 1 yO"
        else:
            st = '$'.join(''.join(row) for row in code_arr) + '!'
        # print(sum(sum(r) for r in V))
        return st

    @staticmethod
    def rle2arr(st):
        rle_groups = re.findall('(\d*)([p-y]?[.boA-X$])', st.rstrip('!'))  # [(2 yO)(1 $)(1 yO)]
        code_list = sum([[c] * (1 if n == '' else int(n)) for n, c in rle_groups], [])  # [yO yO $ yO]
        code_arr = [l.split(',') for l in ','.join(code_list).split('$')]  # [[yO yO] [yO]]
        V = [[0 if c in ['.', 'b'] else 255 if c == 'o' else ord(c) - ord('A') + 1 if len(c) == 1 else (ord(c[0]) - ord('p')) * 24 + (ord(c[1]) - ord('A') + 25) for c in row if
              c != ''] for row in code_arr]  # [[255 255] [255]]
        # lines = st.rstrip('!').split('$')
        # rle = [re.findall('(\d*)([p-y]?[.boA-X])', row) for row in lines]
        # code = [ sum([[c] * (1 if n=='' else int(n)) for n,c in row], []) for row in rle]
        # V = [ [0 if c in ['.','b'] else 255 if c=='o' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row ] for row in code]
        maxlen = len(max(V, key=len))
        A = np.array([row + [0] * (maxlen - len(row)) for row in V]) / 255  # [[1 1] [1 0]]
        # print(sum(sum(r) for r in V))
        return A

    @staticmethod
    def fracs2st(B):
        return ','.join([str(f) for f in B])

    @staticmethod
    def st2fracs(st):
        return [Fraction(st) for st in st.split(',')]

    def clear(self):
        self.cells.fill(0)

    def add(self, part, shift=[0, 0]):
        # assert self.params['R'] == part.params['R']
        h1, w1 = self.cells.shape
        h2, w2 = part.cells.shape
        h, w = min(h1, h2), min(w1, w2)
        i1, j1 = (w1 - w) // 2 + shift[1], (h1 - h) // 2 + shift[0]
        i2, j2 = (w2 - w) // 2, (h2 - h) // 2
        # self.cells[j:j+h, i:i+w] = part.cells[0:h, 0:w]
        vmin = np.amin(part.cells)
        for y in range(h):
            for x in range(w):
                if part.cells[j2 + y, i2 + x] > vmin:
                    self.cells[(j1 + y) % h1, (i1 + x) % w1] = part.cells[j2 + y, i2 + x]
        return self

    def transform(self, tx, mode='RZSF', is_world=False):
        if 'R' in mode and tx['rotate'] != 0:
            self.cells = scipy.ndimage.rotate(self.cells, tx['rotate'], reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')
        if 'Z' in mode and tx['R'] != self.params['R']:
            # print('* {} / {}'.format(tx['R'], self.params['R']))
            shape_orig = self.cells.shape
            self.cells = scipy.ndimage.zoom(self.cells, tx['R'] / self.params['R'], order=0)
            if is_world:
                self.cells = Board(shape_orig).add(self).cells
            self.params['R'] = tx['R']
        if 'F' in mode and tx['flip'] != -1:
            if tx['flip'] in [0, 1]:
                self.cells = np.flip(self.cells, axis=tx['flip'])
            elif tx['flip'] == 2:
                self.cells[:, :-MIDX - 1:-1] = self.cells[:, :MIDX]
            elif tx['flip'] == 3:
                self.cells[:, :-MIDX - 1:-1] = self.cells[::-1, :MIDX]
            elif tx['flip'] == 4:
                i_upper = np.triu_indices(SIZEX, -1); self.cells[i_upper] = self.cells.T[i_upper]
        if 'S' in mode and tx['shift'] != [0, 0]:
            self.cells = scipy.ndimage.shift(self.cells, tx['shift'], order=0, mode='wrap')
        # self.cells = np.roll(self.cells, tx['shift'], (1, 0))
        return self

    def add_transformed(self, part, tx):
        part = copy.deepcopy(part)
        self.add(part.transform(tx, mode='RZF'), tx['shift'])
        return self

    def crop(self):
        vmin = np.amin(self.cells)
        coords = np.argwhere(self.cells > vmin)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        self.cells = self.cells[y0:y1, x0:x1]
        return self

    def restore_to(self, dest):
        dest.params = self.params.copy()
        dest.cells = self.cells.copy()
        dest.names = self.names.copy()

