import numpy as np  # pip3 install numpy
import reikna.fft, reikna.cluda  # pip3 install pyopencl/pycuda, reikna
import PIL.Image, PIL.ImageTk  # pip3 install pillow
import PIL.ImageDraw, PIL.ImageFont
from src.board import Board
from src.automaton import Automaton
from src.analyzer import Analyzer
from src.recorder import Recorder
try:
    import tkinter as tk
except:
    import Tkinter as tk
from fractions import Fraction
import copy, json, csv
import io, time
from src.params import *
from audio import *
from src.tools.setup_mic import MicStream
import matplotlib
HERE_PATH = os.getcwd() +'/'
CLASSIF_PATH = os.getcwd() + "/data/classif/"

AUDIO_CONTROL = True

class Lenia:
    MARKER_COLORS_W = [0x5F, 0x5F, 0x5F, 0x7F, 0x7F, 0x7F, 0xFF, 0xFF, 0xFF]
    MARKER_COLORS_B = [0x9F, 0x9F, 0x9F, 0x7F, 0x7F, 0x7F, 0x0F, 0x0F, 0x0F]
    SAVE_ROOT = 'save'

    def __init__(self):
        self.is_run = True
        self.run_counter = -1
        self.show_freq = 1
        self.is_closing = False
        self.show_what = 0
        self.markers_mode = 0
        self.stats_mode = 0
        self.stats_x = 4
        self.stats_y = 5
        self.is_group_params = False
        self.is_show_fps = False
        self.fps = None
        self.last_time = None
        self.fore = None
        self.back = None
        self.is_layered = False
        self.is_auto_center = False
        self.is_auto_rotate = False
        self.is_auto_load = False
        self.trace_m = None
        self.trace_s = None
        self.last_s_max = None
        self.last_s_min = None
        self.trace_library = {}
        self.search_dir = None
        self.is_search_small = False
        self.is_empty = False

        ''' http://hslpicker.com/ '''
        cmaps = [matplotlib.cm.get_cmap('cool'),
                 matplotlib.cm.get_cmap('PiYG'),
                 matplotlib.cm.get_cmap('hsv'),
                 matplotlib.cm.get_cmap('gnuplot2'),
                 matplotlib.cm.get_cmap('turbo'),
                 matplotlib.cm.get_cmap('rainbow'),
                 matplotlib.cm.get_cmap('RdPu'),
                 matplotlib.cm.get_cmap('magma'),
                 matplotlib.cm.get_cmap('inferno'),
                 matplotlib.cm.get_cmap('Spectral')
                 ]

        self.colormaps = []
        for c in cmaps:
            self.colormaps.append(self.create_colormap(np.array([c(i) for i in np.arange(0, 1, 1/256)])[:, :3] * 8))
        self.colormaps += [
            self.create_colormap(np.array([[0, 188, 212], [178, 235, 242], [255, 87, 34], [221, 44, 0]]) / 255 * 8),
            self.create_colormap(np.array([[0, 0, 4], [0, 0, 8], [0, 4, 8], [0, 8, 8], [4, 8, 4], [8, 8, 0], [8, 4, 0], [8, 0, 0], [4, 0, 0]])),  # BCYR
            self.create_colormap(np.array([[0, 2, 0], [0, 4, 0], [4, 6, 0], [8, 8, 0], [8, 4, 4], [8, 0, 8], [4, 0, 8], [0, 0, 8], [0, 0, 4]])),  # GYPB
            self.create_colormap(np.array([[4, 0, 2], [8, 0, 4], [8, 0, 6], [8, 0, 8], [4, 4, 4], [0, 8, 0], [0, 6, 0], [0, 4, 0], [0, 2, 0]])),  # PPGG
            self.create_colormap(np.array([[4, 4, 6], [2, 2, 4], [2, 4, 2], [4, 6, 4], [6, 6, 4], [4, 2, 2]])),  # BGYR
            self.create_colormap(np.array([[4, 6, 4], [2, 4, 2], [4, 4, 2], [6, 6, 4], [6, 4, 6], [2, 2, 4]])),  # GYPB
            self.create_colormap(np.array([[6, 6, 4], [4, 4, 2], [4, 2, 4], [6, 4, 6], [4, 6, 6], [2, 4, 2]])),  # YPCG
            # self.create_colormap(np.array([[0,0,0],[3,3,3],[4,4,4],[5,5,5],[8,8,8]]))] #B/W
            self.create_colormap(np.array([[8, 8, 8], [7, 7, 7], [5, 5, 5], [3, 3, 3], [0, 0, 0]]), is_marker_w=False),  # W/B
            self.create_colormap(np.array([[0, 0, 0], [3, 3, 3], [5, 5, 5], [7, 7, 7], [8, 8, 8]]))]  # B/W
        self.colormap_id = 0

        self.last_key = None
        self.excess_key = None
        self.info_type = None
        self.clear_job = None
        self.is_save_image = False
        self.file_seq = 0
        self.ang_speed = 0
        self.is_ang_clockwise = False
        self.ang_sides = 1
        self.ang_gen = 1

        self.read_animals()
        self.world = Board((SIZEY, SIZEX))

        self.knob_inits = dict(zip(knobs_keys, [64] * len(knobs_keys)))
        self.current_angle = 0
        self.reset_controls()

        # self.target = Board((SIZEY, SIZEX))
        self.automaton = Automaton(self.world)
        self.analyzer = Analyzer(self.automaton)
        self.recorder = Recorder(self.world)
        self.clear_transform()
        self.create_window()
        # self.signals = SIGNALS
        if AUDIO_CONTROL:
            self.mic_stream = MicStream()
        self.real_time = 0
        self.time_since_save = 0
        self.img_count = 0
        self.audio_bin_count = 0

    def get_controls(self, knob_value, value, end_min, end_max, big=True):
        if value > end_max or value < end_min:
            print('RANGE', end_min, value, end_max)
        end_max = max(end_max, value)
        end_min = min(end_min, value)
        assert knob_value >= 0 and knob_value < 128
        if not big:
            end_max = value + (end_max - value) / 2
            end_min = value - (value - end_min) / 2
        if knob_value == 0:
            low_range = np.array([value])
        else:
            low_delta = (value - end_min) / knob_value
            if low_delta < 1e-5:
                low_range = np.zeros([knob_value + 1])
            else:
                low_range = np.arange(end_min, value + low_delta / 2, low_delta)
        if knob_value == 127:
            high_range = np.array([value])
        else:
            high_delta = (end_max - value) / (127 - knob_value)
            if high_delta < 1e-5:
                high_range = np.zeros([127 - knob_value])
            else:
                high_range = np.arange(value, end_max, high_delta)
        out =  np.concatenate([low_range, high_range], axis=0)
        return out[:128]

    def reset_controls(self):
        self.kernel_id = 0
        self.small_controls = dict()
        self.big_controls = dict()
        self.initial_world_params = copy.deepcopy(self.world.params)
        for k in knobs_keys:
            # print(k)
            end_min, end_max = RANGES[k]
            if k == 'T':
                current_value = self.world.params['T']
            elif k == 'R':
                current_value = self.world.params['R'][0]
            elif k == 'k_mu':
                current_value = self.world.params['m'][0]
            elif k == 'k_s':
                current_value = self.world.params['s'][0]
            elif 'beta' in k:
                i = int(k.split('_')[1])
                if len(self.world.params['b'][0]) > i:
                    current_value = self.world.params['b'][0][i]
                else:
                    current_value = 0
            elif k == 'angle':
                current_value = 0
            # current_value = self.world.params[k] if k == 'T' else self.world.params[k][0]
            self.small_controls[k] = self.get_controls(knob_value=self.knob_inits[k],
                                                       value=current_value,
                                                       end_min=end_min,
                                                       end_max=end_max,
                                                       big=False)
            self.big_controls[k] = self.get_controls(knob_value=self.knob_inits[k],
                                                     value=current_value,
                                                     end_min=end_min,
                                                     end_max=end_max,
                                                     big=True)
            assert self.small_controls[k][self.knob_inits[k]] - current_value < 1e-8
            assert self.big_controls[k][self.knob_inits[k]] - current_value < 1e-8
            assert self.small_controls[k].size == 128
            assert self.big_controls[k].size == 128


# delta = KNOBS_DELTAS[k]
# self.big_controls[k] = np.arange(- delta * self.knob_inits[k], delta * (127 - self.knob_inits[k]) + delta / 2, delta)
# delta /= 10
            # self.small_controls[k] = np.arange(- delta * self.knob_inits[k], delta * (127 - self.knob_inits[k]) + delta / 2, delta)
            # assert self.big_controls[k][self.knob_inits[k]] < 1e-10
            # assert self.small_controls[k][self.knob_inits[k]] < 1e-10
        self.controls = self.small_controls
        self.small = True


    def update_midi_controls(self, controls):
        for control, value in controls.items():
            # print(controls)
            if control in KNOBS_REFS_TO_KEYS.keys():
                key = KNOBS_REFS_TO_KEYS[control]
                self.knob_inits[key] = value
                if key == 'k_mu':
                    self.world.params['m'][self.kernel_id] = self.controls[key][value]#self.initial_world_params['m'][self.kernel_id] + self.controls[key][value]
                    print('K{}, Setting mean to: {}'.format(self.kernel_id, self.world.params['m'][self.kernel_id]))
                elif key == 'k_s':
                    self.world.params['s'][self.kernel_id] = self.controls[key][value]#self.initial_world_params['s'][self.kernel_id] + self.controls[key][value]
                    print('K{}, Setting std to: {}'.format(self.kernel_id,self.world.params['s'][self.kernel_id]))
                elif key == 'T':
                    self.world.params['T'] = self.controls[key][value]#max(self.initial_world_params['T'] + self.controls[key][value], 1)
                    print('Setting T to: {}'.format(self.world.params['T'] ))
                elif key == 'beta_1':
                    self.world.params['b'][self.kernel_id][0] = self.controls[key][value]# max(self.initial_world_params['b'][self.kernel_id][0] + self.controls[key][value], 0)
                    print('K{}, Setting beta 1 to: {}'.format(self.kernel_id, self.world.params['b'][self.kernel_id][0]))
                    print('K{}, Beta', self.world.params['b'][self.kernel_id])
                elif key == 'beta_2':
                    while len(self.world.params['b'][self.kernel_id]) < 2:
                        self.world.params['b'][self.kernel_id].append(0)
                    while len(self.initial_world_params['b'][self.kernel_id]) < 4:
                        self.initial_world_params['b'][self.kernel_id].append(0)
                    self.world.params['b'][self.kernel_id][1] = self.controls[key][value]#max(self.initial_world_params['b'][self.kernel_id][1] + self.controls[key][value], 0)
                    print('K{}, Setting beta 2 to: {}'.format(self.kernel_id, self.world.params['b'][self.kernel_id][1]))
                    print('K{}, Beta'.format(self.kernel_id), self.world.params['b'][self.kernel_id])
                elif key == 'beta_3':
                    while len(self.world.params['b'][self.kernel_id]) < 3:
                        self.world.params['b'][self.kernel_id].append(0)
                    while len(self.initial_world_params['b'][self.kernel_id]) < 4:
                        self.initial_world_params['b'][self.kernel_id].append(0)
                    self.world.params['b'][self.kernel_id][2] = self.controls[key][value]# max(self.initial_world_params['b'][self.kernel_id][2] + self.controls[key][value], 0)
                    print('K{}, Setting beta 3 to: {}'.format(self.kernel_id, self.world.params['b'][self.kernel_id][2]))
                    print('K{}, Beta'.format(self.kernel_id), self.world.params['b'][self.kernel_id])
                elif key == 'beta_4':
                    while len(self.world.params['b'][self.kernel_id]) < 4:
                        self.world.params['b'][self.kernel_id].append(0)
                    while len(self.initial_world_params['b'][self.kernel_id]) < 4:
                        self.initial_world_params['b'][self.kernel_id].append(0)
                    self.world.params['b'][self.kernel_id][3] = self.controls[key][value]#max(self.initial_world_params['b'][self.kernel_id][3] + self.controls[key][value], 0)
                    print('K{}, Setting beta 3 to: {}'.format(self.kernel_id, self.world.params['b'][self.kernel_id][3]))
                    print('K{}, Beta'.format(self.kernel_id), self.world.params['b'][self.kernel_id])
                elif key == 'R':
                    # self.world.params['R'][self.kernel_id] = int(max(self.initial_world_params['R'][self.kernel_id] + self.controls[key][value], 5))
                    print("K{}, R = ".format(self.kernel_id), self.world.params['R'][self.kernel_id])
                    self.tx['R'][0] = self.controls[key][value]#int(max(self.initial_world_params['R'][self.kernel_id] + self.controls[key][value], 5))
                    # self.tx['R'][0] /= 2
                    self.transform_world()
                elif key == 'angle':
                    delta_angle = self.controls[key][value] - self.current_angle
                    self.tx['rotate'] = delta_angle
                    self.transform_world()
                    self.current_angle = self.controls[key][value]


            elif control in BUTTONS_REFS_TO_KEYS.keys() and value == 127:
                key = BUTTONS_REFS_TO_KEYS[control]
                if key == 'random_world':
                    self.random_world()
                elif key == 'random_animal':
                    # change animal to random
                    done = False
                    while not done:
                        self.load_animal_id(np.random.randint(len(self.animal_data)))
                        if self.world.params['kn'][0] == 1:
                            done = True
                elif key == 'colormap':
                    self.colormap_id = (self.colormap_id + 1) % len(self.colormaps)
                    print('Switching to colormap #{}'.format(self.colormap_id + 1))
                elif key == 'random_patch':
                    self.add_random_patch()
                elif key == 'flip1':
                    self.tx['flip'] = 1 if self.tx['flip'] != 0 else -1;
                    self.transform_world()
                elif key == 'flip2':
                    self.tx['flip'] = 2 if self.tx['flip'] != 0 else -1;
                    self.transform_world()
                elif key == 'flip3':
                    self.tx['flip'] = 3 if self.tx['flip'] != 0 else -1;
                    self.transform_world()
                elif key == 'flip4':
                    self.tx['flip'] = 4 if self.tx['flip'] != 0 else -1;
                    self.transform_world()
                elif key == 'k_left':
                    self.kernel_id = (self.kernel_id + 1) % self.world.nb_kernels
                    print('Controlling kernel #{}'.format(self.kernel_id + 1))
                elif key == 'k_right':
                    self.kernel_id = (self.kernel_id - 1) % self.world.nb_kernels
                    print('Controlling kernel #{}'.format(self.kernel_id + 1))
                elif key == 'k_more':
                    self.world.add_kernel()
                    for k in self.world.keys:
                        if k != 'T':
                            self.initial_world_params[k][self.world.nb_kernels - 1] = self.world.params[k][self.world.nb_kernels - 1]
                elif key == 'k_less':
                    self.world.remove_kernel()
                elif key == 'save_pos':
                    cells = self.world.cells
                    files = os.listdir(CLASSIF_PATH + 'pos/')
                    fs = [int(f.split('_')[-1].split('.')[0]) for f in files]
                    if len(fs) > 0:
                        ind = np.max(fs)
                    else:
                        ind = 0
                    np.savetxt(CLASSIF_PATH + 'pos/im_{}.txt'.format(ind + 1), cells)
                    self.img.save(CLASSIF_PATH + 'pos/im_{}.png'.format(ind + 1))
                    print('Added one pos example (#{})'.format(ind))
                elif key == 'save_neg':
                    cells = self.world.cells
                    files = os.listdir(CLASSIF_PATH + 'neg/')
                    fs = [int(f.split('_')[-1].split('.')[0]) for f in files]
                    if len(fs) > 0:
                        ind = np.max(fs)
                    else:
                        ind = 0
                    np.savetxt(CLASSIF_PATH + 'neg/im_{}.txt'.format(ind + 1), cells)
                    self.img.save(CLASSIF_PATH + 'neg/im_{}.png'.format(ind + 1))
                    print('Added one neg example (#{})'.format(ind))
                elif key == 'small/big':
                    if self.small is True:
                        self.small = False
                        self.controls = self.big_controls
                        msg = 'big'
                    else:
                        self.small = True
                        self.controls = self.small_controls
                        msg = 'small'
                    print('Now using {} controls'.format(msg))
        # if 20 in controls.keys() or 36 in controls.keys() or 56 in controls.keys():
        self.automaton.calc_once(is_update=False)
        self.automaton.calc_kernel()

        self.analyzer.new_segment()
        self.check_auto_load()

        if self.is_loop:
            self.roundup(self.world.params)
            self.roundup(self.tx)
            self.automaton.calc_once(is_update=False)
    # self.create_menu()

    def clear_transform(self):
        self.tx = {'shift': [0, 0], 'rotate': 0, 'R': self.world.params['R'], 'flip': -1}

    def read_animals(self):
        with open(HERE_PATH + 'animals.json', encoding='utf-8') as file:
            self.animal_data = json.load(file)

    def load_animal_id(self, id, **kwargs):
        self.world.reset_kernels()
        self.animal_id = max(0, min(len(self.animal_data) - 1, id))
        self.load_part(Board.from_data(self.animal_data[self.animal_id]), **kwargs)
        self.reset_controls()

    def load_animal_code(self, code, **kwargs):
        if not code: return
        id = self.get_animal_id(code)
        if id: self.load_animal_id(id, **kwargs)

    def get_animal_id(self, code):
        code_sp = code.split(':')
        n = int(code_sp[1]) if len(code_sp) == 2 else 1
        it = (id for (id, data) in enumerate(self.animal_data) if data["code"] == code_sp[0])
        for i in range(n):
            id = next(it, None)
        return id

    def load_part(self, part, is_replace=True, is_random=False, is_auto_load=False, repeat=1):
        self.fore = part
        if part.names is not None and part.names[0].startswith('~'):
            part.names[0] = part.names[0].lstrip('~')
            self.world.params['R'] = part.params['R']
            self.automaton.calc_kernel()
        if part.names is not None and is_replace:
            self.world.names = part.names.copy()
        if part.cells is not None:
            if part.params is None:
                part.params = self.world.params
            is_life = ((self.world.params.get('kn') or self.automaton.kn) == 4)
            will_be_life = ((part.params.get('kn') or self.automaton.kn) == 4)
            if not is_life and will_be_life:
                self.colormap_id = len(self.colormaps) - 1
                self.win.title('Conway\'s Game of Life')
            elif is_life and not will_be_life:
                self.colormap_id = 0
                self.world.params['R'] = DEF_R
                self.automaton.calc_kernel()
                self.win.title('Lenia')
            if self.is_layered:
                self.back = copy.deepcopy(self.world)
            if is_replace and not self.is_layered:
                if not is_auto_load:
                    self.world.params = {**part.params, 'R': self.world.params['R']}
                    self.automaton.calc_kernel()
                self.world.clear()
                self.automaton.reset()
                if not is_auto_load:
                    self.analyzer.reset()
            self.clear_transform()
            for i in range(repeat):
                if is_random:
                    self.tx['rotate'] = np.random.random() * 360
                    h1, w1 = self.world.cells.shape
                    h, w = min(part.cells.shape, self.world.cells.shape)
                    self.tx['shift'] = [np.random.randint(d1 + d) - d1 // 2 for (d, d1) in [(h, h1), (w, w1)]]
                    self.tx['flip'] = np.random.randint(3) - 1
                self.world.add_transformed(part, self.tx, kernel_id=0)

    def check_auto_load(self):
        if self.is_auto_load:
            self.load_part(self.fore, is_auto_load=True)
        else:
            self.automaton.reset()

    def transform_world(self):
        if self.is_layered:
            self.world.cells = self.back.cells.copy()
            self.world.params = self.back.params.copy()
            self.world.transform(self.tx, mode='Z', is_world=True)
            self.world.add_transformed(self.fore, self.tx)
        else:
            if not self.is_run:
                if self.back is None:
                    self.back = copy.deepcopy(self.world)
                else:
                    self.world.cells = self.back.cells.copy()
                    self.world.params = self.back.params.copy()
            self.world.transform(self.tx, is_world=True)
        self.automaton.calc_kernel()
        self.analyzer.reset_last()

    def world_updated(self):
        if self.is_layered:
            self.back = copy.deepcopy(self.world)
        self.automaton.reset()
        self.analyzer.reset()

    def clear_world(self):
        self.world.clear()
        self.world_updated()

    def random_world(self):
        border = int(self.world.params['R'][0])
        rand = np.random.rand(SIZEY - border * 2, SIZEX - border * 2)
        self.world.clear()
        self.world.add(Board.from_values(rand))
        self.world_updated()

    def add_random_patch(self):
        size = np.random.randint(0, SIZEX // 3, size=2)
        position = np.random.randint([0, 0], [SIZEX, SIZEY])
        board = np.zeros([SIZEX, SIZEY])
        size_x = min(SIZEX, position[0] + size[0]) - position[0]
        size_y = min(SIZEY, position[1] + size[1]) - position[1]
        rand = np.random.rand(size_x, size_y)
        board[position[0]:min(SIZEX, position[0] + size[0]),
        position[1]:min(SIZEY, position[1] + size[1])] = rand
        self.world.add(Board.from_values(board))
        self.world_updated()

    def toggle_trace(self, small):
        if self.trace_m == None:
            self.trace_m = +1
            self.trace_s = +1
            self.is_search_small = small
            self.is_auto_center = True
            self.stats_mode = 4
            self.stats_x = 4
            self.stats_y = 3
            self.is_group_params = True
            self.trace_library = {}
            self.load_part(self.fore)
            self.backup_m = self.backup_s = copy.deepcopy(self.world)
            self.automaton.reset()
            self.analyzer.reset()
            print('Trace start')
            print('Line right')
        else:
            self.finish_trace()
            print('Trace termined')

    def finish_trace(self):
        self.trace_m = None
        self.trace_s = None
        self.backup_m = self.backup_s = None
        self.stats_mode = 4
        self.show_freq = 1
        self.is_run = False

    def trace_params(self):
        ds = 0.0001 if self.is_search_small else 0.001
        dm = 0.001 if self.is_search_small else 0.01
        if self.analyzer.is_empty or self.analyzer.is_full:
            print('[X] ', self.world.params2st())
            self.analyzer.invalidate_segment()
            self.analyzer.calc_stats()
            self.automaton.reset()
            self.info_type = 'params'
            if self.trace_s == +1:
                self.trace_s = -1
                self.backup_s.restore_to(self.world)
                self.world.params['s'] += self.trace_s * ds
                self.roundup(self.world.params)
                print('Line left')
            elif self.trace_s == -1:
                self.trace_s = +2
                self.backup_s.restore_to(self.world)
                self.world.params['m'] += self.trace_m * dm
                self.roundup(self.world.params)
                print('Line up' if self.trace_m == +1 else 'Line down')
            elif self.trace_s in [+2, -2]:
                self.world.params['s'] += self.trace_s // 2 * ds
                self.roundup(self.world.params)
                m = round(self.world.params['m'] - self.trace_m * dm, ROUND)
                s = self.world.params['s']
                if (m, s) in self.trace_library:
                    data = self.trace_library[(m, s)]
                    self.world.clear()
                    self.world.add(Board.from_data(data))
                else:
                    if self.trace_s == +2:
                        self.trace_s = -2
                        self.backup_s.restore_to(self.world)
                        self.world.params['s'] += self.trace_s // 2 * ds
                        self.world.params['m'] += self.trace_m * dm
                        self.roundup(self.world.params)
                    elif self.trace_s == -2:
                        if self.trace_m == +1:
                            self.trace_m = -1
                            self.trace_s = +2
                            self.backup_m.restore_to(self.world)
                            self.world.params['m'] += self.trace_m * dm
                            self.roundup(self.world.params)
                            print('Line up' if self.trace_m == +1 else 'Line down')
                        elif self.trace_m == -1:
                            # print(len(self.trace_library))
                            # print(self.trace_library.keys())
                            # print(self.trace_library[self.backup_m.params['m'], self.backup_m.params['s']])
                            # print(len(self.analyzer.series))
                            self.finish_trace()
                            print('Trace finished')
        # print('>> ', self.trace_m, self.trace_s)
        elif self.automaton.gen == 800 // 2:
            self.analyzer.clear_segment()
        elif self.automaton.gen == 1000 // 2:
            print('[O] ', self.world.params2st())
            self.analyzer.new_segment()
            self.trace_library[(self.world.params['m'], self.world.params['s'])] = self.world.to_data()
            if self.trace_s in [+2, -2]:
                self.trace_s //= 2
                self.backup_s = copy.deepcopy(self.world)
            self.world.params['s'] += self.trace_s * ds
            self.roundup(self.world.params)
            self.automaton.reset()
            self.info_type = 'params'

    def toggle_search(self, dir, small):
        if self.search_dir == None:
            self.search_dir = dir
            self.is_search_small = small
            self.is_auto_center = True
            self.is_auto_load = True
        else:
            self.stop_search()

    def stop_search(self):
        self.search_dir = None

    def search_params(self):
        s = 's+' if self.is_search_small else ''
        if self.search_dir == +1:
            if self.analyzer.is_empty:
                self.key_press_internal(s + 'w')
            elif self.analyzer.is_full:
                self.key_press_internal(s + 'q')
        elif self.search_dir == -1:
            if self.analyzer.is_empty:
                self.key_press_internal(s + 'a')
            elif self.analyzer.is_full:
                self.key_press_internal(s + 's')

    def create_window(self):
        self.win = tk.Tk()
        self.win.title('Lenia')
        # icon = tk.Image("photo", file="lenia-logo.gif")
        # self.win.call('wm','iconphoto', self.win._w, icon)
        self.win.bind('<Key>', self.key_press_event)
        self.frame = tk.Frame(self.win, width=SIZEX * PIXEL, height=SIZEY * PIXEL)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, width=SIZEX * PIXEL, height=SIZEY * PIXEL)
        self.canvas.place(x=-1, y=-1)
        self.panel1 = self.create_panel(0, 0)
        # self.panel2 = self.create_panel(1, 0)
        # self.panel3 = self.create_panel(0, 1)
        # self.panel4 = self.create_panel(1, 1)
        self.info_bar = tk.Label(self.win)
        self.info_bar.pack()

    def create_panel(self, c, r):
        buffer = np.uint8(np.zeros((SIZEY * PIXEL, SIZEX * PIXEL)))
        img = PIL.Image.frombuffer('P', buffer.shape, buffer, 'raw', 'P', 0, 1)
        photo = PIL.ImageTk.PhotoImage(image=img)
        return self.canvas.create_image(c * SIZEY, r * SIZEX, image=photo, anchor=tk.NW)

    def create_colormap(self, colors, is_marker_w=True):
        nval = 253
        ncol = colors.shape[0]
        colors = np.vstack((colors, np.array([[0, 0, 0]])))
        v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 252 252 252]
        i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
        k = v / (nval - 1) * (ncol - 1)  # interpolate between 0 .. ncol-1
        k1 = k.astype(int)
        c1, c2 = colors[k1, i], colors[k1 + 1, i]
        c = (k - k1) * (c2 - c1) + c1  # interpolate between c1 .. c2
        return np.rint(c / 8 * 255).astype(int).tolist() + (self.MARKER_COLORS_W if is_marker_w else self.MARKER_COLORS_B)

    SHOW_WHAT_NUM = 5

    def update_win(self, show_arr=None):
        if self.show_what == 6:
            xgrad, ygrad = np.gradient(self.automaton.potential)
            grad = np.sqrt(xgrad ** 2 + ygrad ** 2) * self.world.params['R'] * 1.5
        # print(np.min(grad), np.max(grad))

        if show_arr is not None:
            self.draw_world(show_arr, 0, 1)
        elif self.stats_mode in [0, 1, 2]:
            change_range = 1 if not self.automaton.is_soft_clip else 1.4
            if self.show_what == 0:
                self.draw_world(self.world.cells, 0, 1, is_shift=True, is_shift_zero=True, markers=['world', 'arrow', 'scale', 'grid', 'colormap'])
            # if self.show_what==0: self.draw_world(self.analyzer.aaa, 0, 1, is_shift=True, is_shift_zero=True, markers=['world','arrow','scale','grid','colormap'])
            elif self.show_what == 1:
                self.draw_world(self.automaton.potential, 0, 2 * self.world.params['m'], is_shift=True, is_shift_zero=True, markers=['arrow', 'scale', 'grid', 'colormap'])
            elif self.show_what == 2:
                self.draw_world(self.automaton.field, -1, 1, is_shift=True, markers=['arrow', 'scale', 'grid', 'colormap'])
            elif self.show_what == 3:
                self.draw_world(self.automaton.kernel, 0, 1, markers=['scale', 'fixgrid', 'colormap'])
            elif self.show_what == 4:
                self.draw_world(self.automaton.fftshift(np.log(np.absolute(self.automaton.world_FFT))), 0, 5, markers=['colormap'])  # -10, 10
            elif self.show_what == 5:
                self.draw_world(self.automaton.fftshift(np.log(np.absolute(self.automaton.potential_FFT))), -20, 5, markers=['colormap'])  # -40, 10
            elif self.show_what == 6:
                self.draw_world(grad, 0, 1, is_shift=True, markers=['arrow', 'scale', 'grid', 'colormap'])
            elif self.show_what == 7:
                self.draw_world(self.automaton.change, -change_range, change_range, is_shift=True, markers=['arrow', 'scale', 'grid', 'colormap'])
        # elif self.stats_mode in [3, 4]:
        # 	self.draw_black()
        elif self.stats_mode in [5]:
            self.draw_recurrence()

        if self.stats_mode in [1, 2, 3, 4]:
            self.draw_stats()

        if self.recorder.is_recording:  # and self.is_run:
            self.recorder.record_frame(self.img)
        if self.is_save_image:
            self.recorder.save_image(self.img, filename=os.path.join(self.SAVE_ROOT, str(self.file_seq)))
            self.is_save_image = False
        if np.sum(self.world.cells) == 0:
            self.is_empty = True

        photo = PIL.ImageTk.PhotoImage(image=self.img)
        # photo = tk.PhotoImage(width=SIZEX, height=SIZEY)
        self.canvas.itemconfig(self.panel1, image=photo)
        self.win.update()

    def normalize(self, v, vmin, vmax):
        return (v - vmin) / (vmax - vmin)

    def draw_world(self, A, vmin=0, vmax=1, is_shift=False, is_shift_zero=False, markers=[]):
        if is_shift and not self.is_auto_center:
            shift = self.analyzer.total_shift_idx if 'world' in markers else self.analyzer.total_shift_idx - self.analyzer.last_shift_idx
            A = np.roll(A, shift.astype(int), (1, 0))
        # A = scipy.ndimage.shift(A, self.analyzer.total_shift_idx, order=0, mode='wrap')
        if is_shift_zero and self.automaton.is_soft_clip:
            if vmin == 0: vmin = np.amin(A)
        buffer = np.uint8(np.clip(self.normalize(A, vmin, vmax), 0, 1) * 252)  # .copy(order='C')
        # self.draw_grid(buffer, markers, is_fixed='fixgrid' in markers)

        buffer = np.repeat(np.repeat(buffer, PIXEL, axis=0), PIXEL, axis=1)
        zero = np.uint8(np.clip(self.normalize(0, vmin, vmax), 0, 1) * 252)
        for i in range(PIXEL_BORDER):
            buffer[i::PIXEL, :] = zero;
            buffer[:, i::PIXEL] = zero
        self.img = PIL.Image.frombuffer('P', buffer.shape, buffer, 'raw', 'P', 0, 1)

        # self.draw_arrow(markers)
        if is_shift and self.is_auto_center and self.is_auto_rotate:
            self.img = self.img.rotate(-self.ang_speed * self.automaton.time, resample=PIL.Image.NEAREST, expand=False)
        # ang_speed = OG:96, D7:-8, D8:-6, D9:-5.4
        # self.draw_legend(markers)
        # if self.stats_mode in [1]:
        #    self.draw_histo(A, vmin, vmax)
        self.img.putpalette(self.colormaps[self.colormap_id])

    def calc_fps(self):
        freq = 20 if self.show_freq == 1 else 200
        if self.automaton.gen == 0:
            self.last_time = time.time()
        elif self.automaton.gen % freq == 0:
            this_time = time.time()
            self.fps = freq / (this_time - self.last_time)
            self.last_time = this_time

    def change_b(self, i, d):
        B = len(self.world.params['b'])
        if B > 1 and i < B:
            self.world.params['b'][i] = min(1, max(0, self.world.params['b'][i] + Fraction(d, 12)))
            self.automaton.calc_kernel()
            self.check_auto_load()

    def adjust_b(self, d):
        B = len(self.world.params['b'])
        if B <= 0:
            self.world.params['b'] = [1]
        elif B >= 5:
            self.world.params['b'] = self.world.params['b'][0:5]
        else:
            self.world.params['R'] = self.world.params['R'] * B // (B - d)
        # temp_R = self.tx['R']
        # self.tx['R'] = self.tx['R'] * (B-d) // B
        # self.transform_world()
        # self.world.params['R'] = temp_R
        # self.automaton.calc_kernel()

    NUMBERS = {'asciitilde': 'quoteleft', 'ampersand': '1', 'eacute': '2', 'quotedbl': '3', 'apostrophe': '4', 'parenleft': '5', 'minus': '6', 'egrave': '1',
                  'underscore': '8', 'ccedilla': '9', 'agrave': '0'}
    S_NUMBERS = {}
    for key, value in NUMBERS.items():
        S_NUMBERS['s+' + key] = 's+' + value
    NUMBERS.update(S_NUMBERS)
    def key_press_event(self, event):
        ''' TKInter keys: https://www.tcl.tk/man/tcl8.6/TkCmd/keysyms.htm '''
        # Win: shift_l/r(0x1) caps_lock(0x2) control_l/r(0x4) alt_l/r(0x20000) win/app/alt_r/control_r(0x40000)
        # Mac: shift_l(0x1) caps_lock(0x2) control_l(0x4) meta_l(0x8,command) alt_l(0x10) super_l(0x40,fn)
        # print('keysym[{0.keysym}] char[{0.char}] keycode[{0.keycode}] state[{1}]'.format(event, hex(event.state))); return
        print(event)
        key = event.keysym
        state = event.state
        s = 's+' if state & 0x1 or (key.isalpha() and len(key) == 1 and key.isupper()) else ''
        c = 'c+' if state & 0x4 or (not is_windows and state & 0x8) else ''
        a = 'a+' if state & 0x20000 else ''
        key = key.lower()
        if key in self.NUMBERS:
            key = self.NUMBERS[key]
            # s = 's+'
        self.last_key = s + c + a + key
        print(self.last_key)
        self.is_internal_key = False

    def key_press_internal(self, key):
        self.last_key = key
        self.is_internal_key = True

    ANIMAL_KEY_LIST = {'1': 'O2(a)', '2': 'OG2', '3': 'OV2', '4': 'P4(a)', '5': '2S1:5', '6': '2S2:2', '7': '2P6,2,1', '8': '2PG1:2', '9': '3H3', '0': '~gldr', \
                       's+1': '3G2:4', 's+2': '3GG2', 's+3': 'K5(4,1)', 's+4': 'K7(4,3)', 's+5': 'K9(5,4)', 's+6': '3R5', 's+7': '4R6', 's+8': '2D10', 's+9': '4F12',
                       's+0': '~ggun', \
                       'c+1': '4Q(5,5,5,5):3', 'c+2': '3G7', 'c+3': '3EC', 'c+4': 'K4(2,2):3', 'c+5': '2L5', 'c+6': '3B7,1:6', 'c+7': '3R6:4', 'c+8': '4F7', 'c+9': '',
                       'c+0': 'bbug'}

    def process_key(self, k):
        global STATUS

        inc_or_dec = 1 if 's+' not in k else -1
        inc_10_or_1 = (10 if 's+' not in k else 1) if 'c+' not in k else 0
        inc_big_or_not = 0 if 'c+' not in k else 1
        inc_1_or_10 = 1 if 's+' not in k else 10
        inc_mul_or_not = 1 if 's+' not in k else 0
        double_or_not = 2 if 's+' not in k else 1
        inc_or_not = 0 if 's+' not in k else 1
        # print(k)
        is_ignore = False
        if not self.is_internal_key:
            self.stop_search()

        if k in ['escape']:
            self.is_closing = True; self.close()
        elif k in ['enter', 'return']:
            self.is_run = not self.is_run; self.run_counter = -1; self.info_type = 'info'
        elif k in [' ', 'space']:
            self.is_run = True; self.run_counter = 1; self.info_type = 'info'
        # elif k in [' ', 's+space']:
        #     self.is_run = True; self.run_counter = self.show_freq; self.info_type = 'info'
        # elif k in ['bracketright', 's+bracketright']:
        #     self.show_freq = self.show_freq + inc_10_or_1; self.info_type = 'info'
        # elif k in ['bracketleft', 's+bracketleft']:
        #     self.show_freq = self.show_freq - inc_10_or_1; self.info_type = 'info'
        # elif k in ['backslash']:
        #     if not self.is_auto_rotate:
        #         self.ang_gen = self.show_freq if self.show_freq > 1 else self.ang_gen
        #         self.is_ang_clockwise = False
        #         self.show_freq = 1
        #         self.is_auto_rotate = True
        #         self.is_auto_center = True
        #         self.info_type = 'angular'
        #     else:
        #         self.show_freq = 1
        #         self.is_auto_rotate = False
        #         self.is_auto_center = False
        # elif k in ['c+bracketright']:
        #     self.ang_sides += 1; self.info_type = 'angular'
        # elif k in ['c+bracketleft']:
        #     self.ang_sides -= 1; self.info_type = 'angular'
        # elif k in ['c+backslash']:
        #     self.is_ang_clockwise = not self.is_ang_clockwise; self.info_type = 'angular'
        elif k in ['asciicircum']:
            self.colormap_id = (self.colormap_id + inc_or_dec) % len(self.colormaps)
        elif k in ['tab', 's+tab']:
            self.show_what = (self.show_what + inc_or_dec) % self.SHOW_WHAT_NUM
        elif k in ['q', 's+q']:
            self.world.params['m'] += inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['a', 's+a']:
            self.world.params['m'] -= inc_10_or_1 * 0.001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['w', 's+w']:
            self.world.params['s'] += inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['s', 's+s']:
            self.world.params['s'] -= inc_10_or_1 * 0.0001; self.analyzer.new_segment(); self.check_auto_load(); self.info_type = 'params'
        elif k in ['t', 's+t']:
            self.world.params['T'] = max(1, self.world.params['T'] * double_or_not + inc_or_not); self.info_type = 'params'
        elif k in ['g', 's+g']:
            self.world.params['T'] = max(1, self.world.params['T'] // double_or_not - inc_or_not); self.info_type = 'params'
        elif k in ['r', 's+r']:
            self.tx['R'][0] = max(1, self.tx['R'][0] + inc_10_or_1); self.transform_world(); self.info_type = 'params'
        elif k in ['f', 's+f']:
            self.tx['R'] = max(1, self.tx['R'] - inc_10_or_1); self.transform_world(); self.info_type = 'params'
        elif k in ['e', 's+e']:
            self.world.param_P = max(0, self.world.param_P + inc_10_or_1); self.info_type = 'info'
        elif k in ['d', 's+d']:
            self.world.param_P = max(0, self.world.param_P - inc_10_or_1); self.info_type = 'info'
        elif k in ['y', 's+y']:
            self.change_b(0, inc_or_dec); self.info_type = 'params'
        elif k in ['u', 's+u']:
            self.change_b(1, inc_or_dec); self.info_type = 'params'
        elif k in ['i', 's+i']:
            self.change_b(2, inc_or_dec); self.info_type = 'params'
        elif k in ['o', 's+o']:
            self.change_b(3, inc_or_dec); self.info_type = 'params'
        elif k in ['p', 's+p']:
            self.change_b(4, inc_or_dec); self.info_type = 'params'
        elif k in ['semicolon']:
            self.world.params['b'].append(0); self.adjust_b(+1); self.info_type = 'params'
        elif k in ['s+semicolon']:
            self.world.params['b'].pop();     self.adjust_b(-1); self.info_type = 'params'
        elif k in ['c+d']:
            self.world.param_P = 0; self.info_type = 'info'
        elif k in ['c+q', 's+c+q']:
            self.toggle_search(+1, 's+' in k)
        elif k in ['c+a', 's+c+a']:
            self.toggle_search(-1, 's+' in k)
        elif k in ['c+w', 's+c+w']:
            self.toggle_trace('s+' in k)
        elif k in ['c+e', 's+c+e']:
            pass  # randam params and/or peaks
        elif k in ['c+r']:
            self.tx['R'] = DEF_R; self.transform_world(); self.info_type = 'params'
        elif k in ['c+f']:
            self.tx['R'] = self.fore.params['R'] if self.fore else DEF_R; self.transform_world(); self.info_type = 'params'
        elif k in ['c+y', 's+c+y']:
            self.automaton.kn = (self.automaton.kn + inc_or_dec - 1) % len(self.automaton.kernel_core) + 1; self.info_type = 'kn'
        elif k in ['c+u', 's+c+u']:
            self.automaton.gn = (self.automaton.gn + inc_or_dec - 1) % len(self.automaton.field_func) + 1; self.info_type = 'gn'
        elif k in ['c+i']:
            self.automaton.is_soft_clip = not self.automaton.is_soft_clip
        elif k in ['c+o']:
            self.automaton.is_multi_step = not self.automaton.is_multi_step
        elif k in ['c+p']:
            self.automaton.is_inverted = not self.automaton.is_inverted; self.world.params['T'] *= -1; self.world.params['m'] = 1 - self.world.params[
                'm']; self.world.cells = 1 - self.world.cells
        elif k in ['down', 's+down', 'c+down']:
            self.tx['shift'][0] += inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
        elif k in ['up', 's+up', 'c+up']:
            self.tx['shift'][0] -= inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
        elif k in ['right', 's+right', 'c+right']:
            self.tx['shift'][1] += inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
        elif k in ['left', 's+left', 'c+left']:
            self.tx['shift'][1] -= inc_10_or_1 + inc_big_or_not * 50; self.transform_world()
        elif k in ['pageup', 's+pageup', 'c+pageup', 'prior', 's+prior', 'c+prior']:
            self.tx['rotate'] += inc_10_or_1 + inc_big_or_not * 45; self.transform_world()
        elif k in ['pagedown', 's+pagedown', 'c+pagedown', 'next', 's+next', 'c+next']:
            self.tx['rotate'] -= inc_10_or_1 + inc_big_or_not * 45; self.transform_world()
        elif k in ['home']:
            self.tx['flip'] = 0 if self.tx['flip'] != 0 else -1; self.transform_world()
        elif k in ['end']:
            self.tx['flip'] = 1 if self.tx['flip'] != 1 else -1; self.transform_world()
        elif k in ['equal']:
            self.tx['flip'] = 2 if self.tx['flip'] != 0 else -1; self.transform_world()
        elif k in ['s+equal']:
            self.tx['flip'] = 3 if self.tx['flip'] != 0 else -1; self.transform_world()
        elif k in ['c+equal']:
            self.tx['flip'] = 4 if self.tx['flip'] != 0 else -1; self.transform_world()
        elif k in ['m']:
            self.is_auto_center = not self.is_auto_center
        elif k in ['backspace', 'delete']:
            self.clear_world()
        elif k in ['c', 's+c']:
            self.load_animal_id(self.animal_id - inc_1_or_10); self.info_type = 'animal'
        elif k in ['v', 's+v']:
            self.load_animal_id(self.animal_id + inc_1_or_10); self.info_type = 'animal'
        elif k in ['z']:
            self.load_animal_id(self.animal_id); self.info_type = 'animal'
        elif k in ['x', 's+x']:
            self.load_part(self.fore, is_random=True, is_replace=False, repeat=inc_1_or_10)
        elif k in ['b']:
            self.random_world()
        elif k in ['c+z']:
            self.is_auto_load = not self.is_auto_load
        elif k in ['s+c+x']:
            self.is_layered = not self.is_layered
        elif k in ['c+c', 's+c+c', 'c+n', 's+c+n', 'c+b']:
            A = copy.deepcopy(self.world)
            # self.target = copy.deepcopy(self.world)
            A.crop()
            data = A.to_data(is_shorten='s+' not in k)
            if k.endswith('c'):
                self.clipboard_st = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
                self.win.clipboard_clear()
                self.win.clipboard_append(self.clipboard_st)
                # print(self.clipboard_st)
                STATUS.append("> board saved to clipboard as RLE")
            elif k.endswith('n') or k.endswith('b'):
                if not os.path.exists(self.SAVE_ROOT):
                    os.makedirs(self.SAVE_ROOT)
                if k.endswith('b'):
                    self.file_seq += 1
                else:
                    self.file_seq = 0
                path = os.path.join(self.SAVE_ROOT, str(self.file_seq))
                with open(path + '.rle', 'w', encoding='utf8') as file:
                    file.write('#N ' + A.long_name() + '\n')
                    file.write('x = ' + str(A.cells.shape[1]) + ', y = ' + str(A.cells.shape[0]) + ', rule = Lenia(' + A.params2st() + ')\n')
                    file.write(data['cells'].replace('$', '$\n') + '\n')
                data['cells'] = [l if l.endswith('!') else l + '$' for l in data['cells'].split('$')]
                with open(path + '.json', 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=4, ensure_ascii=False)
                with open(path + '.csv', 'w', newline='\n') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.analyzer.stat_name(x=x) for x in self.analyzer.STAT_HEADERS])
                    writer.writerows([e for l in self.analyzer.series for e in l])
                STATUS.append("> data and image saved to '" + path + ".*'")
                self.is_save_image = True
        elif k in ['c+x']:
            A = copy.deepcopy(self.world)
            A.crop()
            stio = io.StringIO()
            writer = csv.writer(stio, delimiter=',', lineterminator='\n')
            writer.writerows(A.cells)
            self.clipboard_st = stio.getvalue()
            self.win.clipboard_clear()
            self.win.clipboard_append(self.clipboard_st)
            STATUS.append("> board saved to clipboard as CSV")
        elif k in ['c+v']:
            self.clipboard_st = self.win.clipboard_get()
            if 'cells' in self.clipboard_st:
                data = json.loads(self.clipboard_st.rstrip(', \t\r\n'))
                self.load_part(Board.from_data(data))
                self.info_type = 'params'
            else:
                delim = '\t' if '\t' in self.clipboard_st else ','
                stio = io.StringIO(self.clipboard_st)
                # stio.setvalue(self.clipboard_st)
                reader = csv.reader(stio, delimiter=delim, lineterminator='\n')
                cells = np.array([[float(st) if st != '' else 0 for st in l] for l in reader])
                print(cells)
                self.load_part(Board.from_values(cells))
        elif k in ['c+m', 's+c+m']:
            self.recorder.toggle_recording(is_save_frames='s+' in k)
        elif k in ['c+g']:
            if self.automaton.has_gpu:
                self.automaton.is_gpu = not self.automaton.is_gpu
        elif k in [m + str(i) for i in range(10) for m in ['', 's+', 'c+', 's+c+']]:
            self.load_animal_code(self.ANIMAL_KEY_LIST.get(k)); self.info_type = 'animal'
        elif k in ['h', 's+h']:
            self.markers_mode = (self.markers_mode + inc_or_dec) % 6
        elif k in ['c+h']:
            self.is_show_fps = not self.is_show_fps
        elif k in ['j', 's+j']:
            self.stats_mode = (self.stats_mode + inc_or_dec) % 6
        elif k in ['c+j']:
            self.analyzer.clear_segment()
        elif k in ['s+c+j']:
            self.analyzer.clear_series()
        elif k in ['k', 's+k']:
            if self.stats_mode == 0: self.stats_mode = 1
            while True:
                self.stats_x = (self.stats_x + inc_or_dec) % len(self.analyzer.STAT_HEADERS);
                self.info_type = 'stats'
                if self.stats_x != self.stats_y and self.stats_x > 2: break
        elif k in ['l', 's+l']:
            if self.stats_mode == 0: self.stats_mode = 1
            while True:
                self.stats_y = (self.stats_y + inc_or_dec) % len(self.analyzer.STAT_HEADERS);
                self.info_type = 'stats'
                if self.stats_x != self.stats_y and self.stats_y > 2: break
        elif k in ['c+k']:
            self.analyzer.is_trim_segment = not self.analyzer.is_trim_segment
        elif k in ['c+l']:
            self.is_group_params = not self.is_group_params
        elif k in ['comma']:
            self.info_type = 'animal'
        elif k in ['period']:
            self.info_type = 'params'
        elif k in ['slash']:
            self.info_type = 'info'
        elif k in ['s+slash']:
            self.info_type = 'angular'
        # elif k in ['c+slash']: m = self.menu.children[self.menu_values['animal'][0]].children['!menu']; m.post(self.win.winfo_rootx(), self.win.winfo_rooty())
        elif k.endswith('_l') or k.endswith('_r'):
            is_ignore = True
        else:
            self.excess_key = k

        self.show_freq = max(1, self.show_freq)
        self.ang_sides = max(1, self.ang_sides)
        self.ang_speed = (-1 if self.is_ang_clockwise else +1) * 360 / self.ang_sides / self.ang_gen * self.world.params['T']
        if not is_ignore and self.is_loop:
            self.roundup(self.world.params)
            self.roundup(self.tx)
            self.automaton.calc_once(is_update=False)
        # self.update_menu()

    # if not is_ignore:
    # v1 = self.target.cells.reshape(-1, 1)
    # v2 = self.world.cells.reshape(-1, 1)
    # print('fitness: {:.4f}'.format(-np.linalg.norm(v1 - v2)))

    def roundup(self, A):
        for (k, x) in A.items():
            if type(x) == float:
                A[k] = round(x, ROUND)

    def get_acc_func(self, key, acc, animal_id=None):
        acc = acc if acc else key if key else None
        if acc: acc = acc.replace('s+', 'Shift+').replace('c+', 'Ctrl+').replace('m+', 'Cmd+').replace('a+', 'Slt+')
        if animal_id:
            func = lambda: self.load_animal_id(int(animal_id))
        else:
            func = lambda: self.key_press_internal(key.lower()) if key else None
        state = 'normal' if key or animal_id else tk.DISABLED
        return {'accelerator': acc, 'command': func, 'state': state}

    def get_animal_nested_list(self):
        root = []
        stack = [root]
        id = 0
        for data in self.animal_data:
            code = data['code']
            if code.startswith('>'):
                next_level = int(code[1:])
                d = len(stack) - next_level
                for i in range(d):
                    stack.pop()
                for i in range(max(-d, 0) + 1):
                    new_list = ('{name} {cname}'.format(**data), [])
                    stack[-1].append(new_list)
                    stack.append(new_list[1])
            else:
                stack[-1].append('&{id}|{name} {cname}|'.format(id=id, **data))
            id += 1
        return root

    def get_nested_attr(self, name):
        obj = self
        for n in name.split('.'):
            obj = getattr(obj, n)
        return obj

    PARAM_TEXT = {'m': 'Field center', 's': 'Field width', 'R': 'Space units', 'T': 'Time units', 'dr': 'Space step', 'dt': 'Time step', 'b': 'Kernel peaks'}
    VALUE_TEXT = {'animal': 'Animal', 'kn': 'Kernel core', 'gn': 'Field func', 'show_what': 'Show', 'colormap_id': 'Colors'}

    def get_info_st(self):
        return 'gen={}, t={}s, dt={}s, world={}x{}, pixel={}, P={}, sampl={}'.format(self.automaton.gen, self.automaton.time, 1 / self.world.params['T'], SIZEX, SIZEY, PIXEL,
                                                                                     self.world.param_P, self.show_freq)

    def get_angular_st(self):
        if self.is_auto_rotate:
            return 'auto-rotate: {} axes={} sampl={} speed={:.2f}'.format('clockwise' if self.is_ang_clockwise else 'anti-clockwise', self.ang_sides, self.ang_gen, self.ang_speed)
        else:
            return 'not in auto-rotate mode'


    def clear_info(self):
        self.info_bar.config(text="")
        self.clear_job = None

    def loop(self):
        self.is_loop = True
        self.win.after(0, self.run)
        self.win.protocol('WM_DELETE_WINDOW', self.close)
        self.win.mainloop()

    def close(self):
        self.is_loop = False
        if self.recorder.is_recording:
            self.recorder.finish_record()
        self.win.destroy()

    def normalize_control(self, val, start_min, start_max, end_min, end_max):
        return (val - start_min) / (start_max - start_min) * (end_max - end_min) + end_min

    def run(self):
        # t_signal = - self.signals[0]
        # sig_t_min, sig_t_max = t_signal.mean() - 2 * t_signal.std(), t_signal.mean() + t_signal.std()
        t_min, t_max = 5, 40
        #
        # r_signal = self.signals[3]
        # sig_r_min, sig_r_max = r_signal.min(), r_signal.max()
        # r_min, r_max = self.controls['R'].min(), self.controls['R'].max()
        #
        # c_signal = self.signals[1]
        #
        # switch_signal = self.signals[4]

        # self.last_real_time = 0
        counter = 0
        # centers = np.arange(0, t_signal.size * WIN_LEN * OVERLAP, OVERLAP * WIN_LEN) * 1000
        # mt_centers = np.arange(0, r_signal.size * MID_WIN_LEN * MID_OVERLAP, MID_WIN_LEN * MID_OVERLAP) * 1000
        # last_switch = self.real_time
        # last_switch_cmap = self.real_time
        times = []

        if AUDIO_CONTROL:
            stream_data = self.mic_stream.get(get_peak_fft=True, get_power=True)
            memory = dict(zip(stream_data.keys(), [[] for _ in range(len(stream_data.keys()))]))
            memory['t'] = []
            keys = sorted(list(memory.keys()))
            back_memory = 5
        t_init_0 = time.time()
        abs_min = np.inf
        abs_max = 0
        counter_since_last_switch = 0
        while self.is_loop:
            t_init = time.time()
            # index_short = np.argmin(np.abs(self.real_time - centers))
            # index_mid = np.argmin(np.abs(self.real_time - mt_centers))

            # if index_mid < (r_signal.size - 1):
            #     # SET R
            #     value = r_signal[index_mid]
            #     R = self.normalize_control(value, sig_r_min, sig_r_max, r_min, r_max)
            #     # print('R: ', R)
            #     self.world.params['R'][0] = int(R)
            #     print(int(R))
            #     self.tx['R'][0] = int(R)  # int(max(self.initial_world_params['R'][self.kernel_id] + self.controls[key][value], 5))
            #     # self.tx['R'][0] /= 2
            #     self.transform_world()
            #     self.automaton.calc_once(is_update=False)
            #     self.automaton.calc_kernel()

            if AUDIO_CONTROL:
                stream_data = self.mic_stream.get(get_peak_fft=True, get_power=True)
                for k in stream_data.keys():
                    memory[k].append(stream_data[k])
                memory['t'].append(t_init - t_init_0)
                inds = np.argwhere( np.array(memory['t']) < ((t_init - t_init_0) - back_memory)).flatten()
                if inds.size > 0:
                    for k in keys:
                        memory[k] = memory[k][inds[-1] + 1:]

                if 'power' in keys:
                    if len(memory['power']) > 2:

                        av_power = np.mean(memory['power'])
                        max_power = np.max(memory['power'])
                        min_power = np.min(memory['power'])
                    else:
                        min_power = 200
                        max_power = 1000
                    if min_power < abs_min:
                        abs_min = min_power
                    if max_power > abs_max:
                        abs_max = max_power
                    # print(abs_min, abs_max)
                    def normalize_power(power, t_min, t_max):
                        return - ((power - min_power) / (max_power - min_power) * (t_max - t_min) - t_max)

            if True: # index_short < (t_signal.size - 1):
                # SET T
                if AUDIO_CONTROL:
                    if 'power' in keys:
                        print(stream_data['power'])
                        if stream_data['power'] > 2500 and counter_since_last_switch > 3:
                            self.colormap_id = (self.colormap_id + 1) % len(self.colormaps)
                            counter_since_last_switch = 0
                            # T = normalize_power(stream_data['power'], t_min=t_min, t_max=10)
                            # print('DETECTED')

                        counter_since_last_switch += 1
                            # T = normalize_power(stream_data['power'], t_min=10, t_max=t_max)
                        # print('T: ', T)
                        # self.world.params['T'] = int(T)

                # countrer += 1
                controls = dict()
                notes = []
                for msg in MIDI_PORT.iter_pending():
                    # print(msg)
                    try:
                        controls[msg.control] = msg.value
                    except:
                        notes.append(msg.note)

                if len(controls.keys()) > 0:
                    self.update_midi_controls(controls)

                    # if list(controls.keys())[0] in NOTES:
                    #     self.update_midi_notes(list(controls.keys())[0])

                if self.is_empty:
                    self.load_animal_id(np.random.randint(len(self.animal_data)))
                    self.is_empty = False
                if self.last_key:
                    self.process_key(self.last_key)
                    self.last_key = None
                if self.is_closing:
                    break
                if self.is_run:
                    self.calc_fps()
                    self.automaton.calc_once()
                    self.analyzer.center_world()
                    # self.analyzer.calc_stats()
                    # self.analyzer.add_stats()
                    if not self.is_layered:
                        self.back = None
                        self.clear_transform()
                    if self.search_dir != None:
                        self.search_params()
                    elif self.trace_m != None:
                        self.trace_params()
                    if self.run_counter != -1:
                        self.run_counter -= 1
                        if self.run_counter == 0:
                            self.is_run = False
                if counter % self.show_freq == 0:
                    #self.update_info_bar()
                    self.update_win()

                # SAVE IMG
                # self.img.save("/home/flowers/Desktop/Perso/Scratch/funky_lenia/data/save/test7/{}.png".format(self.real_time))
                self.img_count += 1
                # print('Im #{}'.format(self.img_count))

            else:
                self.is_loop = False
                print('THE END')
                os.system("ffmpeg -r 20 -i im_%d.png -vcodec mpeg4 -y movie.mp4")
            times.append(time.time() - t_init)

            stop = 1
            # print(times[-1])


if __name__ == '__main__':
    lenia = Lenia()
    lenia.load_animal_code(lenia.ANIMAL_KEY_LIST['2'])
    # lenia.update_menu()
    lenia.loop()

''' for PyOpenCL in Windows:
install Intel OpenCL SDK
install Microsoft Visual C++ Build Tools
in Visual Studio Native Tools command prompt
> set INCLUDE=%INCLUDE%;%INTELOCLSDKROOT%include
> set LIB=%LIB%;%INTELOCLSDKROOT%lib\x86
> pip3 install pyopencl
'''
