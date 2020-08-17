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
        self.value_s = SIGMA
        self.value_m = MU

        ''' http://hslpicker.com/ '''
        self.colormaps = [
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
        # self.target = Board((SIZEY, SIZEX))
        self.automaton = Automaton(self.world)
        self.analyzer = Analyzer(self.automaton)
        self.recorder = Recorder(self.world)
        self.clear_transform()
        self.create_window()

    # self.create_menu()

    def clear_transform(self):
        self.tx = {'shift': [0, 0], 'rotate': 0, 'R': self.world.params['R'], 'flip': -1}

    def read_animals(self):
        with open('animals.json', encoding='utf-8') as file:
            self.animal_data = json.load(file)

    def load_animal_id(self, id, **kwargs):
        self.animal_id = max(0, min(len(self.animal_data) - 1, id))
        self.load_part(Board.from_data(self.animal_data[self.animal_id]), **kwargs)

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
                self.world.add_transformed(part, self.tx)

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
        border = self.world.params['R']
        rand = np.random.rand(SIZEY - border * 2, SIZEX - border * 2)
        self.world.clear()
        self.world.add(Board.from_values(rand))
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

    # def draw_histo(self, A, vmin, vmax):
    # draw = PIL.ImageDraw.Draw(self.img)
    # HWIDTH = 1
    # hist, _ = np.histogram(A, bins=SIZEX//HWIDTH, range=(vmin,vmax))  #, density=True)
    # #print(vmin, vmax, A.min(), A.max())
    # for i in range(hist.shape[0]):
    # h = hist[i]
    # y = h  #(h*SIZEY).astype(int)
    # draw.rectangle([(i*HWIDTH*PIXEL,SIZEY*PIXEL),((i+1)*HWIDTH*PIXEL-1,(SIZEY-y)*PIXEL)], fill=254)
    # del draw

    # def draw_black(self):
    # 	size = (SIZEX*PIXEL,SIZEY*PIXEL)
    # 	self.img = PIL.Image.frombuffer('L', size, np.zeros(size), 'raw', 'L', 0, 1)

    # def draw_grid(self, buffer, markers=[], is_fixed=False):
    # 	R = self.world.params['R']
    # 	n = R // 40 if R >= 15 else -1
    # 	if ('grid' in markers or 'fixgrid' in markers) and self.markers_mode in [0,1,2]:
    # 		for i in range(-n, n+1):
    # 			sx, sy = 0, 0
    # 			if self.is_auto_center and not is_fixed:
    # 				sx, sy = self.analyzer.total_shift_idx.astype(int)
    # 			grid = buffer[(MIDY - sy + i) % R:SIZEY:R, (MIDX - sx) % R:SIZEX:R];  grid[grid==0] = 253
    # 			grid = buffer[(MIDY - sy) % R:SIZEY:R, (MIDX - sx + i) % R:SIZEX:R];  grid[grid==0] = 253

    # def draw_arrow(self, markers=[]):
    # 	draw = PIL.ImageDraw.Draw(self.img)
    # 	R, T = [self.world.params[k] for k in ('R', 'T')]
    # 	midpoint = np.array([MIDX, MIDY])
    # 	d2 = np.array([1, 1]) * 2
    # 	if 'arrow' in markers and self.markers_mode in [0,1,2] and self.world.params['R'] > 2 and self.analyzer.m_last_center is not None and self.analyzer.m_center is not None:
    # 		shift = self.analyzer.total_shift_idx if not self.is_auto_center else np.zeros(2)
    # 		m0 = self.analyzer.m_last_center * R + midpoint + shift - self.analyzer.last_shift_idx
    # 		m1 = self.analyzer.m_center * R + midpoint + shift
    # 		ms = m1 % np.array([SIZEX, SIZEY]) - m1
    # 		m2, m3 = [m0 + (m1 - m0) * n * T for n in [1,2]]
    # 		for i in range(-1, 2):
    # 			for j in range(-1, 2):
    # 				adj = np.array([i*SIZEX, j*SIZEY]) + ms
    # 				draw.line(tuple((m0+adj)*PIXEL) + tuple((m3+adj)*PIXEL), fill=254, width=1)
    # 				[draw.ellipse(tuple((m+adj)*PIXEL-d2) + tuple((m+adj)*PIXEL+d2), fill=c) for (m,c) in [(m0,254),(m1,255),(m2,255),(m3,255)]]
    # 	del draw

    # def draw_legend(self, markers=[]):
    # 	draw = PIL.ImageDraw.Draw(self.img)
    # 	R, T = [self.world.params[k] for k in ('R', 'T')]
    # 	midpoint = np.array([MIDX, MIDY])
    # 	d2 = np.array([1, 1]) * 2
    # 	if 'arrow' in markers and self.markers_mode in [0,1,3,4]:
    # 		x0, y0 = SIZEX*PIXEL-50, SIZEY*PIXEL-35
    # 		draw.line([(x0-90,y0),(x0,y0)], fill=254, width=1)
    # 		[draw.ellipse(tuple((x0+m,y0)-d2) + tuple((x0+m,y0)+d2), fill=c) for (m,c) in [(0,254),(-10,255),(-50,255),(-90,255)]]
    # 		draw.text((x0-95,y0-20), '2s', fill=255)
    # 		draw.text((x0-55,y0-20), '1s', fill=255)
    # 		if self.is_auto_center and self.is_auto_rotate:
    # 			for i in range(self.ang_sides):
    # 				angle = 2*np.pi * i / self.ang_sides
    # 				d = np.array([np.sin(angle), np.cos(angle)])*SIZEX
    # 				draw.line(tuple(midpoint*PIXEL) + tuple((midpoint-d)*PIXEL), fill=254, width=1)
    # 	if 'scale' in markers and self.markers_mode in [0,1,3,4]:
    # 		x0, y0 = SIZEX*PIXEL-50, SIZEY*PIXEL-20
    # 		draw.text((x0+10,y0), '1mm', fill=255)
    # 		draw.rectangle([(x0-R*PIXEL,y0+3),(x0,y0+7)], fill=255)
    # 	if 'colormap' in markers and self.markers_mode in [0,3]:
    # 		x0, y0 = SIZEX*PIXEL-20, SIZEY*PIXEL-70
    # 		x1, y1 = SIZEX*PIXEL-15, 20
    # 		dy = (y1-y0)/253
    # 		draw.rectangle([(x0-1,y0+1),(x1+1,y1-1)], outline=254)
    # 		for c in range(253):
    # 			draw.rectangle([(x0,y0+dy*c),(x1,y0+dy*(c+1))], fill=c)
    # 		draw.text((x0-25,y0-5), '0.0', fill=255)
    # 		draw.text((x0-25,(y1+y0)//2-5), '0.5', fill=255)
    # 		draw.text((x0-25,y1-5), '1.0', fill=255)
    # 	del draw
    #
    # def draw_stats(self):
    # 	draw = PIL.ImageDraw.Draw(self.img)
    # 	series = self.analyzer.series
    # 	name_x = self.analyzer.STAT_HEADERS[self.stats_x]
    # 	name_y = self.analyzer.STAT_HEADERS[self.stats_y]
    # 	if series != [] and self.stats_mode in [1, 2, 3]:
    # 		series = [series[-1]]
    # 	if series != [] and series != [[]]:
    # 		X = [np.array([val[self.stats_x] for val in seg]) for seg in series if len(seg)>0]
    # 		Y = [np.array([val[self.stats_y] for val in seg]) for seg in series if len(seg)>0]
    # 		S = [seg[0][1] for seg in series if len(seg)>0]
    # 		M = [seg[0][0] for seg in series if len(seg)>0]
    # 		if name_x in ['n', 't']: X = [seg - seg.min() for seg in X]
    # 		if name_y in ['n', 't']: Y = [seg - seg.min() for seg in Y]
    # 		xmin, xmax = min(seg.min() for seg in X if seg.size>0), max(seg.max() for seg in X if seg.size>0)
    # 		ymin, ymax = min(seg.min() for seg in Y if seg.size>0), max(seg.max() for seg in Y if seg.size>0)
    # 		smin, smax = min(S), max(S)
    # 		mmin, mmax = min(M), max(M)
    # 		# xmean, ymean = (xmax+xmin) / 2, (ymax+ymin) / 2
    # 		# if xmax-xmin < 0.01: xmin, xmax = xmean-0.01, xmean+0.01
    # 		# if ymax-ymin < 0.01: ymin, ymax = ymean-0.01, ymean+0.01
    # 		# if name_x in ['m_a']:
    # 			# mass = [np.array([val[4] for val in seg]) for seg in series if len(seg)>0]
    # 			# massmax = max(seg.max() for seg in mass if seg.size>0)
    # 			# if name_x in ['m_a']:
    # 				# xmin, xmax = min(xmin, -massmax/32), max(xmax, massmax/32)
    # 		if self.stats_mode in [1]:
    # 			xmax = (xmax - xmin) * 4 + xmin
    # 			ymax = (ymax - ymin) * 4 + ymin
    # 			title_x, title_y = 1, SIZEY * 3 // 4 - 4
    # 		else:
    # 			title_x, title_y = 1, 1
    # 		# font = PIL.ImageFont.truetype('InputMono-Regular.ttf')
    # 		draw.text((title_x*PIXEL,title_y*PIXEL), (name_x+'-'+name_y), fill=255)
    # 		if self.stats_mode in [4]:
    # 			C = reversed([194 // 2**i + 61 for i in range(len(X))])
    # 		else:
    # 			C = [255] * len(X)
    # 		ds = 0.0001 if self.is_search_small else 0.001
    # 		dm = 0.001 if self.is_search_small else 0.01
    # 		for x, y, m, s, c in zip(X, Y, M, S, C):
    # 			is_valid = not np.isnan(x[0])
    # 			if self.is_group_params:
    # 				xmin, xmax = x.min(), x.max()
    # 				ymin, ymax = y.min(), y.max()
    # 				x, y = self.normalize(x, xmin, xmax), self.normalize(y, ymin, ymax)
    # 				s, m = self.normalize(s, smin, smax+ds), self.normalize(m, mmin, mmax+dm)
    # 				x, x0, x1 = [(a * ds/(smax-smin+ds) + s) * (SIZEX*PIXEL - 10) + 5 for a in [x, 0, 1]]
    # 				y, y0, y1 = [(1 - a * dm/(mmax-mmin+dm) - m) * (SIZEY*PIXEL - 10) + 5 for a in [y, 0, 1]]
    # 				draw.rectangle([(x0,y0),(x1,y1)], outline=c, fill=None if is_valid else c)
    # 			else:
    # 				x = self.normalize(x, xmin, xmax) * (SIZEX*PIXEL - 10) + 5
    # 				y = (1-self.normalize(y, ymin, ymax)) * (SIZEY*PIXEL - 10) + 5
    # 			if is_valid:
    # 				draw.line(list(zip(x, y)), fill=c, width=1)
    # 	del draw
    #
    # def draw_recurrence(self, e=0.1, steps=10):
    # 	''' https://stackoverflow.com/questions/33650371/recurrence-plot-in-python '''
    # 	if self.analyzer.series == [] or len(self.analyzer.series[-1]) < 2:
    # 		return
    #
    # 	size = min(SIZEX*PIXEL, SIZEY*PIXEL)
    # 	segment = np.array(self.analyzer.series[-1])[-size:, 4:]
    # 	vmin, vmax = segment.min(axis=0), segment.max(axis=0)
    # 	# vmean = (vmax + vmin) / 2
    # 	# d = vmax - vmin < 0.01
    # 	# vmin[d], vmax[d] = vmean[d] - 0.01, vmean[d] + 0.01
    # 	segment = self.normalize(segment, vmin, vmax)
    # 	D = scipy.spatial.distance.squareform(np.log(scipy.spatial.distance.pdist(segment))) + np.eye(segment.shape[0]) * -100
    # 	buffer = np.uint8(np.clip(-D/2, 0, 1) * 253)
    # 	self.img = PIL.Image.frombuffer('L', buffer.shape, buffer, 'raw', 'L', 0, 1)

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

    def change_b(self, i, d):
        B = len(self.world.params['b'])
        if B > 1 and i < B:
            self.world.params['b'][i] = min(1, max(0, self.world.params['b'][i] + Fraction(d, 12)))
            self.automaton.calc_kernel()
            self.check_auto_load()

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
            self.tx['R'] = max(1, self.tx['R'] + inc_10_or_1); self.transform_world(); self.info_type = 'params'
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


    def update_midi_controls(self, controls):
        CONTROLS = [13, 14, 15, 16, 17, 18, 19, 20, 29, 30, 31, 32, 33, 34, 35, 36, 49, 50, 51, 52, 53, 54, 55, 56, 77, 78, 79, 80, 81, 82, 83, 84]
        RANGE_B = np.arange(0, 1.001, 1/127)
        # print(controls)
        for control, value in controls.items():
            if control == 16:
                self.world.params['m'] += DIFF_MU[value]
                self.value_m = value
                print('Setting mean to: {}'.format(RANGE_MU[value]))
            elif control == 17:
                self.world.params['s'] += DIFF_SIGMA[value]#RANGE_SIGMA[value]
                self.value_s = value
                print('Setting std to: {}'.format(RANGE_SIGMA[value]))
            elif control == 18:
                self.world.params['T'] = RANGE_T[value]
                print('Setting T to: {}'.format(RANGE_T[value]))
            elif control == 19:
                self.world.params['b'][0] = RANGE_B[value]
                print('Setting beta 1 to: {}'.format(RANGE_B[value]))
                print('Beta', self.world.params['b'])
            elif control == 20:
                while len(self.world.params['b']) < 2:
                    self.world.params['b'].append(0)
                self.world.params['b'][1] = RANGE_B[value]
                print('Setting beta 2 to: {}'.format(RANGE_B[value]))
                print('Beta', self.world.params['b'])
            elif control == 21:
                while len(self.world.params['b']) < 3:
                    self.world.params['b'].append(0)
                self.world.params['b'][2] = RANGE_B[value]
                print('Setting beta 3 to: {}'.format(RANGE_B[value]))
                print('Beta', self.world.params['b'])
        # if 20 in controls.keys() or 36 in controls.keys() or 56 in controls.keys():
        # self.automaton.calc_once(is_update=False)
        # self.automaton.calc_kernel()

        self.analyzer.new_segment()
        self.check_auto_load()

        if self.is_loop:
            self.roundup(self.world.params)
            self.roundup(self.tx)
            self.automaton.calc_once(is_update=False)

    def update_midi_notes(self, control):

        print(control)
        if control == 32:
            # change kernel
            self.automaton.kn = (self.automaton.kn + 1) % len(self.automaton.kernel_core) + 1
            self.info_type = 'kn'
            self.automaton.calc_once(is_update=False)
            self.automaton.calc_kernel()

        elif control == 33:
            # change growth function
            self.automaton.gn = (self.automaton.gn + 1) % len(self.automaton.field_func) + 1
            self.info_type = 'gn'
            self.automaton.calc_once(is_update=False)
            self.automaton.calc_kernel()

        elif control == 64:
            # change animal to random
            self.load_animal_id(np.random.randint(len(self.animal_data)))
            DIFF_MU = np.arange(- DELTA_MU * self.value_m, DELTA_MU * (127 - self.value_m) + DELTA_MU / 2, DELTA_MU)
            DIFF_SIGMA = np.arange(- DELTA_SIGMA * self.value_s, DELTA_SIGMA * (127 - self.value_s) + DELTA_SIGMA / 2, DELTA_SIGMA)

        self.analyzer.new_segment()
        self.check_auto_load()

        if self.is_loop:
            self.roundup(self.world.params)
            self.roundup(self.tx)
            self.automaton.calc_once(is_update=False)


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

    def get_value_text(self, name):
        if name == 'animal':
            return '#' + str(self.animal_id + 1) + ' ' + self.world.long_name()
        elif name == 'kn':
            return ["Exponential", "Polynomial", "Step", "Staircase"][(self.world.params.get('kn') or self.automaton.kn) - 1]
        elif name == 'gn':
            return ["Exponential", "Polynomial", "Step"][(self.world.params.get('gn') or self.automaton.gn) - 1]
        elif name == 'colormap_id':
            return ["Vivid blue/red", "Vivid green/purple", "Vivid red/green", "Pale blue/red", "Pale green/purple", "Pale yellow/green", "White/black", "Black/white"][
                self.colormap_id]
        elif name == 'show_what':
            return ["World", "Potential", "Field", "Kernel", "World FFT", "Potential FFT", "Gradient", "Change"][self.show_what]
        elif name == 'markers_mode':
            return ["Marks,legend,colors", "Marks,legend", "Marks", "Legend,colors", "Legend", "None"][self.markers_mode]
        elif name == 'stats_mode':
            return ["None", "Corner", "Overlay", "Segment", "All segments", "Recurrence plot"][self.stats_mode]
        elif name == 'stats_x':
            return self.analyzer.stat_name(i=self.stats_x)
        elif name == 'stats_y':
            return self.analyzer.stat_name(i=self.stats_y)

    # def update_menu(self):
    # 	for name in self.menu_vars:
    # 		self.menu_vars[name].set(self.get_nested_attr(name))
    # 	for (name, info) in self.menu_params.items():
    # 		value = '['+Board.fracs2st(self.world.params[name])+']' if name=='b' else self.world.params[name]
    # 		self.menu.children[info[0]].entryconfig(info[1], label='{text} ({param} = {value})'.format(text=info[2], param=name, value=value))
    # 	for (name, info) in self.menu_values.items():
    # 		value = self.get_value_text(name)
    # 		self.menu.children[info[0]].entryconfig(info[1], label='{text} [{value}]'.format(text=info[2], value=value))

    PARAM_TEXT = {'m': 'Field center', 's': 'Field width', 'R': 'Space units', 'T': 'Time units', 'dr': 'Space step', 'dt': 'Time step', 'b': 'Kernel peaks'}
    VALUE_TEXT = {'animal': 'Animal', 'kn': 'Kernel core', 'gn': 'Field func', 'show_what': 'Show', 'colormap_id': 'Colors'}

    # def create_menu(self):
    # 	self.menu_vars = {}
    # 	self.menu_params = {}
    # 	self.menu_values = {}
    # 	self.menu = tk.Menu(self.win, tearoff=True)
    # 	self.win.config(menu=self.menu)
    #
    # 	items2 = ['^automaton.is_gpu|Use GPU|c+G' if self.automaton.has_gpu else '|No GPU available|']
    # 	self.menu.add_cascade(label='Main', menu=self.create_submenu(self.menu, [
    # 		'^is_run|Start|Return', '|Once|Space'] + items2 + [None,
    # 		'@show_what|Display|Tab', '@colormap_id|Colors|QuoteLeft|`', None,
    # 		'|Show animal name|Comma|,', '|Show params|Period|.', '|Show info|Slash|/', '|Show auto-rotate info|s+Slash|s+/', None,
    # 		'|Save data & image|c+N', '|Save with expanded format|s+c+N',  '|Save next in sequence|c+B',
    # 		'^recorder.is_recording|Record video & gif|c+M', '|Record with frames saved|s+c+M', None,
    # 		'|Quit|Escape']))
    #
    # 	self.menu.add_cascade(label='Move', menu=self.create_submenu(self.menu, [
    # 		'^is_auto_center|Auto-center mode|M', None,
    # 		'|(Small adjust)||s+Up', '|(Large adjust)||m+Up',
    # 		'|Move up|Up', '|Move down|Down', '|Move left|Left', '|Move right|Right']))
    #
    # 	self.menu.add_cascade(label='Rotate', menu=self.create_submenu(self.menu, [
    # 		'|Rotate clockwise|PageUp', '|Rotate anti-clockwise|PageDown', None,
    # 		'|(Small adjust)||s+]', '|Sampling period + 10|BracketRight|]', '|Sampling period - 10|BracketLeft|[', '|Run one sampling period|s+Space', None,
    # 		'^is_auto_rotate|Auto-rotate mode|BackSlash|\\', '|Symmetry axes + 1|c+BracketRight|c+]', '|Symmetry axes - 1|c+BracketLeft|c+[', '^is_ang_clockwise|Clockwise|c+BackSlash|c+\\']))
    #
    # 	items2 = []
    # 	# for (key, code) in self.ANIMAL_KEY_LIST.items():
    # 		# id = self.get_animal_id(code)
    # 		# if id: items2.append('|{name} {cname}|{key}'.format(**self.animal_data[id], key=key))
    # 	self.menu.add_cascade(label='Animal', menu=self.create_submenu(self.menu, [
    # 		'|Place at center|Z', '|Place at random|X',
    # 		'|Previous animal|C', '|Next animal|V', '|Previous 10|s+C', '|Next 10|s+V', None,
    # 		'|Shortcuts 1-10|1', '|Shortcuts 11-20|s+1', '|Shortcuts 21-30|c+1', None,
    # 		('Full list', self.get_animal_nested_list())]))
    #
    # 	self.menu.add_cascade(label='Cells', menu=self.create_submenu(self.menu, [
    # 		'|Clear|Backspace', '|Random|B', '|Random (last seed)|N', None,
    # 		'|Flip vertically|Home', '|Flip horizontally|End',
    # 		'|Mirror horizontally|Equal|=', '|Mirror flip|s+Equal|+', '|Mirror diagonally|c+Equal|c+=', None,
    # 		'|Copy|c+C', '|Copy as table|c+X', '|Paste|c+V', None,
    # 		'^is_auto_load|Auto put (place/paste/random)|c+Z', '^is_layered|Layer mode|s+c+X']))
    #
    # 	items2 = ['|More peaks|SemiColon', '|Fewer peaks|s+SemiColon', None]
    # 	for i in range(5):
    # 		items2.append('|Higher peak {n}|{key}'.format(n=i+1, key='YUIOP'[i]))
    # 		items2.append('|Lower peak {n}|{key}'.format(n=i+1, key='s+'+'YUIOP'[i]))
    # 	items2 += [None, '|Random peaks & field|s+c+E']
    #
    # 	# '@animal||', '#m|Field center', '#s|Field width', '#R|Space units', '#T|Time units', '#b|Kernel peaks',
    # 	self.menu.add_cascade(label='Params', menu=self.create_submenu(self.menu, [
    # 		'|(Small adjust)||s+Q', '|Higher field (m + 0.01)|Q', '|Lower field (m - 0.01)|A',
    # 		'|Wider field (s + 0.001)|W', '|Narrower field (s - 0.001)|S', None,
    # 		'|More states (P + 10)|E', '|Fewer states (P - 10)|D',
    # 		'|Bigger size (R + 10)|R', '|Smaller size (R - 10)|F',
    # 		'|Slower speed (T * 2)|T', '|Faster speed (T / 2)|G', None,
    # 		('Peaks', items2), None,
    # 		'|Reset states|c+D', '|Reset size|c+R', '|Animal\'s original size|c+F']))
    #
    # 	self.menu.add_cascade(label='Options', menu=self.create_submenu(self.menu, [
    # 		'|Search field higher|c+Q', '|Search field lower|c+A', '|Random field|c+E', None,
    # 		'@kn|Kernel core|c+Y', '@gn|Field func|c+U',
    # 		'^automaton.is_soft_clip|Use soft clip|c+I', '^automaton.is_multi_step|Use multi-step|c+O',
    # 		'^automaton.is_inverted|Invert|c+P']))
    #
    # 	self.menu.add_cascade(label='Stats', menu=self.create_submenu(self.menu, [
    # 		'@markers_mode|Show marks|H', '^is_show_fps|Show FPS|c+H', None,
    # 		'@stats_mode|Show stats|J', '@stats_x|Stats X axis|K', '@stats_y|Stats Y axis|L', None,
    # 		'|Clear segment|c+J', '|Clear all segments|s+c+J', '^analyzer.is_trim_segment|Trim segments|c+K', '^is_group_params|Group by params|c+L']))

    def get_info_st(self):
        return 'gen={}, t={}s, dt={}s, world={}x{}, pixel={}, P={}, sampl={}'.format(self.automaton.gen, self.automaton.time, 1 / self.world.params['T'], SIZEX, SIZEY, PIXEL,
                                                                                     self.world.param_P, self.show_freq)

    def get_angular_st(self):
        if self.is_auto_rotate:
            return 'auto-rotate: {} axes={} sampl={} speed={:.2f}'.format('clockwise' if self.is_ang_clockwise else 'anti-clockwise', self.ang_sides, self.ang_gen, self.ang_speed)
        else:
            return 'not in auto-rotate mode'

    def update_info_bar(self):
        global STATUS
        if self.excess_key:
            # print(self.excess_key)
            self.excess_key = None
        if self.info_type or STATUS or self.is_show_fps:
            info_st = ""
            if STATUS:
                info_st = "\n".join(STATUS)
            elif self.is_show_fps and self.fps:
                info_st = 'FPS: {0:.1f}'.format(self.fps)
            elif self.info_type == 'params':
                info_st = self.world.params2st()
            elif self.info_type == 'animal':
                info_st = self.world.long_name()
            elif self.info_type == 'info':
                info_st = self.get_info_st()
            elif self.info_type == 'angular':
                info_st = self.get_angular_st()
            elif self.info_type == 'stats':
                info_st = 'X axis: {0}, Y axis: {1}'.format(self.analyzer.stat_name(i=self.stats_x), self.analyzer.stat_name(i=self.stats_y))
            elif self.info_type in self.menu_values:
                info_st = "{text} [{value}]".format(text=self.VALUE_TEXT[self.info_type], value=self.get_value_text(self.info_type))
            self.info_bar.config(text=info_st)
            STATUS = []
            self.info_type = None
            if self.clear_job is not None:
                self.win.after_cancel(self.clear_job)
            self.clear_job = self.win.after(5000, self.clear_info)

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


    def run(self):

        counter = 0
        while self.is_loop:
            counter += 1

            controls = dict()
            notes = []
            for msg in MIDI_PORT.iter_pending():
                print(msg)
                try:
                    controls[msg.control] = msg.value
                except:
                    notes.append(msg.note)

            if len(controls.keys()) > 0:
                self.update_midi_controls(controls)

                if list(controls.keys())[0] in NOTES:
                    self.update_midi_notes(list(controls.keys())[0])

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
