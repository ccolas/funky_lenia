import numpy as np
from src.params import *

class Analyzer:
    STAT_NAMES = {'p_m': 'Param m', 'p_s': 'Param s', 'n': 'Gen (#)', 't': 'Time (s)',
                  'm': 'Mass (mg)', 'g': 'Growth (mg/s)', 'r': 'Gyradius (mm)',  # 'I':'Moment of inertia'
                  'd': 'Mass-growth distance (mm)', 's': 'Speed (mm/s)', 'w': 'Angular speed (deg/s)',
                  'm_a': 'Mass asymmetry (mg)'}  # 'm_r':'Mass on right (mg)', 'm_l':'Mass on left (mg)'
    # 'a':'Semi-major axis (mm)', 'b':'Semi-minor axis (mm)', 'e':'Eccentricity', 'c':'Compactness', 'w_th':'Shape angular speed (deg/s)'}
    STAT_HEADERS = ['p_m', 'p_s', 'n', 't', 'm', 'g', 'r', 'd', 's', 'w', 'm_a']  # 'm_r', 'm_l'
    # , 'a', 'b', 'e', 'c', 'w_th']
    SEGMENT_LEN = 20

    def get_stat_row(self):
        R, T, pm, ps = [self.world.params[k] for k in ('R', 'T', 'm', 's')]
        return [pm, ps, self.automaton.gen, self.automaton.time,
                self.mass / R / R, self.growth / R / R, np.sqrt(self.inertia / self.mass),  # self.inertia/R/R  # self.inertia*R*R,
                self.mg_dist, self.m_shift * T, self.m_rotate * T, self.mass_asym / R / R, self.mass_right / R / R, self.mass_left / R / R]

    # self.shape_major_axis, self.shape_minor_axis,
    # self.shape_eccentricity, self.shape_compactness, self.shape_rotate]

    def __init__(self, automaton):
        self.automaton = automaton
        self.world = self.automaton.world
        # self.aaa = self.world.cells
        self.is_trim_segment = True
        self.reset()

    def reset_counters(self):
        self.is_empty = False
        self.is_full = False
        self.mass = 0
        self.growth = 0
        self.inertia = 0
        self.m_center = None
        self.g_center = None
        self.mg_dist = 0
        self.m_shift = 0
        self.m_angle = 0
        self.m_rotate = 0
        self.mass_asym = 0
        self.mass_right = 0
        self.mass_left = 0

    # self.shape_major_axis = 0
    # self.shape_minor_axis = 0
    # self.shape_eccentricity = 0
    # self.shape_compactness = 0
    # self.shape_angle = 0
    # self.shape_rotate = 0

    def reset_last(self):
        self.m_last_center = None
        self.m_center = None
        self.m_last_angle = None

    # self.shape_last_angle = None

    def reset(self):
        self.reset_counters()
        self.reset_last()
        self.clear_series()
        self.last_shift_idx = np.zeros(2)
        self.total_shift_idx = np.zeros(2)

    def calc_stats(self):
        self.m_last_center = self.m_center
        self.m_last_angle = self.m_angle
        # self.shape_last_angle = self.shape_angle
        self.reset_counters()

        R, T = [self.world.params[k] for k in ('R', 'T')]
        A = self.world.cells
        G = np.maximum(self.automaton.field, 0)
        h, w = A.shape
        X, Y = self.automaton.X, self.automaton.Y
        m00 = self.mass = A.sum()
        g00 = self.growth = G.sum()
        self.is_empty = (self.mass < EPSILON)
        self.is_full = (A[0, :].sum() + A[h - 1, :].sum() + A[:, 0].sum() + A[:, w - 1].sum() > 0)

        if m00 > EPSILON:
            AX, AY = A * X, A * Y
            m10, m01 = AX.sum(), AY.sum()
            m20, m02 = (AX * X).sum(), (AY * Y).sum()
            mx, my = self.m_center = np.array([m10, m01]) / m00
            mu20, mu02 = m20 - mx * m10, m02 - my * m01
            self.inertia = mu20 + mu02
            # self.inertia = (mu20 + mu02) / m00**2

            # m11 = (AY*X).sum()
            # mu11 = m11 - mx * m10
            # m1 = mu20 + mu02
            # m2 = mu20 - mu02
            # m3 = 2 * mu11
            # t1 = m1 / 2 / m00
            # t2 = np.sqrt(m2**2 + m3**2) / 2 / m00
            # self.shape_major_axis = t1 + t2
            # self.shape_minor_axis = t1 - t2
            # self.shape_eccentricity = np.sqrt(1 - self.shape_minor_axis / self.shape_major_axis)
            # self.shape_compactness = m00 / (mu20 + mu02)
            # self.shape_angle = np.degrees(np.arctan2(m2, m3))
            # if self.shape_last_angle is not None:
            # self.shape_rotate = self.shape_angle - self.shape_last_angle
            # self.shape_rotate = (self.shape_rotate + 540) % 360 - 180

            if g00 > EPSILON:
                g01, g10 = (G * X).sum(), (G * Y).sum()
                gx, gy = self.g_center = np.array([g01, g10]) / g00
                self.mg_dist = np.linalg.norm(self.m_center - self.g_center)

            if self.m_last_center is not None and self.m_last_angle is not None:
                dm = self.m_center - self.m_last_center + self.last_shift_idx / R
                self.m_shift = np.linalg.norm(dm)
                self.m_angle = np.degrees(np.arctan2(dm[1], dm[0])) if self.m_shift >= EPSILON else 0
                self.m_rotate = self.m_angle - self.m_last_angle
                self.m_rotate = (self.m_rotate + 540) % 360 - 180
                if self.automaton.gen <= 2:
                    self.m_rotate = 0

                midpoint = np.array([MIDX, MIDY])
                X, Y = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
                x0, y0 = self.m_last_center * R + midpoint - self.last_shift_idx
                x1, y1 = self.m_center * R + midpoint
                sign = (x1 - x0) * (Y - y0) - (y1 - y0) * (X - x0)
                self.mass_right = (A[sign > 0]).sum()
                self.mass_left = (A[sign < 0]).sum()
                self.mass_asym = self.mass_right - self.mass_left
            # self.aaa = A.copy(); self.aaa[sign<0] = 0

    def stat_name(self, i=None, x=None):
        if not x: x = self.STAT_HEADERS[i]
        return '{0}={1}'.format(x, self.STAT_NAMES[x])

    def new_segment(self):
        if self.series == [] or self.series[-1] != []:
            self.series.append([])

    def clear_segment(self):
        if self.series != []:
            if self.series[-1] == []:
                self.series.pop()
            if self.series != []:
                self.series[-1] = []

    def invalidate_segment(self):
        if self.series != []:
            self.series[-1] = [[self.world.params['m'], self.world.params['s']] + [np.nan] * (len(self.STAT_HEADERS) - 2)]
            self.new_segment()

    def clear_series(self):
        self.series = []

    def add_stats(self):
        if self.series == []:
            self.new_segment()
        segment = self.series[-1]
        v = self.get_stat_row()
        segment.append(v)
        if self.is_trim_segment:
            while len(segment) > self.SEGMENT_LEN * self.world.params['T']:
                segment.pop(0)

    def center_world(self):
        if self.mass < EPSILON or self.m_center is None:
            return
        self.last_shift_idx = (self.m_center * self.world.params['R']).astype(int)
        self.world.cells = np.roll(self.world.cells, -self.last_shift_idx, (1, 0))
        # self.world.cells = scipy.ndimage.shift(self.world.cells, -self.last_shift_idx, order=0, mode='wrap')
        self.total_shift_idx += self.last_shift_idx
