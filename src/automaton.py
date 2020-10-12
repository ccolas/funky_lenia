import numpy as np
from src.params import *
import reikna.fft, reikna.cluda  # pip3 install pyopencl/pycuda, reikna
import copy
class Automaton:
    kernel_core = {
        0: lambda r: (4 * r * (1 - r)) ** 4,  # polynomial (quad4)
        1: lambda r: np.exp(4 - 1 / (r * (1 - r))),  # exponential / gaussian bump (bump4)
        2: lambda r, q=1 / 4: (r >= q) * (r <= 1 - q),  # step (stpz1/4)
        3: lambda r, q=1 / 4: (r >= q) * (r <= 1 - q) + (r < q) * 0.5  # staircase (life)
    }
    field_func = {
        0: lambda n, m, s: np.maximum(0, 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1,  # polynomial (quad4)
        1: lambda n, m, s: np.exp(- (n - m) ** 2 / (2 * s ** 2)) * 2 - 1,  # exponential / gaussian (gaus)
        2: lambda n, m, s: (np.abs(n - m) <= s) * 2 - 1  # step (stpz)
    }

    def __init__(self, world):
        self.world = world
        self.world_FFT = np.zeros(world.cells.shape)
        self.potential_FFT = np.zeros(world.cells.shape)
        self.potential = np.zeros(world.cells.shape)
        self.field = np.zeros(world.cells.shape)
        self.field_old = None
        self.change = np.zeros(world.cells.shape)
        self.X = dict()
        self.Y = dict()
        self.D = dict()
        self.gen = 0
        self.time = 0
        self.is_multi_step = False
        self.is_soft_clip = False
        self.is_inverted = False
        self.kn = 1
        self.gn = 1
        self.is_gpu = True
        self.has_gpu = True
        self.compile_gpu(self.world.cells)
        self.calc_kernel()

    def kernel_shell(self, r, k_id):
        B = len(self.world.params['b'][k_id])
        Br = B * r
        bs = np.array([float(f) for f in self.world.params['b'][k_id]])
        b = bs[np.minimum(np.floor(Br).astype(int), B - 1)]
        kfunc = Automaton.kernel_core[(self.world.params.get('kn')[k_id] or self.kn) - 1]
        return (r < 1) * kfunc(np.minimum(Br % 1, 1)) * b

    @staticmethod
    def soft_max(x, m, k):
        ''' Soft maximum: https://www.johndcook.com/blog/2010/01/13/soft-maximum/ '''
        return np.log(np.exp(k * x) + np.exp(k * m)) / k

    @staticmethod
    def soft_clip(x, min, max, k):
        a = np.exp(k * x)
        b = np.exp(k * min)
        c = np.exp(-k * max)
        return np.log(1 / (a + b) + c) / -k

    # return Automaton.soft_max(Automaton.soft_max(x, min, k), max, -k)

    def compile_gpu(self, A):
        ''' Reikna: http://reikna.publicfields.net/en/latest/api/computations.html '''
        self.gpu_api = self.gpu_thr = self.gpu_fft = self.gpu_fftshift = None
        try:
            self.gpu_api = reikna.cluda.any_api()
            self.gpu_thr = self.gpu_api.Thread.create()
            self.gpu_fft = reikna.fft.FFT(A.astype(np.complex64)).compile(self.gpu_thr)
            self.gpu_fftshift = reikna.fft.FFTShift(A.astype(np.float32)).compile(self.gpu_thr)
        except Exception as exc:
            # if str(exc) == "No supported GPGPU APIs found":
            self.has_gpu = False
            self.is_gpu = False
            print(exc)
        # raise exc

    def run_gpu(self, A, cpu_func, gpu_func, dtype, **kwargs):
        if self.is_gpu and self.gpu_thr and gpu_func:
            op_dev = self.gpu_thr.to_device(A.astype(dtype))
            gpu_func(op_dev, op_dev, **kwargs)
            return op_dev.get()
        else:
            return cpu_func(A)
        # return np.roll(potential_shifted, (MIDX, MIDY), (1, 0))

    def fft(self, A):
        return self.run_gpu(A, np.fft.fft2, self.gpu_fft, np.complex64)

    def ifft(self, A):
        return self.run_gpu(A, np.fft.ifft2, self.gpu_fft, np.complex64, inverse=True)

    def fftshift(self, A):
        return self.run_gpu(A, np.fft.fftshift, self.gpu_fftshift, np.float32)

    def calc_once(self, is_update=True):
        self.potential_FFT = dict()
        self.potential = dict()
        self.change = dict()
        self.fields = []
        Ds = []
        A = self.world.cells
        dt = 1 / self.world.params['T']
        self.world_FFT = self.fft(A)
        if self.world.nb_kernels > len(list(self.kernel_FFT.keys())):
            self.calc_kernel()
        for k_id in range(self.world.nb_kernels):
            self.potential_FFT[k_id] = self.kernel_FFT[k_id] * self.world_FFT
            self.potential[k_id] = self.fftshift(np.real(self.ifft(self.potential_FFT[k_id])))
            gfunc = Automaton.field_func[(self.world.params.get('gn')[k_id] or self.gn) - 1]
            # m = (np.random.rand(SIZEY, SIZEX) * 0.4 + 0.8) * self.world.params['m']
            # s = (np.random.rand(SIZEY, SIZEX) * 0.4 + 0.8) * self.world.params['s']
            m, s = self.world.params['m'][k_id], self.world.params['s'][k_id]
            field = gfunc(self.potential[k_id], m, s)
            self.fields.append(field)
            if self.is_multi_step and self.field_old:
                Ds.append(1 / 2 * (3 * field - self.field_old))
            else:
                Ds.append(field)

        weights = np.array(self.world.kernels_weights) / np.sum(self.world.kernels_weights)
        res = 0
        for i_w, w in enumerate(weights):
            res += self.fields[i_w] * w
        res /= np.sum(weights)
        self.field_old = res.copy()
        D = 0
        for i_w, w in enumerate(weights):
            D += Ds[i_w] * w
        D /= np.sum(weights)
        if not self.is_soft_clip:
            A_new = np.clip(A + dt * D, 0, 1)  # A_new = A + dt * np.clip(D, -A/dt, (1-A)/dt)
        else:
            A_new = Automaton.soft_clip(A + dt * D, 0, 1, 1 / dt)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
        if self.world.param_P > 0:
            A_new = np.around(A_new * self.world.param_P) / self.world.param_P
        self.change = (A_new - A) / dt

        if is_update:
            self.world.cells = A_new
            self.gen += 1
            self.time = round(self.time + dt, ROUND)
        if self.is_gpu:
            self.gpu_thr.synchronize()


    def calc_kernel(self):
        self.X = dict()
        self.Y = dict()
        self.D = dict()
        self.kernels = dict()
        self.kernels_sums = dict()
        self.kernel_FFT = dict()
        for k_id in range(self.world.nb_kernels):
            I, J = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
            self.X[k_id] = (I - MIDX) / self.world.params['R'][k_id]
            self.Y[k_id] = (J - MIDY) / self.world.params['R'][k_id]
            self.D[k_id] = np.sqrt(self.X[k_id] ** 2 + self.Y[k_id] ** 2)
            self.kernels[k_id] = self.kernel_shell(self.D[k_id], k_id)
            self.kernels_sums[k_id] = self.kernels[k_id].sum()
            kernel_norm = self.kernels[k_id] / self.kernels_sums[k_id]
            self.kernel_FFT[k_id] = self.fft(kernel_norm)
            self.kernel_updated = False

    def reset(self):
        self.gen = 0
        self.time = 0
        self.field_old = None
