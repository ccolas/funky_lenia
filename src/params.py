import numpy as np
import warnings
import mido
import os

CONTROLS = [13, 14,  15, 16, 17, 18, 19, 20, 29, 30, 31, 32, 33, 34, 35, 36, 49, 50, 51, 52, 53, 54, 55, 56, 77, 78, 79, 80, 81, 82, 83, 84]
NOTES = list(np.arange(32, 40)) + list(np.arange(48, 56)) + list(np.arange(64, 72))
ports = mido.get_input_names()
MIDI_PORT = mido.open_input(ports[1])

DELTA_MU = 0.0001
MU = 0.1
RANGE_MU = np.arange(MU - DELTA_MU, MU + DELTA_MU, 2 * DELTA_MU / 128)
DIFF_MU = np.arange(- DELTA_MU * 127 / 2, DELTA_MU * (127 - 127 / 2) + DELTA_MU / 2, DELTA_MU)

DELTA_SIGMA = 0.0001
SIGMA = 0.01
RANGE_SIGMA = np.arange(SIGMA - DELTA_SIGMA, SIGMA + DELTA_SIGMA, 2 * DELTA_SIGMA / 128)
DIFF_SIGMA = np.arange(- DELTA_SIGMA * 127 / 2, DELTA_SIGMA * (127 - 127 / 2) + DELTA_SIGMA / 2, DELTA_MU)


T = 10
RANGE_T = np.arange(2, 50, (50 - 2) / 128)
warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # suppress warning from scipy.ndimage.zoom()

X2, Y2, P2, PIXEL_BORDER = 9, 9, 1, 0  # GoL 6,6,3,1   Lenia Lo 7,7,2,0  Hi 9,9,0,0   1<<9=512
SIZEX, SIZEY, PIXEL = 1 << X2, 1 << Y2, 1 << P2
# PIXEL, PIXEL_BORDER = 1,0; SIZEX, SIZEY = 1280//PIXEL, 720//PIXEL    # 720p HD
# PIXEL, PIXEL_BORDER = 1,0; SIZEX, SIZEY = 1920//PIXEL, 1080//PIXEL    # 1080p HD
MIDX, MIDY = int(SIZEX / 2), int(SIZEY / 2)
DEF_R = max(min(SIZEX, SIZEY) // 4 // 5 * 5, 13)
EPSILON = 1e-10
ROUND = 10
STATUS = []
is_windows = (os.name == 'nt')