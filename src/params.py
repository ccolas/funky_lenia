import numpy as np
import warnings
import mido
import os


knobs_keys = ['T', 'R', 'k_mu', 'k_s', 'beta_1', 'beta_2', 'beta_3', 'beta_4']
knobs_refs = [16, 17, 18, 19, 20, 21, 22, 23]
slide_keys = ['angle']
slide_refs = list(range(1))
KNOBS_KEYS_TO_REFS = dict(zip(knobs_keys + slide_keys, knobs_refs + slide_refs))
KNOBS_REFS_TO_KEYS = dict(zip(knobs_refs + slide_refs, knobs_keys + slide_keys))
knobs_keys = knobs_keys + slide_keys
deltas = [0.5, 1, 1e-4, 1e-4, 1/128, 1/128, 1/128, 1/128, 2.5]
KNOBS_DELTAS = dict(zip(knobs_keys, deltas))

reset_keys = ['random_animal', 'random_world']
reset_refs = [64, 65]
perturb_keys = ['random_patch', 'spawn', 'colormap']
perturb_refs = [66, 67, 68]
flip_keys = ['flip1', 'flip2', 'flip3', 'flip4']
flip_refs = [32, 33, 34, 35]
save_keys = ['save_pos', 'save_neg']
save_refs = [38, 39]
kernel_switch_keys = ['k_left', 'k_right', 'k_more', 'k_less']
kernel_switch_refs = [61, 62, 59, 58]
small_big_switch_key = ['small/big']
small_big_switch_ref = [41]
buttons_keys = reset_keys + perturb_keys + flip_keys + save_keys + kernel_switch_keys + small_big_switch_key
buttons_refs = reset_refs + perturb_refs + flip_refs + save_refs + kernel_switch_refs + small_big_switch_ref
BUTTONS_KEYS_TO_REFS = dict(zip(buttons_keys, buttons_refs))
BUTTONS_REFS_TO_KEYS = dict(zip(buttons_refs, buttons_keys))

ports = mido.get_input_names()
MIDI_PORT = mido.open_input(ports[1])

R_RANGE = [50, 200]
T_RANGE = [1, 50]
BETA_RANGE = [0, 1]
MU_RANGE = [0.10, 0.45]
SIGMA_RANGE = [0.01, 0.15]
ANGLE_RANGE = [0, 360]
RANGES = dict(T=T_RANGE,
              R=R_RANGE,
              k_mu=MU_RANGE,
              k_s=SIGMA_RANGE,
              beta_1=BETA_RANGE,
              beta_2=BETA_RANGE,
              beta_3=BETA_RANGE,
              beta_4=BETA_RANGE,
              angle=ANGLE_RANGE)
# R = 20
# DELTA_R = 0.2
#
# DELTA_MU = 0.001
# MU = 0.1
# RANGE_MU = np.arange(MU - DELTA_MU, MU + DELTA_MU, 2 * DELTA_MU / 128)
# DIFF_MU = np.arange(- DELTA_MU * 127 / 2, DELTA_MU * (127 - 127 / 2) + DELTA_MU / 2, DELTA_MU)
#
# DELTA_SIGMA = 0.0001
# SIGMA = 0.01
# RANGE_SIGMA = np.arange(SIGMA - DELTA_SIGMA, SIGMA + DELTA_SIGMA, 2 * DELTA_SIGMA / 128)
# DIFF_SIGMA = np.arange(- DELTA_SIGMA * 127 / 2, DELTA_SIGMA * (127 - 127 / 2) + DELTA_SIGMA / 2, DELTA_MU)
#
#
# T = 10
# RANGE_T = np.arange(2, 50, (50 - 2) / 128)
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