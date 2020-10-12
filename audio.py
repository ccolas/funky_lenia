from pyAudio.pyAudioAnalysis import audioBasicIO
from pyAudio.pyAudioAnalysis import MidTermFeatures
import matplotlib.pyplot as plt
import numpy as np

OVERLAP = 0.5
WIN_LEN = 30 * 1e-3 # in ms
MID_WIN_LEN = 2
MID_OVERLAP = 0.5

file1 = "/home/flowers/Desktop/LENIA/Stems/stems ambient/piste1.wav"
file2 = "/home/flowers/Desktop/LENIA/Stems/stems ambient/piste2.wav"
file3 = "/home/flowers/Desktop/LENIA/Stems/stems ambient/piste3.wav"
files = [file1, file2, file3]
nb_pistes = len(files)

# Open files
Fs = []
x = []
for i in range(nb_pistes):
    [Fsi, xi] = audioBasicIO.read_audio_file(files[i])
    Fs.append(Fsi)
    x.append(xi.mean(axis=1))

overlap = OVERLAP * WIN_LEN

# Extract features
F = []
# for i in range(nb_pistes):
#     Fi, f_names = ShortTermFeatures.feature_extraction(signal=x[i],
#                                                        sampling_rate=Fs[i],
#                                                        window=WIN_LEN * Fs[i],
#                                                        step=overlap * Fs[i])
#     F.append(Fi)

mt_F = []
for i in range(nb_pistes):
    mt_Fi, Fi, f_names = MidTermFeatures.mid_feature_extraction(signal=x[i],
                                                         sampling_rate=Fs[i],
                                                         mid_window=MID_WIN_LEN * Fs[i],
                                                         mid_step=MID_OVERLAP * Fs[i],
                                                         short_window=WIN_LEN * Fs[i],
                                                         short_step=overlap * Fs[i])
    mt_F.append(mt_Fi)
    F.append(Fi)



F = np.array(F)
mt_F = np.array(mt_F)

st_energy_channel_2 = F[1, 1, :]
st_spectral_entropy_channel_2 = F[1, 5, :] > 0.3
st_spectral_entropy_channel_2_bis = F[1, 5, :] > 0.7

mt_zcr_mean = mt_F[0, 0, :]
mt_zrc_mean_3 = mt_F[2, 0, :]
SIGNALS = [st_energy_channel_2, st_spectral_entropy_channel_2, mt_zcr_mean, mt_zrc_mean_3, st_spectral_entropy_channel_2_bis]


if __name__ == '__main__':
    for i in range(len(f_names)):
        if 'delta' not in f_names[i] and 'chroma' not in f_names[i]:
            plt.figure()
            for j in range(3):
                plt.plot(mt_F[j][i, :])
            plt.title(f_names[i])
    plt.show()
    for i in range(len(f_names)):
        if 'delta' not in f_names[i] and 'chroma' not in f_names[i]:
            plt.figure()
            for j in range(3):
                plt.plot(F[j][i, :])
            plt.title(f_names[i])
# plt.show()
# # # Plots
# for i in range(3):
#     plt.subplot(2,1,1)
#     plt.plot(F[i][0,:])
#     plt.xlabel('Frame no')
#     plt.ylabel(f_names[0])
#     plt.subplot(2,1,2)
#     plt.plot(F[i][1,:])
#     plt.xlabel('Frame no')
#     plt.ylabel(f_names[1])
# plt.show()