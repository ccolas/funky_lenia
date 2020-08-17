from pyAudio.pyAudioAnalysis import audioBasicIO
from pyAudio.pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt

OVERLAP = 0.5
WIN_LEN = 50 * 1e-3  # in ms

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
for i in range(nb_pistes):
    Fi, f_names = ShortTermFeatures.feature_extraction(x[i], Fs[i], WIN_LEN * Fs[i], overlap * Fs[i])
    F.append(Fi)

# Plots
for i in range(3):
    plt.subplot(2,1,1)
    plt.plot(F[i][0,:])
    plt.xlabel('Frame no')
    plt.ylabel(f_names[0])
    plt.subplot(2,1,2)
    plt.plot(F[i][1,:])
    plt.xlabel('Frame no')
    plt.ylabel(f_names[1])
plt.show()