import pyaudio
import numpy as np



class MicStream():
    def __init__(self, chunk=2**12, rate=44100):

        self.chunk = chunk
        self.rate = rate
        self.p=pyaudio.PyAudio()
        self.stream=self.p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=rate,
                                input=True,
                                frames_per_buffer=chunk)


    def demo(self):
        for i in range(int(10*44100/1024)): #go for a few seconds
            data = np.frombuffer(self.stream.read(self.chunk),dtype=np.int16)
            peak=np.average(np.abs(data))*2
            bars="#"*int(50*peak/2**16)
            print("%04d %05d %s"%(i,peak,bars))

    def get(self, get_peak_fft=False, get_power=False):
        data = np.frombuffer(self.stream.read(self.chunk), dtype=np.int16)
        out = dict(signal=data.copy())

        if get_peak_fft:
            hanning_data = data * np.hanning(len(data))  # smooth the FFT by windowing data
            fft = abs(np.fft.fft(hanning_data).real)
            fft = fft[:int(len(fft) / 2)]  # keep only first half
            freq = np.fft.fftfreq(self.chunk, 1.0 / self.rate)
            freq = freq[:int(len(freq) / 2)]  # keep only first half
            freqPeak = freq[np.where(fft == np.max(fft))[0][0]] + 1
            out['peakfft'] = freqPeak
        if get_power:
            out['power'] = np.average(np.abs(data)) * 2

        return out

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()