import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def white_noise(size, mean=0, std_dev=1):
    return np.random.normal(mean, std_dev, size)

# Пример
size = 1000
wn = white_noise(size)
plt.plot(wn)
plt.title("Белый шум")
plt.show()


def pink_noise(size):
    # Генерация белого шума
    white = np.random.normal(0, 1, size)

    # Генерация розового шума через частотную фильтрацию белого шума
    # Используем метод, при котором накладываем экспоненциальный фильтр на частотные компоненты
    freq = np.fft.fftfreq(size)
    # Нормализуем частоты (предотвращаем деление на 0)
    freq[0] = 1e-6
    spectrum = np.fft.fft(white)

    # Модификация спектра для создания розового шума
    spectrum /= np.sqrt(np.abs(freq))  # Делим спектр на корень из частоты (для розового шума)

    # Преобразуем обратно в временную область
    pink = np.fft.ifft(spectrum)

    return pink.real  # Возвращаем только действительную часть, так как мы работаем с реальными числами


# Пример
size = 1000
pn = pink_noise(size)
plt.plot(pn)
plt.title("Розовый шум")
plt.show()

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)

def white_noise_filtered(size, cutoff, fs):
    white = np.random.normal(0, 1, size)
    return lowpass_filter(white, cutoff, fs)

# Пример
fs = 5000  # Частота дискретизации
cutoff = 1000  # Частота среза
size = 1000
filtered_wn = white_noise_filtered(size, cutoff, fs)
plt.plot(filtered_wn)
plt.title("Белый шум срезанный фильтром")
plt.show()

def impulse_noise(size, density=0.1, magnitude=5):
    noise = np.zeros(size)
    num_impulses = int(size * density)
    for _ in range(num_impulses):
        index = np.random.randint(0, size)
        noise[index] = np.random.choice([magnitude, -magnitude])
    return noise

# Пример
size = 1000
impulse = impulse_noise(size)
plt.plot(impulse)
plt.title("Импульсная помеха")
plt.show()