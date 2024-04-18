#-------------------------------------------------------------------------------------
# eqfilter前後のゲイン[dB]の比較プログラム
#-------------------------------------------------------------------------------------

import os
import shutil
import numpy as np
import re
import subprocess
import matplotlib.pyplot as plt
import cmath
from pathlib import Path
from scipy.integrate import simps
from scipy.interpolate import interp1d

def linear_interpolation(freq, gain, target_freq):
    # freqの中でtarget_freqに最も近い2つの周波数を見つける
    idx = 0
    while idx < len(freq) - 1 and freq[idx + 1] < target_freq:
        idx += 1
    
    # 線形補間
    if idx == len(freq) - 1:  # 最後の要素の場合
        interpolated_gain = gain[idx]
    else:
        # 2つの周波数と対応するgainを取得
        freq_lower, freq_upper = freq[idx], freq[idx + 1]
        gain_lower, gain_upper = gain[idx], gain[idx + 1]
        
        # 線形補間
        interpolated_gain = gain_lower + (gain_upper - gain_lower) * ((target_freq - freq_lower) / (freq_upper - freq_lower))
    
    return interpolated_gain

#特に根拠はないがエンジニアリングセンスによってターゲットカーブのローとハイを削る関数
def engineering_sense(freq, gain):
    a_low = [-30, 4, 1.5, 0.2]
    a_hi  = [-2, 6, 1.5, 5]
    
    gain_out = gain + a_low[0]*(a_low[2]/np.log10(freq) - a_low[3])**a_low[1] + a_hi[0]*(a_hi[2]/(np.log10(freq) - a_hi[3]))**a_hi[1]
    
    return gain_out
    

# file read and setting--------------------------------------------------------------
def read_eq_settings(file_path):
    frequencies = []
    gains = []
    q_values = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Filter'):
                if line.split()[2] == "ON":
                    freq = float(line.split()[5])
                    gain = float(line.split()[8])
                    q = float(line.split()[11])
                    frequencies.append(freq)
                    gains.append(gain)
                    q_values.append(q)
                    #print("freq: ",freq," gain: ",gain," q: ",q)
    return frequencies, gains, q_values

# k-file read and setting--------------------------------------------------------------
def read_k_settings(file_path):
    frequencies = []
    gains = []
    with open(file_path, 'r') as file:
        for line in file:
            freq = float(re.split(r'[, \t]+', line)[0])
            gain = float(re.split(r'[, \t]+', line)[1])
            frequencies.append(freq)
            gains.append(gain)
            #print("freq: ",freq," gain: ",gain)
    return frequencies, gains

# msp3 data read and setting--------------------------------------------------------------
def read_msp3(file_path):
    frequencies = []
    gains = []
    with open(file_path, 'r') as file:
        i = 0
        for line in file:
            freq = float(re.split(r'[, \t]+', line)[0])
            gain = float(re.split(r'[, \t]+', line)[1])
            if len(frequencies) != 0:
                freq_pre = frequencies[len(frequencies) - 1]
            else:
                freq_pre = 0
                
            if freq_pre != freq:
                frequencies.append(freq)
                gains.append(gain)
            i += 1
            #print("freq: ",freq," gain: ",gain)
    return frequencies, gains

# calculate equalizing curve--------------------------------------------------------
def calculate_eq_curve(frequencies, gains, q_values, f_range):
    eq_curve = np.full(len(f_range), 0.0)
    Ptotal = np.full(len(f_range), 0.775)
    for freq, gain, q in zip(frequencies, gains, q_values):
        Pi = Ptotal*calculate_peak_filter(freq, gain, q, f_range)
        Gi = 3*np.log2(Pi/0.775)
        eq_curve += Gi
    return eq_curve
    

# calculate peak filter--------------------------------------------------------
def calculate_peak_filter(f0, gain, q, freq):

    w0 = 2 * np.pi * f0 / freq
    w  = 2 * np.pi * 1
    jw = np.zeros(len(freq)) + 1j * w
    g  = 2**(gain/6)
    
    b0 = w0**2 + 1j * np.zeros(len(freq))
    b1 = g*(w0/q) + 1j * np.zeros(len(freq))
    b2 = np.full(len(freq), 1) + 1j * np.zeros(len(freq))
    a0 = w0**2 + 1j * np.zeros(len(freq))
    a1 = 1/g*w0/q + 1j * np.zeros(len(freq))
    a2 = np.full(len(freq), 1) + 1j * np.zeros(len(freq))
        
    H = (b0 + b1 * jw + b2 * jw**2) / (a0 + a1 * jw + a2 * jw**2)
    
    output = np.abs(H)

    return output

# calculate_k_filter-----------------------------------------
def calculate_k_filter_curve(freq):
    fs = 48000

    a1_1 = -1.69065929318241
    a2_1 = 0.73248077421585
    b0_1 = 1.53512485958697
    b1_1 = -2.69169618940638
    b2_1 = 1.19839281085285
    a1_2 = -1.99004745483398
    a2_2 = 0.99007225036621
    b0_2 = 1
    b1_2 = -2
    b2_2 = 1
    
    omega = 2*np.pi * freq / fs
    jw = np.zeros(len(freq)) + 1j * -omega
    exw = np.exp(jw)
    ex2w = np.exp(2 * jw)
    hd_1 = (complex(b0_1,0) + complex(b1_1,0) * exw + complex(b2_1,0) * ex2w) / (complex(1,0) + complex(a1_1,0) * exw + a2_1 * ex2w)
    rlb  = (complex(b0_2,0) + complex(b1_2,0) * exw + complex(b2_2,0) * ex2w) / (complex(1,0) + complex(a1_2,0) * exw + a2_2 * ex2w)
    k = 20*np.log10(np.abs(hd_1)*np.abs(rlb))
    
    return k

# apply k-filter--------------------------------------------------------
def apply_k_filter(eq_curve, k_curve):

    return k_curve + eq_curve

# calc slope--------------------------------------------------------
def calc_slope_curve(freq, slope):
    fr0 = np.log2(1000)
    
    slope_curve = slope*(np.log2(freq) - fr0)
    
    return slope_curve

# apply slope--------------------------------------------------------
def apply_slope(eq_curve, slope_curve):
    
    return eq_curve + slope_curve

# apply filter--------------------------------------------------------
def apply_filter(filter_curve, msp3_curve):
    
    return filter_curve + msp3_curve

# Plot the curves---------------------------------------------------------------
def plot_eq_curve(data, output_folder):
    
    # プロット設定
    plt.figure(figsize=(8, 6))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('Filtered Frequency Response Data')
    plt.xscale('log')
    plt.grid(True)
    plt.xlim(20, 20000)
    plt.ylim(-20, 20)

    # データのプロット
    plt.plot(data[:, 0], data[:, 1], label='Org FR', color='pink')
    plt.plot(data[:, 0], data[:, 2], label='Slope', color='limegreen')
    plt.plot(data[:, 0], data[:, 3], label='Filter', color='lightblue')
    plt.plot(data[:, 0], data[:, 4], label='Filter - Slope', color='steelblue')
    plt.plot(data[:, 0], data[:, 5], label='Filtered FR', color='tomato')

    # 凡例の表示
    plt.legend()

    # グラフの保存
    plt.savefig('filtered_FR_data_plot.png')

    # グラフの表示
    plt.close()
    
    # ファイルを移動し、上書きする
    os.replace("filtered_FR_data_plot.png", output_folder.joinpath("filtered_FR_data_plot.png"))

# calcurate RMS -----------------------------------------------------------------------

def calc_rms(curve, freq):
    
    curve = 0.775*2**(curve/6)
    
    x = np.log10(freq)
    
    rms = simps(curve, x) / (np.log10(20000) - np.log10(20))
    #rms = np.sqrt(np.mean(curve**2))
    
    rms = 6*np.log2(rms/0.775)
    
    return rms
    
# main----------------------------------------------------------------------------------
def eqCalc(eq_path, target_path, k_filter_path, slope):

    f_range = np.logspace(np.log10(20), np.log10(20000), 1000)
    
    f0s, g_eqs, qs = read_eq_settings(eq_path)
    fqs, gs        = read_msp3(target_path)
    fq_ks, g_ks    = read_k_settings(k_filter_path)
    
    interpolator = interp1d(fqs, gs, kind='linear', fill_value="extrapolate")
    target_curve = interpolator(f_range)
    
    eq_curve = calculate_eq_curve(f0s, g_eqs, qs, f_range)
    
    interpolator = interp1d(fq_ks, g_ks, kind='linear', fill_value="extrapolate")
    k_filter_curve = interpolator(f_range)
    
    slope_curve = calc_slope_curve(f_range, slope)
    
    k_filter_curve2 = apply_slope(k_filter_curve, slope_curve)
    
    curve = apply_k_filter(k_filter_curve2, target_curve)
    
    k_filtered_curve = apply_filter(eq_curve, curve)

    rms_k = calc_rms(curve, f_range)
    
    rms_f = calc_rms(k_filtered_curve, f_range)
    
    rms_diff = rms_k - rms_f
    
    return rms_k, rms_f, rms_diff
    
def specCalc(file2_path, file3_path, output_folder, slope):
    
    f_range = np.logspace(np.log10(20), np.log10(20000), 1000)
    
    fq2, g2 = read_k_settings(file2_path)
    fq3, g3 = read_msp3(file3_path)

    interpolator = interp1d(fq2, g2, kind='linear', fill_value="extrapolate")
    k_filter_curve = interpolator(f_range)
    
    interpolator2 = interp1d(fq3, g3, kind='linear', fill_value="extrapolate")
    msp3_curve = interpolator2(f_range)
    
    slope_curve = calc_slope_curve(f_range, slope)
    
    k_filter_curve2 = apply_slope(k_filter_curve, slope_curve)
    
    #target_curve_eqLoudness = -k_filter_curve2
    target_curve_eqLoudness = engineering_sense(f_range, -k_filter_curve2) #エンジニアリングセンスによりローとハイを削ったもの
    
    gain_tmp = linear_interpolation(f_range, target_curve_eqLoudness, 1000)
    target_curve_eqLoudness_std = target_curve_eqLoudness - gain_tmp
    
    target_curve = apply_filter(k_filter_curve2, msp3_curve)
    
    
    data = np.column_stack((f_range, msp3_curve, slope_curve, k_filter_curve, k_filter_curve2, target_curve, target_curve_eqLoudness_std))
    #np.savetxt(out_path, data[:,[0,5]], delimiter=',', fmt='%.6f')
    np.savetxt(output_folder.resolve().joinpath("target_curve_eqLoudness.txt"), data[:,[0,6]], delimiter=',', fmt='%.6f')
    plot_eq_curve(data, output_folder)