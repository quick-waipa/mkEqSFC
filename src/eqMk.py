#-------------------------------------------------------------------------------------
# 周波数特性データから音場補正用のeqfilterを生成するプログラム
#-------------------------------------------------------------------------------------

import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def load_data(file_path):
    """
    Load frequency data from a text file.
    
    Args:
    - file_path (str): Path to the text file.
    
    Returns:
    - DataFrame: Loaded data.
    """
    df = pd.read_csv(file_path, header=None, names=['freq', 'gain'])
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the dataframe.
    
    Args:
    - df (DataFrame): Dataframe containing frequency and gain columns.
    
    Returns:
    - DataFrame: Dataframe with duplicate rows removed.
    """
    df_no_duplicates = df.drop_duplicates()
    return df_no_duplicates

def gaussian_function(x, a, b, c):
    """
    Gaussian function for modeling peaks.
    
    Args:
    - x (array): Input data.
    - a (float): Amplitude.
    - b (float): Center frequency.
    - c (float): Standard deviation.
    
    Returns:
    - array: Gaussian function values.
    """
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def voigt_function(x, amplitude, mean, sigma, gamma):
    from scipy.special import wofz
    from numpy import sqrt, pi
    sigma = float(sigma)
    gamma = float(gamma)
    z = (x - mean + 1j*gamma) / (sigma * sqrt(2))
    return amplitude * wofz(z).real / (sigma * sqrt(2*pi))


def interpolate_gain(df, target_freq):
    """
    Interpolate gain at the target frequency using neighboring frequencies.
    
    Args:
    - df (DataFrame): Dataframe containing frequency and gain columns.
    - target_freq (float): Target frequency for interpolation.
    
    Returns:
    - float: Interpolated gain at the target frequency.
    """
    # Find the neighboring frequencies
    freqs = df['freq'].values
    gains = df['gain'].values
    idx = np.argsort(abs(freqs - target_freq))
    nearest_freqs = freqs[idx[:2]]
    nearest_gains = gains[idx[:2]]
    
    # Perform linear interpolation
    interpolated_gain = np.interp(target_freq, nearest_freqs, nearest_gains)
    return interpolated_gain

def find_peak_and_dip(df, df_t):
    """
    Find the peak and dip from the data.
    
    Args:
    - df (DataFrame): Dataframe containing frequency and gain columns.
    
    Returns:
    - float: Frequency of the peak or dip.
    - float: Gain at the peak or dip frequency.
    """
    
    df_t_tmp = df_t.rename(columns={'gain':'gain_t'})
    merged_df = pd.merge(df, df_t_tmp, on='freq')
    merged_df['gain'] = merged_df['gain'] - merged_df['gain_t']
    df_r = merged_df[['freq','gain']]
    
    # Find the maximum and minimum gain
    max_g = df_r['gain'].max()
    min_g = df_r['gain'].min()
    
    # Check which one is further from the target gain
    max_g_freq = df_r[df_r['gain'] == max_g]['freq'].values[0]
    min_g_freq = df_r[df_r['gain'] == min_g]['freq'].values[0]
    
    #tg_max = df_t.loc[df_t['freq'] == max_g_freq, 'gain'].values[0]
    #tg_min = df_t.loc[df_t['freq'] == min_g_freq, 'gain'].values[0]
    
    
    if abs(max_g) > abs(min_g):
        # Peak is further from the target gain
        return max_g_freq, df.loc[df['freq'] == max_g_freq, 'gain'].values[0]
    else:
        # Dip is further from the target gain
        return min_g_freq, df.loc[df['freq'] == min_g_freq, 'gain'].values[0]

def estimate_neighbor_freq(df_filtered, freq, gain, window_oct):
    """
    Estimate the neighbor frequency using Gaussian peak modeling.
    
    Args:
    - df_filtered (DataFrame): Filtered data.
    - freq (float): Frequency of the peak or dip.
    - gain (float): Gain at the peak or dip frequency.
    
    Returns:
    - float: Estimated neighbor frequency.
    """
    
    # Get frequencies and gains from the filtered dataframe
    freqs = df_filtered['freq'].values
    gains = df_filtered['gain'].values
    
    # Extract data within the window around the peak frequency
    f_hi = 2**(np.log2(freq) + window_oct)
    f_lo = 2**(np.log2(freq) - window_oct)
    idx = (freqs >= f_lo) & (freqs <= f_hi)
    freqs_window = np.log10(freqs[idx])
    gains_window = gains[idx]
    
    # Fit Gaussian function to the data
    sigma = 1/(np.sqrt(2*np.pi)*gain)
    gamma = 1
    p0  = [gain, np.log10(freq), sigma]  # Initial guess for the parameters
    pv0 = [gain, np.log10(freq), sigma, gamma]
    
    try:
        params, _ = curve_fit(gaussian_function, freqs_window, gains_window, p0=p0)
    except RuntimeError:
        print("---Gaussian Curve fitting failed.---")
        amplitude = gain
        
        #try:
        #    params, _ = curve_fit(voigt_function, freqs_window, gains_window, p0=pv0)
        #except RuntimeError:
        #    print("---Voigt Curve fitting failed.---")
        #else:
        #    print("---Complete Voigt fitting---")
        #    amplitude, center_freq, std_dev = params
    else:
        print("---Complete Gaussian fitting---")
        # Get the parameters of the Gaussian function
        amplitude, center_freq, std_dev = params
    
    # Estimate neighbor frequency as the frequency at -3 dB from the peak
    #neighbor_freq = 10**center_freq * 10**((gain - 3) / (20 * amplitude))
    neighbor_freq = freq * 10**((gain - 3) / (20 * amplitude))
    
    return neighbor_freq
    
def estimate_q_value(df_filtered, freq, gain, window_oct, max_q, min_q, default_q):
    # Find the neighboring frequencies closest to 3 dB below the peak/dip gain
    freqs = df_filtered['freq'].values
    gains = df_filtered['gain'].values
    neighbor_freq = estimate_neighbor_freq(df_filtered, freq, gain, window_oct)
    
    #print("neighbor_freq", neighbor_freq)
    
    #print("neighbor_freq :", neighbor_freq)
    
    
    # Check if neighbor_freq is assigned before using it
    if neighbor_freq is None:
        print("Neighbor frequency not found.")
        return None
    
    # Check if neighbor_freq is lower than the peak/dip frequency
    if neighbor_freq < freq:
        print("Neighbor frequency is lower than the peak/dip frequency.")
        q_value = default_q
        return q_value
    
    # Calculate the Q value
    mfreq = 10**(np.log10(freq) - (np.log10(neighbor_freq) - np.log10(freq)))
    oct = np.log2(neighbor_freq / mfreq)
    q_value = 1.41 / oct
    if q_value > max_q:
        q_value = max_q
    elif q_value < min_q:
        q_value = min_q
        
    return q_value


def apply_eq(gains, eq_curve):
    
    return gains + eq_curve 

def mk_eq(f0s, gains, q_values, freqs):
    
    eq_curve = np.full(len(freqs), 0.0)
    Ptotal = np.full(len(freqs), 0.775)
    for f0, gain, q in zip(f0s, gains, q_values):
        Pi = Ptotal*calculate_peak_filter(f0, gain, q, freqs)
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

def plot_data_and_curve(freqs, gains0, gains, eq_curve, t_curve, out, output_folder):

    data = np.column_stack((freqs, gains0))
    eqd_data = np.column_stack((freqs, gains))
    eq_curve = np.column_stack((freqs, eq_curve))
    t_curve = np.column_stack((freqs, t_curve))

    # プロット設定
    plt.figure(figsize=(8, 6))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('Equalization Curve')
    plt.xscale('log')
    plt.grid(True)
    plt.xlim(20, 20000)
    plt.ylim(-20, 20)

    # データのプロット
    plt.plot(data[:, 0], data[:, 1], label='Frequency Respons', color='lightblue', linewidth=2)
    plt.plot(eqd_data[:, 0], eqd_data[:, 1], label='EQd Frequency Respons', color='steelblue', linewidth=2)
    plt.plot(eq_curve[:, 0], eq_curve[:, 1], label='EQ Curve', color='deeppink', linewidth=1)
    plt.plot(t_curve[:, 0], t_curve[:, 1], '--', label='Target Curve', color='tomato')

    # 凡例の表示
    plt.legend()

    # グラフの保存
    plt.savefig('equalization_data_plot.png')

    # グラフの表示
    plt.close()
    
    os.rename("equalization_data_plot.png" ,out + "equalization_data_plot.png" )
    
    # ファイルを移動し、上書きする
    os.replace(out + "equalization_data_plot.png", output_folder.joinpath(out + "equalization_data_plot.png"))
    
    
def write_eq_settings(fs, gs, qs, out_path, model_str):
    
    current_datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    with open(out_path, 'w') as file:
        file.write("Filter Settings file\n\n")
        file.write("Room EQ V5.31\n")
        file.write(f"Dated: {current_datetime}\n\n")
        file.write("Notes:\n\n")
        file.write("Equaliser: Generic\n")
        file.write("FR Data Used: " + model_str + "\n")
        
        for i in range(len(fs)):
            freq = fs[i]
            gain = gs[i]
            q    = qs[i]
            n    = i + 1
            file.write(f"Filter {n}: ON  PK  Fc  {freq:.2f} Hz  Gain  {gain:.2f} dB  Q  {q:.3f}\n")    


def eqMk(data):
    
    #INPUT================================================================

    band_num = data['band_num'] #30 # EQバンド数

    file_path = data['file_path'] #周波数特性データ
    out_path  = data['out_path'] #eq filter
    model_str = data['model_str'] #eq filterファイルの中に書き込むコメント

    max_q     = data['max_q'] # 8 # Q値の最大値
    min_q     = data['min_q'] # 1 # Q値の最小値
    default_q = data['default_q'] # 4 # エラーになったときにとりあえず設定するQ値

    window_oct = data['window_oct'] # 0.1  # ガウス関数でフィッティングするときにどのくらいのオクターブ幅を参照するかという値 [oct]
    
    low_cutoff = data['low_cutoff'] #80  # Low frequency cutoff [Hz]
    high_cutoff = data['high_cutoff'] #3000  # High frequency cutoff [Hz]

    target = data['target'] #-3 #ターゲットゲインレベル [dB]
    target_path = data['target_path']
    
    out = data['out']
    output_folder = data['output_folder']
    
    target_on = data['target_on']

    #=======================================================================
    
    # Load data
    df   = load_data(file_path)
    df_t = load_data(target_path)
    
    # Remove duplicates
    df_no_duplicates = remove_duplicates(df)
    
    
    df_curve = df_no_duplicates
    freqs  = df_curve['freq']
    gains0 = df_curve['gain']
    gains  = df_curve['gain']
    
    interpolator = interp1d(np.log10(df_t['freq']), df_t['gain'], kind='linear', fill_value="extrapolate")
    t_curve = interpolator(np.log10(freqs))
    if target_on:
        t_curve = t_curve + target
    else:
        t_curve = np.zeros_like(freqs) + target
    df_t_curve = pd.DataFrame({'freq':freqs, 'gain':t_curve})
    
    q_values = []
    f0s = []
    eq_gains = [] 
    
    print("===========================================================================")
    print("Generate EQ curve")
    print("Load FR Data:   ", file_path)
    print("Output EQ Data: ", out_path)
    print("===========================================================================")
    
    for i in range(0,band_num,1):
        
        print("band num: ", i + 1)
        
        gains = df_curve['gain']
        
        # Filter the data around the peak or dip frequency
        df_filtered = df_curve[(df_curve['freq'] >= low_cutoff) & (df_curve['freq'] <= high_cutoff)]
        
        # Find the peak or dip
        target_freq, target_gain = find_peak_and_dip(df_filtered, df_t_curve)
        
        print(f"Target frequency: {target_freq} Hz")
        print(f"Target gain: {target_gain} dB")
    
        # Estimate the Q value
        q_value = estimate_q_value(df_curve, target_freq, target_gain, window_oct, max_q, min_q, default_q)
        
        print(f"Estimated Q value: {q_value}")
        
        q_values.append(q_value)
        f0s.append(target_freq)
        eq_gains.append(-target_gain + df_t_curve.loc[df_t_curve['freq'] == target_freq, 'gain'].values[0])

        # make equalizer
        eq_curve = mk_eq(f0s, eq_gains, q_values, freqs)
        
        # apply equalizer
        eqd_curve = apply_eq(gains0, eq_curve)
        
        df_curve = pd.DataFrame({'freq':freqs, "gain":eqd_curve})
        
        print("---------------------------------------")
    
    gains = df_curve['gain']
    
    # write eq settings
    write_eq_settings(f0s, eq_gains, q_values, out_path, model_str)
    
    # Plot data and fitting curve
    plot_data_and_curve(freqs, gains0, gains, eq_curve, t_curve ,out, output_folder)
    