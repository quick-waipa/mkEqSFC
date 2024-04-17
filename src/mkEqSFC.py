#================================================================================
#================================================================================
# Eq data creation program for Sound Field Correction
#================================================================================
#================================================================================
import os
import pandas as pd
import yaml
import tkinter as tk
from tkinter import filedialog
from tkinter import Tk, ttk
from ttkthemes import ThemedTk
from pathlib import Path

import eqCalc
import eqMk

Ver = "1.03"

# ウィンドウを閉じた際にPythonを終了する
def close_window():
    root.destroy()
    root.quit()
    
    
# YAMLファイルからデータを読み込んでGUIに表示する関数
def load_yaml():
    global data_config
    with open(config_file_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    if data_config is None:
        data_config = {}
        
    # GUI 上で表示する順番を指定する
    order = [
        'output_folder',
        'data_file',
        'k_file',
        'target_file',
        #'out',
        'eq1_file',
        'eq2_file',
        'slope',
        'band_num',
        'max_q',
        'min_q',
        'default_q',
        'low_cutoff1',
        'high_cutoff1',
        'low_cutoff2',
        'high_cutoff2',
        'window_oct',
        'target',
    ]
    
    # data_config のキーの順序を取得し、それを parameter_order として使用
    global parameter_order
    parameter_order = list(order)

            
def save_yaml():
    """YAMLファイルにデータを保存する関数"""
    with open(config_file_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f)

def save_data():
    """データを保存する関数"""
    for key, entry in entries.items():
        # 数値データの場合は float 型に変換して保存
        if key == 'band_num':
            try:
                data_config[key] = int(entry.get())
            except ValueError:
                # 数値に変換できない場合はそのまま保存
                data_config[key] = entry.get()
        elif key in numeric_keys:
            try:
                data_config[key] = float(entry.get())
            except ValueError:
                # 数値に変換できない場合はそのまま保存
                data_config[key] = entry.get()
        else:
            # 数値データでない場合はそのまま保存
            data_config[key] = entry.get()
    save_yaml()       
    
def open_file_dialog(entry_widget):
    path = Path(entry_widget.get())
    if path:
        file_path = filedialog.askopenfilename(initialdir=path.parent)
    else:
        file_path = filedialog.askopenfilename(initialdir=Path.cwd())
        
    if file_path:
        entry_widget.delete(0, tk.END)  # エントリウィジェットの現在の内容を削除
        entry_widget.insert(0, file_path)  # 選択されたファイルパスを挿入

def open_folder_dialog(entry_widget):
    path = Path(entry_widget.get())
    if path:
        folder_path = filedialog.askdirectory(initialdir=path)
    else:
        folder_path = filedialog.askdirectory(initialdir=Path.cwd())
        
    if folder_path:
        entry_widget.delete(0, tk.END)  # エントリウィジェットの現在の内容を削除
        entry_widget.insert(0, folder_path)  # 選択されたフォルダパスを挿入
        
#GUIを作成する関数----------------------------------------------------------------------------
def create_gui():
    global root
    root = ThemedTk(theme='adapta')
    root.title("mkEqSFC")
    root.protocol("WM_DELETE_WINDOW", close_window)
    load_yaml()
    
    # カスタムフォントを作成
    font = ("Meiryo", 10)  # フォント名とサイズを指定
    font_b = ("consolas", 10, "bold")
    # 日本語フォントを設定
    font_jp = ("Meiryo", 10)
    
    # ウィンドウのサイズ
    root.geometry("990x920")
    
    #テーマ
    #s = ttk.Style()
    #s.theme_use('black')
    
    # ウィンドウの高さを変更不可能にする
    root.resizable(width=False, height=False)
    
    # 各データを表示する間隔を設定するパディング
    row_padding = 2
    
    parent = ttk.Frame(root)
    parent.pack(fill="both", expand="yes")
    
    # 新しいラベルを作成して配置
    ttk.Label(parent, text=' [Output Folder] --------------------------------------------------------------------------------------------------------------------------', font=font_b).grid(row=0, column=0, columnspan=3, sticky="w", pady=5, padx=0)
    ttk.Label(parent, text=' [Input Data] -----------------------------------------------------------------------------------------------------------------------------', font=font_b).grid(row=2, column=0, columnspan=3, sticky="w", pady=5, padx=0)
    ttk.Label(parent, text=' [Output File Name] -----------------------------------------------------------------------------------------------------------------------', font=font_b).grid(row=6, column=0, columnspan=3, sticky="w", pady=5, padx=0)
    ttk.Label(parent, text=' [Application of Characteristic Filter to Frequency Respons] ------------------------------------------------------------------------------', font=font_b).grid(row=9, column=0, columnspan=3, sticky="w", pady=5, padx=0)
    ttk.Label(parent, text=' [Make EQ Curves] -------------------------------------------------------------------------------------------------------------------------', font=font_b).grid(row=11, column=0, columnspan=3, sticky="w", pady=5, padx=0)
    ttk.Label(parent, text=' ', font=font_b).grid(row=22, column=0, columnspan=3, sticky="w", pady=0, padx=0)
    
    entry_frames = []
    
    # 各データを表示
    row_index = 0
    for key in parameter_order:
        if row_index == 0 or row_index == 2 or row_index == 6 or row_index == 9 or row_index == 11:
            row_index += 1
        
        value = data_config.get(key, "")
        description = parameter_descriptions.get(key, "No description available")
        label = ttk.Label(parent, text=f"{description}:", font=font)
        label.grid(row=row_index, column=0, sticky="e", pady=5, padx=0)
        
        if key in file_path_keys or key in folder_path_keys:
            entry_frame = ttk.Frame(parent)
            entry_frame.grid(row=row_index, column=1,  columnspan=2, sticky="ew", pady=row_padding, padx=5)
            entry_frames.append(entry_frame)
            
            value = int(data_config.get(key, "")) if isinstance(data_config.get(key, ""), int) else data_config.get(key, "")
            entry = ttk.Entry(entry_frame, width=80, font=font)
            entry.insert(0, str(value))
            entry.grid(row=0, column=0, sticky="ew") 
            entries[key] = entry
            
            button_text = "File" if key in file_path_keys else "Folder"
            if key in file_path_keys:
                button = ttk.Button(entry_frame, text="File", command=lambda entry=entry: open_file_dialog(entry))
            else:
                button = ttk.Button(entry_frame, text="Folder", command=lambda entry=entry: open_folder_dialog(entry))
            button.grid(row=0, column=2, padx=None)
            
        elif key in file_name_keys:
            entry = ttk.Entry(parent, width=30, font=font)
            entry.insert(0, str(value))
            entry.grid(row=row_index, column=1, sticky="w", pady=row_padding, padx=5)
            entries[key] = entry
        else:
            entry = ttk.Entry(parent, width=20, font=font)
            entry.insert(0, str(data_config.get(key, "")))
            entry.grid(row=row_index, column=1, sticky="w", pady=row_padding, padx=5)
            entries[key] = entry
        
        com = param_com.get(key, "No Comment")
        label = ttk.Label(parent, text=f"{com}", font=font_jp)
        label.grid(row=row_index, column=2, sticky="w", pady=5, padx=5)
            
        row_index += 1
    
    # Saveボタンを作成
    button_save = ttk.Button(parent, text='Save to Config File', command=save_data)
    button_save.grid(row=row_index + 1, column=0, columnspan=3, pady=8)
    
    # 「Run Calculate」ボタンを作成
    button_ok = ttk.Button(parent, text='Run Calculate', command=calculate)
    button_ok.grid(row=row_index + 2, column=0, columnspan=3, pady=8)
    row_index += 1
    
    # ウィンドウのリサイズに合わせてEntryの幅を広げる
    #root.grid_columnconfigure(2, weight=1) # 列の調整
    #entry_frames[0].grid_columnconfigure(0, weight=1) # 列の調整
    #entry_frames[1].grid_columnconfigure(0, weight=1) # 列の調整
    #entry_frames[2].grid_columnconfigure(0, weight=1) # 列の調整
    #entry_frames[3].grid_columnconfigure(0, weight=1) # 列の調整
    
    root.mainloop()

#メインとなる計算-----------------------------------------------------------------------------
def calculate():
    global data_config
    
    """OKボタンが押されたときの処理"""
    save_data()
    
    with open(config_file_path,'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    #INPUT==========================================================================
    output_folder = Path(config['output_folder'])
    k_file     = Path(config['k_file'])      #なんらかの特性フィルター（等LOUDNESS曲線など）のデータ
    data_file  = Path(config['data_file'])   #入力データのファイル名を指定。周波数特性データ。 ※20～20000Hzのデータがないとエラーになる
    #out        = str(config['out'])         #アウトプットファイル名を指定。slopeと特性フィルターを適用した周波数特性データ
    slope      = float(config['slope'])       #ターゲット生成につかうスロープ[dB/oct] (ピンクノイズ:-3dB/oct)
    #================================================================================
    band_num  =  int(config['band_num'])    # EQバンド数
    eq1_file  =  str(config['eq1_file'])    #アウトプットファイル名を指定。REW形式のeq filterファイル。(フラットターゲットのもの)
    eq2_file  =  str(config['eq2_file'])    #アウトプットファイル名を指定。REW形式のeq filterファイル。(slope + 特性フィルターを適用したターゲットのもの)
    model_str =  str(config['data_file'])   #eq filterファイルの中に書き込むコメント

    max_q     =  float(config['max_q']) # Q値の最大値
    min_q     =  float(config['min_q']) # Q値の最小値
    default_q =  float(config['default_q']) # エラーになったときにとりあえず設定するQ値

    window_oct = float(config['window_oct'])  # ガウス関数でフィッティングするときにどのくらいのオクターブ幅を参照するかという値 [oct]
    
    low_cutoff1 = float(config['low_cutoff1'])  # Low frequency cutoff [Hz]
    high_cutoff1 = float(config['high_cutoff1'])  # High frequency cutoff [Hz]
    
    low_cutoff2 = float(config['low_cutoff2'])  # Low frequency cutoff [Hz]
    high_cutoff2 = float(config['high_cutoff2'])  # High frequency cutoff [Hz]

    target = float(config['target']) #EQ生成のときのターゲットゲインレベル [dB]
    target_file = Path(config['target_file'])

    
    # スクリプトファイルのディレクトリを取得
    script_dir = Path.cwd()
    
    # ファイルパスを絶対パス化
    file2_path   = k_file.resolve()
    file3_path   = data_file.resolve()
    target_path  = target_file.resolve()
    target2_path = output_folder.resolve().joinpath("target_curve_eqLoudness.txt")
    #out_path     = output_folder.resolve().joinpath(out)
    
    print("================================================================================")
    print("================================================================================")
    print("                                   mkEqSFC")
    print("           Eq Data Creation Program for Sound Field Correction")
    print("                                           Made By Quick-Waipa")
    print("                                                      Ver." + Ver)
    print("================================================================================")
    print("================================================================================")
    print("")
    
    
    
    # eqCalc.pyでslope+等ラウドネス曲線を適用した周波数特性データを作成=========
    eqCalc.specCalc(file2_path, file3_path, output_folder, slope)
    
    # EQデータ作成=======================================================
    # フラットターゲット--------------------------------------------------
    eq1_path   = output_folder.joinpath(eq1_file)

    data = {'band_num':band_num,
            'file_path':file3_path,
            'out_path':eq1_path,
            'model_str':model_str,
            'max_q':max_q,
            'min_q':min_q,
            'default_q':default_q,
            'window_oct':window_oct,
            'low_cutoff':low_cutoff1,
            'high_cutoff':high_cutoff1,
            'target':target,
            'target_path':target_path,
            'out':'',
            'output_folder':output_folder,
            'target_on':True,
    }
    eqMk.eqMk(data)
    
    # slope - 等ラウドネス曲線ターゲット-------------------------------------
    eq2_path   = output_folder.joinpath(eq2_file)
    
    data2 = {'band_num':band_num,
            #'file_path':out_path,
            'file_path':file3_path,
            'out_path':eq2_path,
            'model_str':model_str,
            'max_q':max_q,
            'min_q':min_q,
            'default_q':default_q,
            'window_oct':window_oct,
            'low_cutoff':low_cutoff2,
            'high_cutoff':high_cutoff2,
            'target':target,
            'target_path':target2_path,
            'out':'f_',
            'output_folder':output_folder,
            'target_on':True,
    }
    eqMk.eqMk(data2)
    
    #eqCalc.pyでEQフィルター適用後のゲインの増減を計算===========================
    print("================================================")
    print("Calculate RMS")
    print("================================================")
    print("Normal FR Data")
    rms_k1, rms_f1, rms_diff1 = eqCalc.eqCalc(eq1_path, file3_path, file2_path, slope)
    print("------------------------------------------------")
    print("rms_before_EQ-filter[dB]: ", rms_k1)
    print("rms_after_EQ-filter [dB]: ", rms_f1)
    print("rms_diff:           [dB]: ", rms_diff1)
    print("------------------------------------------------")
    print("Filtered FR Data")
    rms_k2, rms_f2, rms_diff2 = eqCalc.eqCalc(eq2_path, file3_path, file2_path, slope)
    print("------------------------------------------------")
    print("rms_before_EQ-filter[dB]: ", rms_k2)
    print("rms_after_EQ-filter [dB]: ", rms_f2)
    print("rms_diff:           [dB]: ", rms_diff2)
    print("------------------------------------------------")
    
    with open("rms.txt", "w") as f:
        f.write("================================================\n")
        f.write("Calculate RMS\n")
        f.write("================================================\n")
        f.write("Normal FR Data\n")
        f.write("------------------------------------------------\n")
        f.write("rms_before_equalization[dB]: " + str(rms_k1) + "\n")
        f.write("rms_after_equalization [dB]: " + str(rms_f1) + "\n")
        f.write("rms_diff:              [dB]: " + str(rms_diff1) +"\n")
        f.write("------------------------------------------------\n")
        f.write("Filtered FR Data\n")
        f.write("------------------------------------------------\n")
        f.write("rms_before_equalization[dB]: " + str(rms_k2) + "\n")
        f.write("rms_after_equalization [dB]: " + str(rms_f2) + "\n")
        f.write("rms_diff:              [dB]: " + str(rms_diff2) + "\n")
        f.write("------------------------------------------------\n")
    
    # ファイルを移動し、上書きする
    os.replace("rms.txt", output_folder.joinpath("rms.txt"))
    
    print("Calculation completed successfully.")
    
    
#メイン関数--------------------------------------------------------------------------------------------
def main():
    # GUIを作成して実行する
    global config_file_path
    global entries, data_config, numeric_keys, parameter_descriptions, param_com, folder_path_keys, file_path_keys, file_name_keys
    entries = {}
    data_config = {}
    config_file_path = "config.yaml"
    numeric_keys = ['slope', 'band_num', 'max_q', 'min_q', 'default_q',
                    'window_oct', 'low_cutoff1', 'high_cutoff1',
                    'low_cutoff2', 'high_cutoff2', 'target']
    
    # 各パラメータのタイトル
    parameter_descriptions = {
        'output_folder': ' Output Folder Path',
        'k_file': ' Filter Data File Path',
        'data_file': ' Speaker FR Data File Path',
        #'out': ' Filter Applied FR Data File Name',
        'slope': ' Slope [dB/oct]',
        'band_num': ' Band Number',
        'eq1_file': ' EQ (Normal FR) File Name',
        'eq2_file': ' EQ (Filtered FR) File Name',
        'max_q': ' Max Q [-]',
        'min_q': ' Min Q [-]',
        'default_q': ' Default Q [-]',
        'window_oct': ' Window Octave [oct]',
        'low_cutoff1': ' Low Cutoff (Normal FR) [Hz]',
        'high_cutoff1': ' High Cutoff (Normal FR) [Hz]',
        'low_cutoff2': '      Low Cutoff (Filtered FR) [Hz]',
        'high_cutoff2': ' High Cutoff (Filtered FR) [Hz]',
        'target': ' EQ Creating Target Level [dB]',
        'target_file': ' EQ Target Curve File Path',
    }

    # 各パラメータに対応するコメント文
    param_com = {
        'output_folder': '',
        'k_file': '',
        'data_file': '',
        #'out': '特性フィルター適用後の周波数特性データの保存ファイル名',
        'slope': 'リファレンス音源のスロープ。(参考：ピンクノイズ：-3 dB/oct)',
        'band_num': 'EQバンド数',
        'eq1_file': 'EQデータ（通常Ve.）の保存ファイル名',
        'eq2_file': 'EQデータ（特性フィルター適用Ver.）の保存ファイル名',
        'max_q': 'Q値の最大値',
        'min_q': 'Q値の最小値',
        'default_q': '計算がエラーになったときにとりあえず設定するQ値',
        'window_oct': 'ガウス関数フィッティング時にサンプリングする周波数幅',
        'low_cutoff1': 'EQ（通常Ver.）作成時の周波数カットオフ下限値',
        'high_cutoff1': 'EQ（通常Ver.）作成時の周波数カットオフ上限値',
        'low_cutoff2': 'EQ（特性フィルター適用Ver.）作成時の周波数カットオフ下限値',
        'high_cutoff2': 'EQ（特性フィルター適用Ver.）作成時の周波数カットオフ上限値',
        'target': 'EQ作成時のターゲットレベル',
        'target_file': '',
    }
    
    # ファイルパスを選択できるデータのキー
    folder_path_keys = ['output_folder']
    file_path_keys = ['k_file', 'data_file', 'target_file']
    #file_name_keys = ['out','eq1_file','eq2_file']
    file_name_keys = ['eq1_file','eq2_file']
    create_gui()
    
    
if __name__ == "__main__":
    main()