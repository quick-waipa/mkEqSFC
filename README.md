# mkEqSFC
[English README is here](https://github.com/quick-waipa/mkEqSFC/blob/main/README_ENG.md)  

note記事：スピーカーの音場補正プログラムを自作してみた：  
https://note.com/waipa/n/ncecf0fcbd168
    
***
- Ver.1.00 2024/04/15
   - 新規リリース
- Ver.1.01 2024/04/16
   - ターゲットカーブファイルを通常の周波数特性データのみに適用するようにmkEqSFC.pyとeqMk.pyの内容を変更。
- Ver.1.02 2024/04/17
   - フィルター適用済みターゲットカーブを作成してからEQを計算するように変更。
- Ver.1.03 2024/04/18
   - 普通のcsvファイルが読めるように修正。
- Ver.1.04 2024/04/18
   - グラフの描画をmatplotlibに変更。
- Ver.1.05 2024/04/19
   - 周波数特性データを1000Hzのゲインで自動的に規格化するように修正
- テスト環境: 
   - Windows 10 22H2 (Python 3.11.5)
   - Ubuntu 22.04.3 LTS (Python 3.10.12)
  
***
## 説明：

スピーカーの周波数特性データを元に、音場補正用のEQデータを生成するプログラムです。
従来の音場補正では単純なフラット特性（あるいは単純に多少の傾きをつけたもの）をターゲットにイコライジングしており、結果として補正後の音が曇ったり引っ込んだり音質が劣化して聞こえたりしていましたが、このプログラムでは等ラウドネス曲線などのフィルターを適用し、より精密なターゲットを設定することで、より「人間的に質の良い」補正が可能となっています。


基本的な使い方は以下の通りです。

- まず、インプットデータとしてスピーカーの周波数特性データを用意する（周波数[Hz]、ゲイン[dB]、カンマ区切りデータファイル）。注：20Hz～20000Hzのデータがないとエラーになります。
- mkEqSFC.exeを起動し、各種入力を設定する。
- 「Run Calculate」ボタンを押して計算を実行。アウトプットとして以下が出力されます。

  - 特性フィルター適用後のターゲットカーブデータ
  - 各種プロット画像ファイル（.png）
  - 2種類のEQ設定ファイル(REWエクスポートファイル形式)
  - EQ適用前後のRMS値の変化（rms.txt）

- Equalizer APOなどのソフトでEQファイルを適用します。

ここで、2種類のEQ設定ファイルについて説明します。
EQ設定ファイルは以下の2種類が出力されます。

- オリジナルの周波数特性データに対する音場補正用EQデータ
- 特性フィルター(例えば等ラウドネス曲線)を適用した周波数特性データに対する音場補正用EQデータ

この2種類のEQによる補正を適切な割合でミックスすることで、より良い音場補正が実現できる、という考えです。
たとえばEqualizer APOでは入力音声を二つにわけてそれぞれ異なったEQを適用し、再度ミックスする、ということが可能です。
ミックスする際、ゲイン調整のためにrms.txtに記載されたRMS[<sup>1</sup>]の変化値を参考にしてください。

[<sup>1</sup>]: 実際はシンプソン積分をしているのでRMS値ではないですが便宜上RMSと呼ぶことにします

***
## 入力データ：  
各入力データについて説明します。  
#### [Output Folder]
 + **Output Folder Path:** 出力されるファイルを格納するフォルダのパスを入力して下さい
 
#### [Input Data]
 + **Speaker FR Data File Path:** データ形式：Hz, Gain：カンマ区切り。スピーカーの周波数特性データのファイルパスを入力して下さい。なお、20Hz～20000Hzのデータがないとエラーになりますので注意。
 + **Filter Data File Path:** データ形式：Hz, Gain：カンマ区切り。特性フィルター（等ラウドネス曲線など）のファイルパスを入力して下さい。ISO 226:2023の75phonにおけるデータを用意してあるのでそれを利用するとよいでしょう(ISO_226_2023_75phon.txt)。
 + **EQ Target Curve File Path: Data Format:** データ形式：Hz, Gain：カンマ区切り。EQを作成する際にターゲットとなるカーブのファイルパスを入力して下さい。これは変に傾きをつけるよりもフラットな方が上手くいく気がします(target_curve_Flat.txt)。
 
#### [Output File Name]
 + **EQ(Normal FR)File Name:** EQデータ（通常の周波数特性）が出力されるので、そのファイル名を決めて入力して下さい。
 + **EQ(Filtered FR)File Name:** EQデータ（フィルター後の周波数特性）が出力されるので、そのファイル名を決めて入力して下さい。
 
#### [Application of Characteristic Filter to Frequency Response]
 + **Slope [dB/oct]:** 普段聴いている音楽がどの程度スロープを持つかによって値を変えて下さい。特にこだわりがない場合、ピンクノイズの値を設定するのが良いと思います(-3 dB/oct)。
 
#### [Make EQ Curve]
 + **Band Number:** 作成するEQのバンド数を入力して下さい。
 + **Max Q [-]:** Q値の最大値を設定して下さい。
 + **Min Q [-]:** Q値の最小値を設定して下さい。
 + **Default Q [-]:** 計算がエラーになったときにとりあえず設定するQ値です。4くらいにしておけばいい気がします。
 + **Low Cutoff(Normal FR) [Hz]:** EQ（通常の周波数特性Ver.）作成時に使用する周波数特性データの周波数の下限値を設定して下さい。
 + **High Cutoff(Normal FR) [Hz]:** EQ（通常の周波数特性Ver.）作成時に使用する周波数特性データの周波数の上限値を設定して下さい。
 + **Low Cutoff(Filtered FR) [Hz]:** EQ（フィルター後の周波数特性Ver.）作成時に使用する周波数特性データの周波数の下限値を設定して下さい。
 + **High Cutoff(Filtered FR) [Hz]:** EQ（フィルター後の周波数特性Ver.）作成時に使用する周波数特性データの周波数の上限値を設定して下さい。
 + **Window Octave [oct]:** EQ作成時に周波数特性データの凹凸をガウス関数でフィッティングするのですが、その際にサンプリングする周波数幅を設定します。とりあえず0.1octくらいにしておけばいい気がします。
 + **EQ Creating Target Level [dB]:** EQ作成時のターゲットレベルを設定して下さい。
 
***
## インストールが必要なライブラリ：
以下のライブラリをインストールしてください。

- pandas 
- yaml 
- tkinter 
- ttkthemes 
- numpy 
- scipy  
- matplotlib

インストールには以下のバッチファイル/シェルをご利用下さい。
- Windows: install_win.bat
- Linux: install_linux.sh

***
## プログラムの実行：
プログラムの実行コマンドは以下です。  
    
`python mkEqSFC.py`  

また以下で実行ファイルを作成できます。  
- Windows: build.bat  
- Linux: `sh build.sh`  

注：`config.yaml`は実行ファイルと同じフォルダに入れてください。  
注：Linuxで実行する場合、`mkEqSFC.py` の `create_gui()` 関数内の日本語フォントを書き換えた方がいいかもしれません。

***
## 作成者：
- クイック賄派
- HP: https://quickwaipa.web.fc2.com/
- E-Mail: quickwaipa@gmail.com

## ライセンス
Copyright (c) 2024 Quick-Waipa  
This software is released under the MIT License, see LICENSE.
