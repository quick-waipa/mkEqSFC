# mkEqSFC
![OutputImage](output/f_equalization_data_plot.png)
[日本語版 README はこちら](https://github.com/quick-waipa/mkEqSFC/blob/main/README.md)

note article： I created my own sound field correction program for speakers (Japanese).：  
https://note.com/waipa/n/ncecf0fcbd168

***
- Ver.1.00 2024/04/15
    - initial release
- Ver.1.01 2024/04/16
    - Changed the contents of the mkEqSFC.py and eqMk.py so that the target curve file applies only to normal frequency response data.
- Ver.1.02 2024/04/17
   - Changed so that EQ is calculated after creating a filtered target curve.
- Ver.1.03 2024/04/18
   - Fixed to be able to read regular csv files.
- Ver.1.04 2024/04/18
   - Changed graph drawing to matplotlib.
- Ver.1.05 2024/04/19
   - Modified to automatically normalize frequency response data with a gain of 1000 Hz
- Ver.1.06 2024/04/19
   - A parameter can be used to specify how much to fill the dip.
- Ver.1.07 2024/04/21
   - Compatible with headphone correction
- Ver.1.08 2024/04/22
   - Modified to calculate the ear canal transfer function in the code.
- Ver.1.09 2024/04/23
   - All comment text, etc., is in English because it may cause errors in some environments.
- Test Environment: 
   - Windows 10 22H2 (Python 3.11.5)
   - Ubuntu 22.04.3 LTS (Python 3.10.12)
  
***
## Description:

This program generates EQ data for sound field correction based on speaker frequency response data.
Conventional sound field correction equalizes by targeting a simple flat characteristic (or simply a slightly skewed one), resulting in the corrected sound sound sounding foggy, tucked in, or degraded in sound quality, but this program applies filters such as equal loudness curves and sets more precise targets, resulting in a more "human quality" correction.

The basic usage is as follows

- First, prepare speaker frequency response data as input data (frequency [Hz], gain [dB], comma-separated data file). Note: Error will occur if there is no data between 20Hz and 20000Hz.
- Start mkEqSFC.exe and set various inputs.
- Press the "Run Calculate" button to execute the calculation. The following will be output as output.

  - Target Curve data after applying the characteristic filter
  - Various plot image files (.png)
  - Two EQ setting files (REW export file format)
  - RMS value changes before and after EQ application (rms.txt)

- Apply the EQ files using software such as Equalizer APO.

Here, the two types of EQ setting files are explained.
The following two types of EQ setting files are output.

- EQ data for sound field correction to the original frequency response data
- EQ data for sound field correction for frequency response data to which a characteristic filter (e.g. equal loudness curve) is applied.

The idea is that better sound field correction can be achieved by mixing these two types of EQ correction in appropriate proportions.
For example, Equalizer APO can split the input audio into two, apply different EQ to each, and mix them again.
When mixing, please refer to the RMS value in the rms.txt file to adjust the gain.

***
## Libraries that need to be installed:
Please install the following libraries.

- pandas 
- yaml 
- tkinter 
- ttkthemes 
- numpy 
- scipy  
- matplotlib

Please use the following batch file/shell script for installation:
- Windows: install_win.bat
- Linux: install_linux.sh

***
## Running the program:
The command to run the program is.  
    
`python mkEqSFC.py`  

You can also create an executable file with  
- Windows: build.bat  
- Linux: `sh build.sh`  

Note: Please place `config.yaml` in the same folder as the executable file.  
***

## Input Data:
![InputImage](image/input.png)
#### [Output Folder]
 + **Output Folder Path:** Please input the path to the folder where output files will be stored.
 
#### [Input Data]
 + **Speaker FR Data File Path:** Data Format: Hz, Gain (comma-separated). Please input the file path of the speaker's frequency response data. Note that an error will occur if there is no data between 20Hz and 20,000Hz.
 + **Filter Data File Path: Data Format:** Hz, Gain (comma-separated). Please input the file path of the characteristic filter (e.g., ISO 226:2023 75phon data can be utilized from ISO_226_2023_75phon.txt).
 + **EQ Target Curve File Path: Data Format:** Hz, Gain (comma-separated). Please input the file path of the curve that serves as the target when creating EQ. It's suggested to use a flat curve rather than adding any tilt (target_curve_Flat.txt). However, in the case of headphone correction, it is necessary to consider the head transfer function, so it is better to set an appropriate target curve (e.g., Harman curve).
 + **HRTF - ECTF File Path: Data Format:** Hz, Gain (comma-separated). Enter only if you are correcting for headphones. Enter the file path of the head transfer function (HRTF). Diffuse Field Curve measured at the eardrum (ISO11904_1_DF_ED.csv).
  
#### [Output File Name]
 + **EQ(Normal FR)File Name:** Input the file name for the EQ data (normal frequency response).
 + **EQ(Filtered FR)File Name:** Input the file name for the EQ data (frequency response after filtering).
 
#### [Application of Characteristic Filter to Frequency Response]
 + **Slope [dB/oct]:** Adjust this value based on how much slope your typical music has. Setting it to the value of pink noise is recommended (-3 dB/oct).
 
#### [Make EQ Curve]
 + **Band Number:** Input the number of bands for the EQ to create.
 + **Max Q [-]:** Set the maximum value for Q.
 + **Min Q [-]:** Set the minimum value for Q.
 + **Default Q [-]:** This is the Q value to set in case of calculation errors. Setting it around 4 is recommended.
 + **Low Cutoff(Normal FR) [Hz]:** Set the lower frequency limit of the frequency response data used when creating EQ (normal frequency response version).
 + **High Cutoff(Normal FR) [Hz]:** Set the upper frequency limit of the frequency response data used when creating EQ (normal frequency response version).
 + **Low Cutoff(Filtered FR) [Hz]:** Set the lower frequency limit of the frequency response data used when creating EQ (filtered frequency response version).
 + **High Cutoff(Filtered FR) [Hz]:** Set the upper frequency limit of the frequency response data used when creating EQ (filtered frequency response version).
 + **Window Octave [oct]:** When creating EQ, this sets the frequency width to sample when fitting the frequency response data with a Gaussian function. Setting it around 0.1 octaves is recommended.
 + **EQ Creating Target Level [dB]:** Set the target level when creating EQ.
 + **How Much to Fill the Dip:** You can specify how much to fill the dip (0.0 to 1.0).
 
***
## Author
- Quick-Waqipa
- HP: https://quickwaipa.web.fc2.com/
- E-Mail: quickwaipa@gmail.com

## License
Copyright (c) 2024 Quick-Waipa  
This software is released under the MIT License, see LICENSE.
