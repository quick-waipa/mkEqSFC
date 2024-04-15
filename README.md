#mkEqSFC
[English README is here](https://github.com/quick-waipa/mkEqSFC/blob/main/README_ENG.md)

***
- Ver.1.00 2024/04/15
 - �V�K�����[�X
- �e�X�g��: 
 - Windows 10 22H2 (Python 3.11.5)
 - Ubuntu 22.04.3 LTS (Python 3.10.12)
  
***
##�����F

�X�s�[�J�[�̎��g�������f�[�^�����ɁA����␳�p��EQ�f�[�^�𐶐�����v���O�����ł��B
�]���̉���␳�ł͒P���ȃt���b�g�����i���邢�͒P���ɑ����̌X�����������́j���^�[�Q�b�g�ɃC�R���C�W���O���Ă���A���ʂƂ��ĕ␳��̉����܂�����������񂾂艹�����򉻂��ĕ��������肵�Ă��܂������A���̃v���O�����ł͓����E�h�l�X�Ȑ��Ȃǂ̃t�B���^�[��K�p���A��萸���ȃ^�[�Q�b�g��ݒ肷�邱�ƂŁA���u�l�ԓI�Ɏ��̗ǂ��v�␳���\�ƂȂ��Ă��܂��B


��{�I�Ȏg�����͈ȉ��̒ʂ�ł��B

- �܂��A�C���v�b�g�f�[�^�Ƃ��ăX�s�[�J�[�̎��g�������f�[�^��p�ӂ���i���g��[Hz]�A�Q�C��[dB]�A�J���}��؂�f�[�^�t�@�C���j�B���F20Hz�`20000Hz�̃f�[�^���Ȃ��ƃG���[�ɂȂ�܂��B
- mkEqSFC.exe���N�����A�e����͂�ݒ肷��B
- �uRun Calculate�v�{�^���������Čv�Z�����s�B�A�E�g�v�b�g�Ƃ��Ĉȉ����o�͂���܂��B

  - �����t�B���^�[�K�p��̎��g�������f�[�^
  - �e��v���b�g�摜�t�@�C���i.png�j
  - 2��ނ�EQ�ݒ�t�@�C��(REW�G�N�X�|�[�g�t�@�C���`��)
  - EQ�K�p�O���RMS�l�̕ω��irms.txt�j

- Equalizer APO�Ȃǂ̃\�t�g��EQ�t�@�C����K�p���܂��B

�����ŁA2��ނ�EQ�ݒ�t�@�C���ɂ��Đ������܂��B
EQ�ݒ�t�@�C���͈ȉ���2��ނ��o�͂���܂��B

- �I���W�i���̎��g�������f�[�^�ɑ΂��鉹��␳�pEQ�f�[�^
- �����t�B���^�[(�Ⴆ�Γ����E�h�l�X�Ȑ�)��K�p�������g�������f�[�^�ɑ΂��鉹��␳�pEQ�f�[�^

����2��ނ�EQ�ɂ��␳��K�؂Ȋ����Ń~�b�N�X���邱�ƂŁA���ǂ�����␳�������ł���A�Ƃ����l���ł��B
���Ƃ���Equalizer APO�ł͓��͉������ɂ킯�Ă��ꂼ��قȂ���EQ��K�p���A�ēx�~�b�N�X����A�Ƃ������Ƃ��\�ł��B
�~�b�N�X����ہA�Q�C�������̂��߂�rms.txt�ɋL�ڂ��ꂽRMS[<sup>1</sup>]�̕ω��l���Q�l�ɂ��Ă��������B

[<sup>1</sup>]: ���ۂ̓V���v�\���ϕ������Ă���̂�RMS�l�ł͂Ȃ��ł����֋X��RMS�ƌĂԂ��Ƃɂ��܂�

***
##���̓f�[�^�F  
�e���̓f�[�^�ɂ��Đ������܂��B  
####[Output Folder]
 + **Output Folder Path:** �o�͂����t�@�C�����i�[����t�H���_�̃p�X����͂��ĉ�����
 
####[Input Data]
 + **Speaker FR Data File Path:** �f�[�^�`���FHz, Gain�F�J���}��؂�B�X�s�[�J�[�̎��g�������f�[�^�̃t�@�C���p�X����͂��ĉ������B���ΓI�ȃQ�C�����x���͂ǂ̂��炢�ł��v�Z��͖�肠��܂��񂪁A�v���b�g�̕`��͈͂�����̂�0dB�t�߂ɂȂ�悤�ɒ����������������ł��傤�B
 + **Filter Data File Path:** �f�[�^�`���FHz Gain�F�X�y�[�X��؂�B�����t�B���^�[�i�����E�h�l�X�Ȑ��Ȃǁj�̃t�@�C���p�X����͂��ĉ������BISO 226:2023��75phon�ɂ�����f�[�^��p�ӂ��Ă���̂ł���𗘗p����Ƃ悢�ł��傤(ISO_226_2023_75phon.txt)�B
 + **EQ Target Curve File Path: Data Format:** �f�[�^�`���FHz, Gain�F�J���}��؂�BEQ���쐬����ۂɃ^�[�Q�b�g�ƂȂ�J�[�u�̃t�@�C���p�X����͂��ĉ������B����͕ςɌX������������t���b�g�ȕ�����肭�����C�����܂�(target_curve_Flat.txt)�B
 
####[Output File Name]
 + **Filter Applied FR Data File Name:** �����t�B���^�[��K�p������̎��g�������f�[�^���o�͂����̂ŁA���̃t�@�C���������߂ē��͂��ĉ������B
 + **EQ(Normal FR)File Name:** EQ�f�[�^�i�ʏ�̎��g�������j���o�͂����̂ŁA���̃t�@�C���������߂ē��͂��ĉ������B
 + **EQ(Filtered FR)File Name:** EQ�f�[�^�i�t�B���^�[��̎��g�������j���o�͂����̂ŁA���̃t�@�C���������߂ē��͂��ĉ������B
 
####[Application of Characteristic Filter to Frequency Response]
 + **Slope [dB/oct]:** ���i�����Ă��鉹�y���ǂ̒��x�X���[�v�������ɂ���Ēl��ς��ĉ������B���ɂ�����肪�Ȃ��ꍇ�A�s���N�m�C�Y�̒l��ݒ肷��̂��ǂ��Ǝv���܂�(-3 dB/oct)�B
 
####[Make EQ Curve]
 + **Band Number:** �쐬����EQ�̃o���h������͂��ĉ������B
 + **Max Q [-]:** Q�l�̍ő�l��ݒ肵�ĉ������B
 + **Min Q [-]:** Q�l�̍ŏ��l��ݒ肵�ĉ������B
 + **Default Q [-]:** �v�Z���G���[�ɂȂ����Ƃ��ɂƂ肠�����ݒ肷��Q�l�ł��B4���炢�ɂ��Ă����΂����C�����܂��B
 + **Low Cutoff(Normal FR) [Hz]:** EQ�i�ʏ�̎��g������Ver.�j�쐬���Ɏg�p������g�������f�[�^�̎��g���̉����l��ݒ肵�ĉ������B
 + **High Cutoff(Normal FR) [Hz]:** EQ�i�ʏ�̎��g������Ver.�j�쐬���Ɏg�p������g�������f�[�^�̎��g���̏���l��ݒ肵�ĉ������B
 + **Low Cutoff(Filtered FR) [Hz]:** EQ�i�t�B���^�[��̎��g������Ver.�j�쐬���Ɏg�p������g�������f�[�^�̎��g���̉����l��ݒ肵�ĉ������B
 + **High Cutoff(Filtered FR) [Hz]:** EQ�i�t�B���^�[��̎��g������Ver.�j�쐬���Ɏg�p������g�������f�[�^�̎��g���̏���l��ݒ肵�ĉ������B
 + **Window Octave [oct]:** EQ�쐬���Ɏ��g�������f�[�^�̉��ʂ��K�E�X�֐��Ńt�B�b�e�B���O����̂ł����A���̍ۂɃT���v�����O������g������ݒ肵�܂��B�Ƃ肠����0.1oct���炢�ɂ��Ă����΂����C�����܂��B
 + **EQ Creating Target Level [dB]:** EQ�쐬���̃^�[�Q�b�g���x����ݒ肵�ĉ������B
 
***
##�C���X�g�[�����K�v�ȃ��C�u�����F
�ȉ��̃��C�u�������C���X�g�[�����Ă��������B

- pandas: 
- yaml: 
- tkinter: 
- ttkthemes: 
- numpy: 
- gnuplot:

�C���X�g�[���ɂ͈ȉ��̃o�b�`�t�@�C��/�V�F���������p�������B
- Windows: install_win.bat
- Linux: install_linux.sh

���s�R�}���h�͈ȉ��ł��B  
`python mkEqSFC.py`

���F`config.yaml`�͎��s�t�@�C���Ɠ����t�H���_�ɓ���Ă��������B  
���FLinux�Ŏ��s����ꍇ�A`mkEqSFC.py` �� `create_gui()` �֐����̓��{��t�H���g����������������������������܂���B

***
##�쐬�ҁF
- �N�C�b�N�d�h
- HP: https://quickwaipa.web.fc2.com/
- E-Mail: quickwaipa@gmail.com

##���C�Z���X
Copyright (c) 2024 Quick-Waipa  
This software is released under the MIT License, see LICENSE.
