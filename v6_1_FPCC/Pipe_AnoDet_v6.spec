# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimport = collect_submodules('tensorflow_core')
data = collect_data_files('tensorflow_core', subdir=None, include_py_files=True)
sys.setrecursionlimit(1000000)

block_cipher = None


a = Analysis(['Pipe_AnoDet_v6.py'],
             pathex=['D:\\Ting-Han\\Project\\工研院委託計畫\\台塑石化管線數據分析\\Official Code\\Source Code\\v6_1_FPCC'],
             binaries=[],
             datas=data,
             hiddenimports=hiddenimport,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Pipe_AnoDet_v6',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True, icon='valve.ico' )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Pipe_AnoDet_v6')
