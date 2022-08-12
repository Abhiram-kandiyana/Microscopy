# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['Count_Annotation_Tool.py'],
             pathex=['C:\\Users\\palakdave\\MedicalImagingProject\\SourceCodes\\MyCodes_Palak\\slicewise_exp\\Leica_images_codes\\count_annotation_tool'],
             binaries=[],
             datas=[],
             hiddenimports=[],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Count_Annotation_Tool',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
