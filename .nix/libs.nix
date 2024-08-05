# Essential packages for python modules, path fixed by fix-python
let pkgs = import (builtins.getFlake "nixpkgs") { };
in with pkgs; [
  gcc.cc
  glibc
  glib   # CV2
  zlib   # numpy, pandas
  libGL  # CV2
  libxkbcommon   # PySide6
  fontconfig     # PySide6
  xorg.libX11         # PySide6
  freetype            # PySide6
  dbus           # Pyside6 >:(
]
