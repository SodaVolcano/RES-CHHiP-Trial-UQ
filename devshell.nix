{
  pkgs ?
    import <nixpkgs> {
      config.allowUnfree = true;
    },
}:
pkgs.mkShell {
  # nativeBuildInputs is usually what you want -- tools you need to run
  propagatedBuildInputs = with pkgs; [
    stdenv.cc.cc.lib
  ];
  postShellHook = ''
    unset LD_LIBRARY_PATH
  '';
  preferLocalBuild = true;
  nativeBuildInputs = with pkgs.buildPackages; [
    autoPatchelfHook
    vim
    (vscode-with-extensions.override {
      vscodeExtensions = with vscode-extensions; [
        bbenoist.nix
        ms-python.vscode-pylance
        ms-python.python
        ms-toolsai.jupyter
        ms-toolsai.jupyter-keymap
        ms-python.debugpy
        ms-python.vscode-pylance
        ms-python.black-formatter
        vscodevim.vim # <33333
        christian-kohler.path-intellisense
        visualstudioexptteam.vscodeintellicode
        usernamehw.errorlens # inline error message
      ];
    })
  ];
}
