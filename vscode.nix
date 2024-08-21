{
  pkgs ?
    import <nixpkgs> {
      config.allowUnfree = true;
    },
}:
pkgs.mkShell {
  # nativeBuildInputs is usually what you want -- tools you need to run
  nativeBuildInputs = with pkgs.buildPackages; [
    vscode
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
