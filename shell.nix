{
  pkgs,
  LD_LIBRARY_PATH,
}: let
  pythonCommand = "nixGL uv run python \"$@\"";
in
  pkgs.mkShellNoCC {
    inherit LD_LIBRARY_PATH;
    preferLocalBuild = true;
    packages = with pkgs; [
      git
      uv
      alejandra
      pkgs.nixgl.auto.nixGLDefault

      # aliases
      (writeShellScriptBin "py3" pythonCommand)
      (writeShellScriptBin "py" pythonCommand)
      (writeShellScriptBin "python" pythonCommand)
      (writeShellScriptBin "python3" pythonCommand)
      (writeShellScriptBin "black" "uv run black \"$@\"")
      (writeShellScriptBin "pytest" "uv run pytest \"$@\"")
      (writeShellScriptBin "isort" "uv run isort \"$@\"")
    ];

    shellHook = ''
      python --version
      uv sync
      alejandra .
    '';
  }
