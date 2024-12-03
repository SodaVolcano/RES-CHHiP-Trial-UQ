{
  pkgs,
  LD_LIBRARY_PATH,
}:
pkgs.mkShellNoCC {
  inherit LD_LIBRARY_PATH;
  preferLocalBuild = true;
  packages = with pkgs; [
    git
    uv
    alejandra

    # aliases
    (writeShellScriptBin "py3" "uv run python")
    (writeShellScriptBin "py" "uv run python")
    (writeShellScriptBin "python" "uv run python")
    (writeShellScriptBin "python3" "uv run python")
    (writeShellScriptBin "black" "uv run black .")
    (writeShellScriptBin "pytest" "uv run pytest .")
    (writeShellScriptBin "isort" "uv run isort .")
  ];

  shellHook = ''
    python --version
    uv sync
    alejandra .
  '';
}
