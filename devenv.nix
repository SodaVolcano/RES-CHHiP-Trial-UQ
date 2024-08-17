{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  # https://devenv.sh/basics/
  env.SHELL = "zsh";

  # https://devenv.sh/packages/
  packages = with pkgs; [git alejandra];

  # https://devenv.sh/scripts/
  scripts = {
    "python".exec = "poetry run python3";
    "python3".exec = "poetry run python3";
    "py3".exec = "poetry run python3";
    "py".exec = "poetry run python3";
  };

  enterShell = ''
    poetry --version
    python --version
    alejandra . &> /dev/null
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep "2.42.0"
  '';

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/languages/
  languages.nix.enable = true;

  languages.python = {
    enable = true;
    version = "3.12.4";

    poetry = {
      enable = true;
      install = {
        enable = true;
        installRootPackage = false;
        onlyInstallRootPackage = false;
        compile = false;
        quiet = false;
        groups = [];
        ignoredGroups = [];
        onlyGroups = [];
        extras = [];
        allExtras = false;
        verbosity = "no";
      };
    };
  };
  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
