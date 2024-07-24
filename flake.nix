{
  description = "Test flake for importing packages";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fix-python.url = "github:GuillaumeDesforges/fix-python";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    fix-python
  }:
    flake-utils.lib.eachDefaultSystem (system: let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };
    in {
      packages = with pkgs; {
        fix-python = fix-python;

     };
    });
}
