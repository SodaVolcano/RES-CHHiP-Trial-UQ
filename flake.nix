{
  description = "Python project using uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs = {
    self,
    nixpkgs,
    nixpkgs-python,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    pythonVersion = "3.12.4";
    pythonVersionShort = "3.12"; # used to locate /bin/python${version}
  in {
    devShells = forAllSystems (system: let
      python = nixpkgs-python.packages.${system}.${pythonVersion};
      pkgs = nixpkgs.legacyPackages.${system};

      # resolve dynamic library paths
      LD_LIBRARY_PATH = "${(with pkgs;
        lib.makeLibraryPath [
          zlib
          zstd
          stdenv.cc.cc
          curl
          openssl
          attr
          libssh
          bzip2
          libxml2
          acl
          libsodium
          util-linux
          xz
          systemd
          glib.out
        ])}:${pkgs.libGL}/lib"; # libGL don't work for makeLibraryPath
    in {
      default = import ./shell.nix {inherit pkgs LD_LIBRARY_PATH;};
    }); # end mkshell devshell
  };
}
