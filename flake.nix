{
  description = "Python project using uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs = {
    self,
    nixpkgs,
    nixgl,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
  in {
    devShells = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [nixgl.overlay];
        cudaSupport = true;
      };

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
