{ lib
, stdenv
, fetchurl
, cmake
}:



stdenv.mkDerivation rec {
  pname = "plastimatch";
  version = "1.9.4";

  # src = fetchFromGitLab {
  #   owner = "plastimatch";
  #   repo = "plastimatch";
  #   rev = version;
  #   hash = "sha256-zNPbShNuq4CPqpNXeNjmK8/CZGybMuhxU5iLuZQBDHk=";
  # };

src = fetchurl {
  url = "https://sourceforge.net/projects/plastimatch/files/Source/plastimatch-1.9.4.tar.bz2";
  sha256 = "TBwwk7YPul/RN8ODZxFKma2X3gj7YS5c3uwfgv8ZCUs=";
};

  nativeBuildInputs = [
    cmake
  ];

  meta = with lib; {
    description = "Plastimatch is an open source software for image computation. Our main focus is high-performance volumetric registration, segmentation, and image processing of volumetric medical images";
    homepage = "https://gitlab.com/plastimatch/plastimatch.git";
    license = licenses.unfree; # FIXME: nix-init did not found a license
    maintainers = with maintainers; [ ];
    mainProgram = "plastimatch";
    platforms = platforms.all;
  };
}
