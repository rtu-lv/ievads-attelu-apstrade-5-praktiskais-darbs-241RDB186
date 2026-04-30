let
  indirect = builtins.getFlake
   or (with builtins; findFile nixPath);
in
{
  pkgs ? import (indirect "nixpkgs") {},
  lib ? pkgs.lib,
}:
pkgs.mkShellNoCC {
  packages = with pkgs.python3.pkgs; with pkgs;  [
    python3

    # Modules
    numpy
    opencv-python
    matplotlib

    # Matplotlib GTK4 backend
    gtk4
    pygobject3
  ];

  MPLBACKEND = "gtk4cairo";
}

