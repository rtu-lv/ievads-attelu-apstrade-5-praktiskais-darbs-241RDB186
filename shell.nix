let
  indirect = builtins.getFlake
   or (with builtins; findFile nixPath);
in
{
  pkgs ? import (indirect "nixpkgs") {},
  lib ? pkgs.lib,
}:
pkgs.mkShellNoCC {
  packages = with pkgs; [
    python3
    gtk4
  ] ++ (with python3.pkgs; [
    numpy
    opencv-python
    matplotlib
    pygobject3
  ]);

  MPLBACKEND = "gtk4cairo";
}

