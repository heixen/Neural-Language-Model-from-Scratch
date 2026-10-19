{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.zst";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python3
          python3Packages.pip
          python3Packages.virtualenv
          python3Packages.notebook
          python3Packages.numpy
          python3Packages.pandas
          python3Packages.gensim
          python3Packages.matplotlib
          python3Packages.nltk
          python3Packages.pillow
          python3Packages.black
          python3Packages.scikit-learn
        ];
      };
    };
}
