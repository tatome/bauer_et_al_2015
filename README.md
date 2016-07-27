# bauer_et_al_2015

This git repository contains everything necessary to create the manuscript 
for the paper entitled "Attention Modeled as Information in Learning 
Multisensory Integration" by Bauer, Magg, and Wermter, including
running the experiments.

The experiments can be run and the manuscript can be typeset by simply
typing "make" in this directory.  For this to work, you need GNU make,
LaTeX, latexmk, Python 2.7, and bash.

Running the experiments takes a lot of time, especially on slow
computers.  If you just want to check out the code, have a look at the
python scripts in the subdirectory attention/code/.  The evaluation,
including plotting graphs, is handled by the scripts in 
attention/code/evaluation/.

The experiment is fully parameterized. To change any of the parameters,
edit the file attention/simulation/config.yaml.


Johannes Bauer, July 2014.
