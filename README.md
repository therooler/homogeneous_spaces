# Code for quantum gates from homogeneous spaces
This repository contains the code for _Geometric Quantum Machine Learning with Horizontal Quantum Gates_
by Roeland Wiersema, Alexander F. Kemper, Bojko N. Bakalov, Nathan Killoran

Arxiv link: https://arxiv.org/abs/2406.04418

Install Python 3.9,
```bash
conda create --prefix=./homspace python==3.9
```
and install the requirements,
```bash
conda activate ./homspace
pip install -r requirements.txt
```
One can then run the scripts `FIG1_su2.py`, `FIG2_compare_gates.py` and `FIG3_blochsphere.py` to reproduce the 
figures of the paper. One can also run 
```bash
python reproduce_figures.py
```
to produce all the data and plots automatically.

The notebook `A zoo of quantum gates.ipynb` contains a variety of methods to construct the horizontal subspace generated
by a symmetry $K$ acting on a Lie group $G$.
