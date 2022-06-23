## Supporting code for Multi-Plasticity Networks.

To reproduce results from the paper, see the directory "paper".

The code contained here can be used to construct networks with mutli-plasticity layers, including the two-layer MPN analyzed in the paper. The notebook `mpn_demonstration.ipynb` demonstrates the following three architures:
- The two-layer MPN analyzed in the paper.
- A three-layer MPN (first two layers have multi-plasticity)
- A recurrent MPN (input and recurrent layers have multi-plasticity).


### LICENSING/CONTRIBUTING NOTE:
The following applies to files in this repo as well as the identically named files within the "paper" directory.

The LICENSING/CONTRIBUTING files listed in this repo only cover the following files:
- `analysis.py`
- `context_data.py`
- `mpn_demonstration.ipynb`
- `paper/mpns_paper.ipynb`

The following files inheret the associated LICENSING/CONTRIBUTING of the forked repo:
- `net_utils.py`
- `networks.py`

The following files inheret the associated LICENSING/CONTRIBUTING from: https://arxiv.org/pdf/2010.15114.pdf
- `int_data.py`
