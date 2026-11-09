# SOURCE.md — provenance of the files of this package

Refreshed on Morricone on 2026-11-09 16:29 -03 by `experiments/bilby_exact/make_remote_package.sh` of the GOLD
repository (`~/Catalog`, git 585ce42 plus uncommitted changes).

- `code/*.py`: byte-identical copies of `~/Catalog/experiments/bilby_exact/`;
- `code/gold/`: copies of `~/Catalog/gold/__init__.py` and `summaries.py` (the posterior summaries);
- `code/paper.mplstyle`: `~/Catalog/validation/paper.mplstyle` (figure style, LaTeX);
- `inputs/sensitivity_curves/`: `~/Catalog/sensitivity_curves/`;
- `inputs/gold_events_slim.hdf5`: `events/` and `summary/doublet/dl_runaway` of `catalog_O5.hdf5`;
- `inputs/ranking.csv`: all 1023 events by SNR rank, as written by `run_nessai.py` on Morricone;
- `reference/`: Morricone's rank-1 numbers (`validate_remote.py --make-reference`) and corner plot.

```text
79d2b4bc6c4b10bfc219ee4fba3d10f54c21a4a47b1de0b15c8cc9c89338c791  code/common.py
609cf795ef1cf06a441e92f2cd3aba7330a76b61f315741b5817a7f8fe24f4ce  code/export_exact.py
01f0873342baa984a4eabe1b023a66bf2de667ddc5ca14c4f2ba4bd646fe2a7a  code/gold/__init__.py
8aba4a3b74ec7e9dd964d347dae14fe2fc31a0e9f08987bf69e2a7f177f5b585  code/gold/summaries.py
bf3e449fe8e0afbb8043020b339a87fc65d8160bdb9cccc1437ed43fe1ed62ab  code/paper.mplstyle
553caf9961898d42e32661da5c74925cf280133973e9a6d8869dd74a721248cb  code/run_bilby_exact.py
2ceeea6c61e796a38f8f414b97f754738f1624f84331f549e308b0318cdef0be  code/run_nessai.py
232839b4877904917335dfda5cfd4a9ec1f6e97ab91eca8d14edf999bd6cd469  code/validate_remote.py
aacce66c798c134d022e4d5bf75917580f3218b86e9fa3dbf9caf166f31b3e3e  environment/gold_fast_pip_freeze.txt
0f91cb28d0238a8a9292c18146736d07dc883ec7968bc9905337a30811aa1028  environment/gold_fast.yml
17ef06b6ff8b0223e2c9a9ed70f5225d904ffdcc591352945cdb65ee3d00692f  inputs/gold_events_slim.hdf5
6b69c72aee64cc8b2395df24164fe8dc3dc9e064da67367b0e147bbd74ddf930  inputs/ranking.csv
8d6860d053bc7c76805ab0a4593c466391c0bf9c6f0f5607bebd41307ffc2ef5  inputs/remote_events.csv
77c32fff1bffc2489ed18c5fa015a8b0a80bf86c33ca9ed1573b8868f5a7d43d  inputs/sensitivity_curves/KAGRA_O5.txt
e7d1a64ee3cd10645cfe4e7c79ffad2adc7181f6e1b3e6900246adc2a9a5dd0f  inputs/sensitivity_curves/LIGO_O5.txt
66849764830536915c40b3ff93e02727d06e2d80a1ab04c1194802354e95701e  inputs/sensitivity_curves/VIRGO_O5.txt
5e119500c45e222daf4ce6bd7e940aae363570eb572870ff161530ba8778e1ff  reference/corner_rank0001_morricone.png
2aa23b268711da926cb71938b56f366c0ce36cb1705643d0273026dc02b7d1f7  reference/ref_015625.json
871861198cc6f8932b74bb50379fce4de343f500f280a2bec2e4fd516ff7c1a1  reference/ref_015625_posterior.npz
dfc964d8821b5b4f69461643db3b25572af4a203b0e1fd9fd0821174888ad675  reference/scratch/distance_lookup.npz
```
