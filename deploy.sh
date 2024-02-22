#!/bin/bash

python run-active-fno1d.py --config=configs/burgers/mf_ensembles.py --config.active.heuristic=predvar --config.anneal.method=exp --config.anneal.alpha=0.005 --workdir=__res_burgers_alpha2__
python run-active-fno1d.py --config=configs/burgers/mf_ensembles.py --config.active.heuristic=mutual_info --config.anneal.method=exp --config.anneal.alpha=0.005 --workdir=__res_burgers_alpha2__
python run-active-fno1d.py --config=configs/burgers/mf_ensembles.py --config.active.heuristic=self_mutual_info --config.anneal.method=exp --config.anneal.alpha=0.005 --workdir=__res_burgers_alpha2__

python run-active-fno1d.py --config=configs/burgers/mf_ensembles.py --config.active.heuristic=random_full --config.anneal.cost_anneal=False --workdir=__res_burgers__
python run-active-fno1d.py --config=configs/burgers/mf_ensembles.py --config.active.heuristic=random_low --config.anneal.cost_anneal=False --workdir=__res_burgers__
python run-active-fno1d.py --config=configs/burgers/mf_ensembles.py --config.active.heuristic=random_high --config.anneal.cost_anneal=False --workdir=__res_burgers__


