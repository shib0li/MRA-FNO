# coding=utf-8

import ml_collections
from configs.default_fno1d import get_default_configs

def get_config():


    config = get_default_configs()

    # data
    data = config.data
    data.path = 'data/matfiles'
    data.target_fid = 129
    data.nall = 1200

    data.fids_list = [33, 129]
    data.fids_cost = [0.0031, 0.1276]

    # training
    training = config.training
    training.epochs = 200
    training.use_ratio = False
    training.batch_ratio = 0.1
    training.resample = False
    training.dropout = 0.2

    # testing
    testing = config.testing
    testing.batch_size = 20

    # ensembles
    config.ensembles = ensembles = ml_collections.ConfigDict()
    ensembles.num_ensembles = 5
    ensembles.use_disk = True

    # active
    config.active = active = ml_collections.ConfigDict()
    active.heuristic = 'NA'
    active.init_size = 10
    active.pool_size = 970
    active.batch_budget = 1
    active.acq_fsamples = 7
    active.coresets_dist = 'euc'
    active.steps = 500


    # active
    config.anneal = anneal = ml_collections.ConfigDict()
    anneal.method = 'NA'
    anneal.alpha = 0.00
    anneal.anneal_step = 100
    anneal.cost_anneal = True


    return config

