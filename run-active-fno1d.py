import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import hdf5storage

from absl import app
from absl import flags

from ml_collections.config_flags import config_flags
from tqdm.auto import tqdm, trange
import pickle

from infras.misc import cprint, create_path, get_logger
from infras.fno_utilities import *
from infras.exputils import recover_exp_snapshots

from models.mutils1d import *

from heuristics import *
from datasets import MFData1D

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    if config.active.heuristic == 'NA':
        raise Exception('Error: active heuristic not found ...')

    if config.anneal.cost_anneal:
        exp_signature = '{}_annealed_{}'.format(config.active.heuristic, config.anneal.method)
    else:
        exp_signature = config.active.heuristic

    exp_log_workdir = os.path.join(workdir, 'logs')
    exp_evals_workdir = os.path.join(workdir, 'evals', exp_signature)
    exp_dicts_workdir = os.path.join(workdir, 'dicts', exp_signature)

    recovered_step = recover_exp_snapshots(workdir, exp_signature)

    if recovered_step is None:

        cprint('y', 'No valid snapshots founded...')
        create_path(exp_log_workdir, verbose=False)
        create_path(exp_evals_workdir, verbose=False)
        create_path(exp_dicts_workdir, verbose=False)

        logger = get_logger(
            os.path.join(exp_log_workdir, 'log-{}.txt'.format(exp_signature)),
            displaying=False
        )
        logger.info('=============== Experiment Setup ===============')
        logger.info(config)
        logger.info('================================================')

        logger.info('loading data...')

        active_data = MFData1D(config)

        logger.info('Done!')

        hist_fids = []
        hist_costs = []

        costs_scheduler = costs_scheduler_vendor(config, logger)

        init_step = 0

    else:
        cprint('g', 'Valid snapshots founded at step {}...'.format(recovered_step))
        logger = get_logger(
            os.path.join(exp_log_workdir, 'log-{}.txt'.format(exp_signature)),
            displaying=False,
            append=True,
        )
        logger.info('>>>>>>>>>>> Reload exp configs >>>>>>>>>>> \n')
        active_data = MFData1D(config)

        step_evals_dir = os.path.join(exp_evals_workdir, 'step{}'.format(recovered_step))
        step_dicts_dir = os.path.join(exp_dicts_workdir, 'step{}'.format(recovered_step))

        recovered_fids = np.load(os.path.join(step_evals_dir, 'hist_fids_{}.npy'.format(exp_signature)))
        recovered_costs = np.load(os.path.join(step_evals_dir, 'hist_costs_{}.npy'.format(exp_signature)))
        with open(os.path.join(step_evals_dir, 'data_idx_{}.pickle'.format(exp_signature)), 'rb') as handle:
            recovered_data_idx = pickle.load(handle)

        # print('>>>>>>>>>>>>>>', recovered_pool_idx.shape)
        # print(saved_fids)
        # print(saved_costs)

        hist_fids = recovered_fids.tolist()
        hist_costs = recovered_costs.tolist()
        # print(hist_fids)
        # print(hist_costs)
        active_data.fids_train_idx = recovered_data_idx['fids_train_dix']
        active_data.pool_idx = recovered_data_idx['pool_idx']


        costs_scheduler = costs_scheduler_vendor(config, logger)
        costs_scheduler.steps = recovered_step-1

        init_step = recovered_step

        logger.info('<<<<<<<<<<< Resume experiments <<<<<<<<<<< \n')

    #


    for i in tqdm(range(init_step, config.active.steps)):

        logger.info('***** Active Step {} *****'.format(i + 1))

        step_evals_dir = os.path.join(exp_evals_workdir, 'step{}'.format(i + 1))
        step_dicts_dir = os.path.join(exp_dicts_workdir, 'step{}'.format(i + 1))

        create_path(step_evals_dir, verbose=False)
        create_path(step_dicts_dir, verbose=False)

        # _, models_errs = train_mf_fno1d_ensembles(config, active_data, logger, path_dicts=step_dicts_dir)
        # logger.info(' - avg best_rmse={:.5f} ...'.format(models_errs.mean()))

        _, models_errs = train_mf_fno1d_ensembles_vnet(config, active_data, logger, path_dicts=step_dicts_dir)
        logger.info(' - avg best_rmse={:.5f} ...'.format(models_errs.mean()))

        np.save(
            os.path.join(step_evals_dir, 'test_rmse_{}.npy'.format(exp_signature)),
            models_errs
        )

        # models_list = load_model_ensembles(config, step_dicts_dir)
        # Xpool_list, ypool_list = active_data.get_pool_data()


        models_list = load_model_ensembles_vnet(config, step_dicts_dir)
        Xpool_list, ypool_list = active_data.get_pool_data()

        # ensembles_predicts = eval_mf_ensembles_preds(config, Xpool_list, models_list)
        ensembles_predicts = eval_mf_ensembles_preds_vnet(config, Xpool_list, models_list)

        if config.active.heuristic == 'predvar':
            queries_fid, queries_input = eval_mfids_predvar(config, ensembles_predicts, costs_scheduler)
        elif config.active.heuristic == 'mutual_info':
            queries_fid, queries_input = eval_mfids_mutual_info(config, ensembles_predicts, costs_scheduler)
        elif config.active.heuristic == 'self_mutual_info':
            queries_fid, queries_input = eval_mfids_self_mutual_info(config, ensembles_predicts, costs_scheduler)
        elif config.active.heuristic == 'random_full':
            queries_input = np.random.choice(active_data.pool_idx.size, size=config.active.batch_budget, replace=False)
            queries_fid = np.random.choice(len(config.data.fids_list), size=config.active.batch_budget, replace=True)
        elif config.active.heuristic == 'random_low':
            queries_input = np.random.choice(active_data.pool_idx.size, size=config.active.batch_budget, replace=False)
            queries_fid = np.zeros(config.active.batch_budget).astype(int)
        elif config.active.heuristic == 'random_high':
            queries_input = np.random.choice(active_data.pool_idx.size, size=config.active.batch_budget, replace=False)
            queries_fid = ((len(config.data.fids_list) - 1) * np.ones(config.active.batch_budget)).astype(int)
        else:
            raise Exception('Error, heuristic {} not implemented'.format(config.active.heuristic))

        # print(queries_fid, queries_input)

        hist_fids.append(queries_fid[0])
        hist_costs.append(config.data.fids_cost[queries_fid[0]])
        # cprint('g', hist_costs)

        np.save(
            os.path.join(step_evals_dir, 'hist_fids_{}.npy'.format(exp_signature)),
            np.array(hist_fids)
        )

        np.save(
            os.path.join(step_evals_dir, 'hist_costs_{}.npy'.format(exp_signature)),
            np.array(hist_costs)
        )

        logger.info(' - {} selects {} to query at fid {}...'.format(
            exp_signature,
            active_data.pool_idx[queries_input],
            queries_fid
        ))

        active_data.update(queries_input, queries_fid, logger)
        # print(active_data.fids_train_idx)
        data_dict = {
            'fids_train_dix': copy.deepcopy(active_data.fids_train_idx),
            'pool_idx': copy.deepcopy(active_data.pool_idx)
        }
        with open(os.path.join(step_evals_dir, 'data_idx_{}.pickle'.format(exp_signature)), 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
    app.run(main)
