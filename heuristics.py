import torch
import numpy as np
import time
import os
import torch.nn.functional as F


from infras.fno_utilities import *
from infras.misc import cprint, create_path, get_logger


def _eval_sf_predvar(pred_samples):
    pred_samples_flat = pred_samples.flatten(2,-1)
    # cprint('r', pred_samples_flat.shape)
    pred_var = pred_samples_flat.std(1).mean(1).square()
    # cprint('b', pred_var.shape)
    return pred_var.data.cpu().numpy()



def eval_mfids_predvar(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is predict variance')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_predvars = []

        for sf_mu_samples in mu_samples:
            # print(sf_mu_samples.shape)
            # sf_pred_var_vrf = _eval_sf_predvar_vrf(sf_mu_samples)
            # print(sf_pred_var_vrf)
            # mf_predvars.append(pred_var)
            sf_pred_var = _eval_sf_predvar(sf_mu_samples)
            # print((sf_pred_var-sf_pred_var).sum())
            mf_predvars.append(sf_pred_var)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_predvars):
            cost_i = fids_costs[fidx]
            nhvals = hvals/cost_i
            vec_fid_idx = np.ones_like(hvals)*fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)

        # cprint('g', mf_predvars)
        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:,0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx,1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)


        return queries_fid, queries_input


def _eval_sf_mixture_entropy(pred_mu_samples, pred_std_samples):

    pred_mu_samples_flat = pred_mu_samples.flatten(2,-1)
    pred_std_samples_flat = pred_std_samples.flatten(2,-1)

    # cprint('r', pred_mu_samples_flat.shape)
    # cprint('b', pred_std_samples_flat.shape)

    pred_mu_star = pred_mu_samples_flat.mean(1)
    pred_var_star = (pred_std_samples_flat**2+pred_mu_samples_flat**2).mean(1) - pred_mu_star**2

    pred_entropy = torch.log(2*np.pi*pred_var_star).sum(1)
    # cprint('g', pred_entropy)

    return pred_entropy.data.cpu().numpy()


def eval_mfids_mixture_entropy(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is mixture entropy')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []

        for sf_mu_samples, sf_std_samples in zip(mu_samples, std_samples):
            sf_hvals = _eval_sf_mixture_entropy(sf_mu_samples, sf_std_samples)
            mf_hvals.append(sf_hvals)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals/cost_i
            vec_fid_idx = np.ones_like(hvals)*fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)


        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:,0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx,1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)

        return queries_fid, queries_input

def _eval_samples_entropy(mu_samples, std_samples):

    var_samples = std_samples ** 2

    mu_hat = mu_samples.mean(0)
    var_hat = var_samples.mean(0)

    K = mu_samples.shape[0]
    A = (mu_samples - mu_hat).T / np.sqrt(K)

    IK = torch.eye(K).to(mu_samples.device)

    logdet_term1 = torch.log(var_hat).sum()
    logdet_term2 = torch.logdet(A.T @ torch.diag(1 / var_hat) @ A + IK)
    entropy = logdet_term1 + logdet_term2

    return entropy

def _eval_sf_full_entropy(pred_mu_samples, pred_std_samples):

    pred_info = []
    for i in range(pred_mu_samples.shape[0]):

        preds_mu = pred_mu_samples[i, ...].flatten(1, -1)
        preds_std = pred_std_samples[i, ...].flatten(1, -1)

        # cprint('r', preds_mu.shape)
        # cprint('b', preds_std.shape)

        Hx = _eval_samples_entropy(preds_mu, preds_std)

        pred_info.append(Hx.item())
    #

    pred_info = np.array(pred_info)

    return pred_info


def eval_mfids_full_entropy(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is full entropy')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []

        for sf_mu_samples, sf_std_samples in zip(mu_samples, std_samples):
            sf_hvals = _eval_sf_full_entropy(sf_mu_samples, sf_std_samples)
            mf_hvals.append(sf_hvals)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals / cost_i
            vec_fid_idx = np.ones_like(hvals) * fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)

        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)

        return queries_fid, queries_input

def eval_mfids_full_entropy_3d(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is full entropy')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []

        for sf_mu_samples, sf_std_samples in zip(mu_samples, std_samples):

            info_buff = []
            for t in range(config.data.T_out):
                # cprint('b', sf_mu_samples[...,t].shape)
                # cprint('b', sf_std_samples[...,t].shape)
                sf_info_t = _eval_sf_full_entropy(
                    sf_mu_samples[...,t],
                    sf_std_samples[...,t]
                )
                info_buff.append(sf_info_t)
            #

            #sf_hvals = _eval_sf_full_entropy(sf_mu_samples, sf_std_samples)
            sf_hvals = sum(info_buff)
            # print(sf_hvals)
            mf_hvals.append(sf_hvals)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals / cost_i
            vec_fid_idx = np.ones_like(hvals) * fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)

        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)

        return queries_fid, queries_input

def _eval_sf_mutual_info(pred_mu_samples, pred_std_samples, Fmu_samples, Fstd_samples):

    Fmu_samples_flat = Fmu_samples.flatten(2,-1)
    Fstd_samples_flat = Fstd_samples.flatten(2,-1)
    assert Fmu_samples_flat.shape[0] == Fstd_samples_flat.shape[0]
    nF = Fmu_samples_flat.shape[0]
    # cprint('r', ns)

    pred_info = []

    for i in range(pred_mu_samples.shape[0]):
        preds_mu = pred_mu_samples[i, ...].flatten(1, -1)
        preds_std = pred_std_samples[i, ...].flatten(1, -1)

        # cprint('b', preds_mu.shape)
        # cprint('b', preds_std.shape)

        # cprint('r', preds_mu.shape)
        # cprint('b', preds_std.shape)

        #
        # cprint('r', Fmu_samples.permute(1,0,2,3).shape)
        # cprint('r', Fstd_samples.permute(1,0,2,3).shape)

        # Fmu = Fmu_samples.permute(1,0,2,3).flatten(1,-1)
        # Fstd = Fstd_samples.permute(1,0,2,3).flatten(1,-1)

        # Fmu = Fmu_samples.permute(1, 0, 2, 3)
        # Fstd = Fstd_samples.permute(1, 0, 2, 3)

        # print(Fmu_samples_flat.shape)
        # print(Fstd_samples_flat.shape)

        Hx = _eval_samples_entropy(preds_mu, preds_std)
        info_list = []

        for s in range(nF):
            Fmu_s = Fmu_samples_flat[s,:]
            Fstd_s = Fstd_samples_flat[s,:]
            # print(Fmu_s.shape, Fstd_s.shape)
            HF = _eval_samples_entropy(Fmu_s, Fstd_s)

            HxF = _eval_samples_entropy(
                torch.hstack([preds_mu, Fmu_s]),
                torch.hstack([preds_std, Fstd_s]),
            )

            # cprint('r', Hx)
            # cprint('b', HF)
            # cprint('g', HxF)

            info_s = Hx + HF - HxF
            info_list.append(info_s)
        #

        info = sum(info_list)

        # print(info)

        pred_info.append(info.item())
    #

    pred_info = np.array(pred_info)

    # print(pred_info)

    return pred_info

def eval_mfids_mutual_info(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is mutual info')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []

        Fidx = np.random.choice(
            mu_samples[-1].shape[0],
            size=config.active.acq_fsamples,
            replace=False
        )

        Fmu_samples = mu_samples[-1][Fidx]
        Fstd_samples = std_samples[-1][Fidx]

        # print(Fmu_samples.shape)
        # print(Fstd_samples.shape)

        for pred_mu_samples, pred_std_samples in zip(mu_samples, std_samples):
            # print(pred_mu_samples.shape, pred_std_samples.shape)
            # cprint('r', pred_std_samples.shape)
            # cprint('b', pred_mu_samples.shape)

            sf_info = _eval_sf_mutual_info(pred_mu_samples, pred_std_samples, Fmu_samples, Fstd_samples)
            mf_hvals.append(sf_info)
            # print(sf_info)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []


        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals / cost_i
            vec_fid_idx = np.ones_like(hvals) * fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)
        # cprint('y', mf_query_mat)

        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)


        return queries_fid, queries_input


def eval_mfids_mutual_info_3d(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is mutual info')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []

        Fidx = np.random.choice(
            mu_samples[-1].shape[0],
            size=config.active.acq_fsamples,
            replace=False
        )

        Fmu_samples = mu_samples[-1][Fidx]
        Fstd_samples = std_samples[-1][Fidx]

        # print(Fmu_samples.shape)
        # print(Fstd_samples.shape)

        for pred_mu_samples, pred_std_samples in zip(mu_samples, std_samples):
            # print(pred_mu_samples.shape, pred_std_samples.shape)
            # cprint('r', pred_std_samples.shape)
            # cprint('b', pred_mu_samples.shape)

            info_buff = []
            for t in range(config.data.T_out):
                # cprint('b', pred_mu_samples[...,t].shape)
                # cprint('b', pred_std_samples[...,t].shape)
                # cprint('r', Fmu_samples[...,t].shape)
                # cprint('r', Fstd_samples[...,t].shape)
                sf_info_t = _eval_sf_mutual_info(
                    pred_mu_samples[...,t],
                    pred_std_samples[...,t],
                    Fmu_samples[...,t],
                    Fstd_samples[...,t]
                )
                info_buff.append(sf_info_t)
            #

            # sf_info = _eval_sf_mutual_info(pred_mu_samples, pred_std_samples, Fmu_samples, Fstd_samples)
            sf_info = sum(info_buff)
            mf_hvals.append(sf_info)
            # print(sf_info)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals / cost_i
            vec_fid_idx = np.ones_like(hvals) * fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)

        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)


        return queries_fid, queries_input

def _eval_sf_self_mutual_info(
        pred_mu_samples,
        pred_std_samples,
        hf_pred_mu_samples,
        hf_pred_std_samples
):
    pred_info = []
    for i in range(pred_mu_samples.shape[0]):
    # for i in range(1):
        preds_mu = pred_mu_samples[i, ...].flatten(1, -1)
        preds_std = pred_std_samples[i, ...].flatten(1, -1)

        hf_preds_mu = hf_pred_mu_samples[i, ...].flatten(1,-1)
        hf_preds_std = hf_pred_std_samples[i, ...].flatten(1,-1)

        # cprint('r', preds_mu.shape)
        # cprint('b', preds_std.shape)
        #
        # cprint('g', hf_preds_mu.shape)
        # cprint('g', hf_preds_std.shape)

        # cprint('r', preds_mu.shape)
        # cprint('r', preds_std.shape)
        #
        # cprint('b', hf_preds_mu.shape)
        # cprint('b', hf_preds_std.shape)

        Hx = _eval_samples_entropy(preds_mu, preds_std)
        HF = _eval_samples_entropy(hf_preds_mu, hf_preds_std)
        HxF = _eval_samples_entropy(
            torch.hstack([preds_mu, hf_preds_mu]),
            torch.hstack([preds_std, hf_preds_std]),
        )

        info = Hx+HF-HxF

        # print(Hx)
        # print(HF)
        # print(HxF)
        # print(info)

        # print(info)

        pred_info.append(info.item())
    #

    pred_info = np.array(pred_info)
    # print(pred_info)

    return pred_info


def eval_mfids_self_mutual_info(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is DMFAL mutual info')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []


        for pred_mu_samples, pred_std_samples in zip(mu_samples, std_samples):
            # print(pred_mu_samples.shape, pred_std_samples.shape)
            # cprint('r', pred_std_samples.shape)
            # cprint('b', pred_mu_samples.shape)

            hf_pred_mu_samples = mu_samples[-1]
            hf_pred_std_samples = std_samples[-1]

            sf_info = _eval_sf_self_mutual_info(
                pred_mu_samples,
                pred_std_samples,
                hf_pred_mu_samples,
                hf_pred_std_samples
            )
            mf_hvals.append(sf_info)
            # print(sf_info)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals / cost_i
            vec_fid_idx = np.ones_like(hvals) * fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)

        # cprint('y', mf_query_mat)

        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)

        return queries_fid, queries_input

def eval_mfids_self_mutual_info_3d(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is DMFAL mutual info')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []


        for pred_mu_samples, pred_std_samples in zip(mu_samples, std_samples):
            # print(pred_mu_samples.shape, pred_std_samples.shape)
            # cprint('r', pred_std_samples.shape)
            # cprint('b', pred_mu_samples.shape)

            hf_pred_mu_samples = mu_samples[-1]
            hf_pred_std_samples = std_samples[-1]

            # cprint('r', pred_mu_samples.shape)
            # cprint('r', pred_std_samples.shape)
            # cprint('b', hf_pred_mu_samples.shape)
            # cprint('b', hf_pred_std_samples.shape)

            info_buff = []
            for t in range(config.data.T_out):
                # cprint('b', pred_mu_samples[...,t].shape)
                # cprint('b', pred_std_samples[...,t].shape)
                # cprint('r', hf_pred_mu_samples[...,t].shape)
                # cprint('r', hf_pred_std_samples[...,t].shape)
                sf_info_t = _eval_sf_self_mutual_info(
                    pred_mu_samples[...,t],
                    pred_std_samples[...,t],
                    hf_pred_mu_samples[...,t],
                    hf_pred_std_samples[...,t]
                )
                info_buff.append(sf_info_t)
            #

            # sf_info = _eval_sf_self_mutual_info(
            #     pred_mu_samples,
            #     pred_std_samples,
            #     hf_pred_mu_samples,
            #     hf_pred_std_samples
            # )
            sf_info = sum(info_buff)
            mf_hvals.append(sf_info)
            # print(sf_info)
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals / cost_i
            vec_fid_idx = np.ones_like(hvals) * fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)

        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)

        return queries_fid, queries_input

def _eval_sf_bald(pred_mu_samples, pred_std_samples):

    pred_info = []
    for i in range(pred_mu_samples.shape[0]):
        preds_mu = pred_mu_samples[i, ...].flatten(1, -1)
        preds_std = pred_std_samples[i, ...].flatten(1, -1)

        # cprint('r', preds_mu.shape)
        # cprint('b', preds_std.shape)

        Hx = _eval_samples_entropy(preds_mu, preds_std)

        pred_info.append(Hx.item())
    #

    pred_info = np.array(pred_info)

    return pred_info

def eval_mfids_bald(config, ensembles_predicts, costs_scheduler):

    with torch.no_grad():

        cprint('r', 'heuristic is BALD')

        mu_samples, std_samples, models_vars = ensembles_predicts
        mf_hvals = []

        for pred_mu_samples, pred_std_samples in zip(mu_samples, std_samples):
            sf_info = _eval_sf_bald(pred_mu_samples, pred_std_samples)
            mf_hvals.append(sf_info)
        #

        assert len(config.data.fids_list) == len(mf_hvals)

        # cprint('r', type(models_vars))

        ensembles_var = np.array(models_vars)
        # cprint('y', ensembles_var)

        for i, fid in enumerate(config.data.fids_list):
            reg_fid = 0.5 * fid * np.mean(np.log(2*np.pi*np.e*ensembles_var))
            mf_hvals[i] = mf_hvals[i] - reg_fid
        #

        fids_costs = costs_scheduler.step()

        mf_query_mat = []

        for fidx, hvals in enumerate(mf_hvals):
            cost_i = fids_costs[fidx]
            nhvals = hvals / cost_i
            vec_fid_idx = np.ones_like(hvals) * fidx
            vec_data_idx = np.arange(nhvals.size)
            sf_query_mat = np.vstack([nhvals, vec_fid_idx, vec_data_idx]).T
            mf_query_mat.append(sf_query_mat)
        #

        mf_query_mat = np.vstack(mf_query_mat)
        cprint('y', mf_query_mat)

        if np.isnan(mf_query_mat).any():
            raise Exception('Error: nan found in acquisition')

        ordered_idx = np.flip(np.argsort(mf_query_mat[:, 0]))
        argmax_idx = ordered_idx[:config.active.batch_budget]

        queries_fid = mf_query_mat[argmax_idx, 1].astype(int)
        queries_input = mf_query_mat[argmax_idx, 2].astype(int)

        # cprint('r', queries_fid)
        # cprint('r', queries_input)

        return queries_fid, queries_input















