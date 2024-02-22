import copy

import torch
import numpy as np
import time
import os
import torch.nn.functional as F

# from models.fno2d import FNO2d
# from models.fno2d_ensemble import FNO2d_Emsemble

from models.fno1d import FNO1d
from models.fno1d_ensemble import FNO1d_Emsemble
from models.fno1d_ensemble_vnet import FNO1d_Emsemble_vnet
from models.fno1d_dropout import FNO1d_Dropout
from models.fno1d_coresets import FNO1d_Coresets

from infras.fno_utilities import *
from infras.misc import cprint, create_path, get_logger


def _train_one_mf_fno1d_original(config, dataset, init_state=None):
    modes = config.model.modes
    width = config.model.width
    model = FNO1d(modes, width).to(config.device)

    batch_ratio = config.training.batch_ratio
    epochs = config.training.epochs
    learning_rate = config.optim.learning_rate
    weight_decay = config.optim.weight_decay
    iters_per_epoch = int(1.0 / batch_ratio)
    iterations = epochs * iters_per_epoch

    # cprint('r', epochs)
    # cprint('r', batch_ratio)
    # cprint('b', iters_per_epoch)
    # cprint('r', iterations)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    best_rmse = np.inf
    best_model = None

    Xtr_list, ytr_list = dataset.get_train_data()
    Xte_list, yte_list = dataset.get_test_data()
    Xte, yte = Xte_list[-1], yte_list[-1]

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xte, yte),
        batch_size=config.testing.batch_size,
        shuffle=False
    )


    myloss = LpLoss(size_average=False)

    for ep in range(epochs):

        for iter in range(iters_per_epoch):

            mfid_loss = []

            optimizer.zero_grad()

            for fid, Xfid, yfid in zip(config.data.fids_list, Xtr_list, ytr_list):
                # cprint('r', fid)
                # cprint('b', Xfid.shape)
                # cprint('b', yfid.shape)

                bz = round(Xfid.shape[0] * batch_ratio)
                batch_idx = np.random.choice(a=Xfid.shape[0], size=bz, replace=False)

                x, y = Xfid[batch_idx, ...], yfid[batch_idx, ...]
                x, y = x.to(config.device), y.to(config.device)
                # cprint('r', x.shape)
                # cprint('b', y.shape)

                out = model(x).reshape(x.shape[0], fid)

                loss_fid = myloss(
                    out.view(out.shape[0], -1),
                    y.view(y.shape[0], -1)
                )

                # print(loss_fid)

                mfid_loss.append(loss_fid)
            #

            # cprint('r', mfid_loss)

            loss = sum(mfid_loss)

            # cprint('b', loss)

            loss.backward()

            optimizer.step()
            scheduler.step()
        #
        with torch.no_grad():

            pred = []

            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                out = model(x).reshape(x.shape[0], fid)
                pred.append(out.data.cpu().numpy())
            #

            pred = np.concatenate(pred)
            np_y_test = yte.data.cpu().numpy()

            # cprint('r', pred.shape)
            # cprint('b', np_y_test.shape)

            direct_rmse = np.sqrt(((np_y_test - pred) ** 2).sum()) / np.sqrt((np_y_test ** 2).sum())

            if direct_rmse < best_rmse:
                best_rmse = direct_rmse
                best_model = copy.deepcopy(model)

            cprint('y', 'epoch={}, rmse={}, best_rmse={}'.format(ep, direct_rmse, best_rmse))

        #
    #

    return best_rmse, best_model


def train_mf_fno1d_original(config, dataset, logger, path_dicts=None):
    if path_dicts is None and config.ensembles.use_disk is True:
        raise Exception('No path found to save the ensembles')

    models_ensembles = []
    models_errs = []

    # for i in trange(config.ensembles.num_ensembles, desc='train ensembles'):
    for i in range(config.ensembles.num_ensembles):

        t_start = time.time()
        best_rmse_i, model_i = _train_one_mf_fno1d_original(config, dataset)
        t_end = time.time()
        logger.info('    ({:3f} secs) ensemble-learner {}, best_rmse={},'.format(t_end - t_start, i + 1, best_rmse_i))

        if config.ensembles.use_disk:
            dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
            torch.save(model_i.state_dict(), os.path.join(path_dicts, dict_name))

        models_ensembles.append(model_i)
        models_errs.append(best_rmse_i)
    #

    models_errs = np.array(models_errs)

    return models_ensembles, models_errs


def _train_one_mf_fno1d_ensemble(config, dataset, init_state=None):
    modes = config.model.modes
    width = config.model.width
    model = FNO1d_Emsemble(modes, width).to(config.device)

    # if init_state is not None:
    #     model.load_state_dict(init_state)
    # else:
    #     cprint('y', 'WARNING: NO INIT STATE FOUND.. FNO2D with new INIT')

    batch_ratio = config.training.batch_ratio
    epochs = config.training.epochs
    learning_rate = config.optim.learning_rate
    weight_decay = config.optim.weight_decay
    iters_per_epoch = int(1.0 / batch_ratio)
    iterations = epochs * iters_per_epoch

    # cprint('r', epochs)
    # cprint('r', batch_ratio)
    # cprint('b', iters_per_epoch)
    # cprint('r', iterations)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    best_rmse = np.inf
    best_model = None

    Xtr_list, ytr_list = dataset.get_train_data()
    Xte_list, yte_list = dataset.get_test_data()
    Xte, yte = Xte_list[-1], yte_list[-1]

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xte, yte),
        batch_size=config.testing.batch_size,
        shuffle=False
    )

    dist_gamma = torch.distributions.Gamma(
        torch.tensor(1.0).to(config.device),
        torch.tensor(1.0).to(config.device)
    )

    myloss = LpLoss(size_average=False)

    for ep in range(epochs):

        for iter in range(iters_per_epoch):

            mfid_loss = []

            optimizer.zero_grad()

            for fid, Xfid, yfid in zip(config.data.fids_list, Xtr_list, ytr_list):
                # cprint('r', fid)
                # cprint('b', Xfid.shape)
                # cprint('b', yfid.shape)

                bz = round(Xfid.shape[0] * batch_ratio)
                batch_idx = np.random.choice(a=Xfid.shape[0], size=bz, replace=False)

                # print(bz)
                # print(batch_idx)

                x, y = Xfid[batch_idx, ...], yfid[batch_idx, ...]
                x, y = x.to(config.device), y.to(config.device)
                # cprint('r', x.shape)
                # cprint('b', y.shape)

                out = model(x).reshape(x.shape[0], fid)

                v = torch.log1p(torch.exp(model.rho)) ** 2

                logprob_prec = dist_gamma.log_prob(1. / v)

                nllh_term1 = y.numel() * (0.5 * torch.log(v)).sum()
                nllh_term2 = 0.5 * torch.sum(torch.square(y - out) / v)

                loss_fid = nllh_term1 + nllh_term2 - logprob_prec

                mfid_loss.append(loss_fid)
            #

            # cprint('r', mfid_loss)

            loss = sum(mfid_loss)

            # cprint('b', loss)

            loss.backward()

            optimizer.step()
            scheduler.step()
        #
        with torch.no_grad():

            pred = []

            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                out = model(x).reshape(x.shape[0], fid)
                pred.append(out.data.cpu().numpy())
            #

            pred = np.concatenate(pred)
            np_y_test = yte.data.cpu().numpy()

            # cprint('r', pred.shape)
            # cprint('b', np_y_test.shape)

            direct_rmse = np.sqrt(((np_y_test - pred) ** 2).sum()) / np.sqrt((np_y_test ** 2).sum())

            if direct_rmse < best_rmse:
                best_rmse = direct_rmse
                best_model = copy.deepcopy(model)

            # cprint('g', 'epoch={}, v={}, rmse={}, best_rmse={}'.format(ep, torch.log1p(torch.exp(model.rho)) ** 2, direct_rmse, best_rmse))

        #
    #

    return best_rmse, best_model


def train_mf_fno1d_ensembles(config, dataset, logger, path_dicts=None):
    if path_dicts is None and config.ensembles.use_disk is True:
        raise Exception('No path found to save the ensembles')

    models_ensembles = []
    models_errs = []

    # for i in trange(config.ensembles.num_ensembles, desc='train ensembles'):
    for i in range(config.ensembles.num_ensembles):

        t_start = time.time()
        best_rmse_i, model_i = _train_one_mf_fno1d_ensemble(config, dataset)
        t_end = time.time()
        logger.info('    ({:3f} secs) ensemble-learner {}, var={}, best_rmse={},'.format(t_end - t_start, i + 1, np.square(np.log1p(np.exp(model_i.rho.item()))), best_rmse_i))

        if config.ensembles.use_disk:
            dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
            torch.save(model_i.state_dict(), os.path.join(path_dicts, dict_name))

        models_ensembles.append(model_i)
        models_errs.append(best_rmse_i)
    #

    models_errs = np.array(models_errs)

    return models_ensembles, models_errs


def load_model_ensembles(config, path_dicts=None):
    if path_dicts is None:
        raise Exception('no valid path found')

    models_list = []
    for i in range(config.ensembles.num_ensembles):
        modes = config.model.modes
        width = config.model.width
        model = FNO1d_Emsemble(modes, width)
        dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
        cprint('r', 'loading {} from {}'.format(dict_name, path_dicts))
        model.load_state_dict(torch.load(os.path.join(path_dicts, dict_name)))
        model.to(config.device)
        models_list.append(model)
    #

    return models_list


def eval_mf_ensembles_preds(config, X_list, models_list):

    with torch.no_grad():

        assert len(models_list) == config.ensembles.num_ensembles

        pred_mu_samples_list = []
        pred_std_samples_list = []

        ensembles_var = []

        for model_i in models_list:
            ensembles_var.append(np.log1p(np.exp(model_i.rho.item())) ** 2)
        #

        for fid, X in zip(config.data.fids_list, X_list):
            # cprint('r', fid)
            # cprint('b', X.shape)

            sf_data_loader = torch.utils.data.DataLoader(
                X,
                batch_size=config.testing.batch_size,
                shuffle=False
            )

            sf_pred_mu_samples = []
            sf_pred_std_samples = []

            for i, model_i in enumerate(models_list):

                pred_mu_i = []
                pred_std_i = []

                for x in sf_data_loader:
                    x = x.to(config.device)
                    out = model_i(x).reshape(x.shape[0], fid)
                    # cprint('b', out.shape)

                    pred_rho = model_i.rho
                    pred_std = torch.log1p(torch.exp(pred_rho)) * torch.ones_like(out).to(config.device)
                    # print(pred_std)
                    pred_mu_i.append(out)
                    pred_std_i.append(pred_std)
                #

                pred_mu_i = torch.cat(pred_mu_i)
                pred_std_i = torch.cat(pred_std_i)
                # cprint('g', pred_mu_i.shape)
                # cprint('r', pred_std_i.shape)
                # print(pred_std_i)

                sf_pred_mu_samples.append(pred_mu_i.unsqueeze(0))
                sf_pred_std_samples.append(pred_std_i.unsqueeze(0))
            #

            # cprint('g', torch.cat(sf_pred_mu_samples).shape)
            # cprint('b', torch.cat(sf_pred_std_samples).shape)

            sf_pred_mu_samples = torch.cat(sf_pred_mu_samples).permute(1,0,2)
            sf_pred_std_samples = torch.cat(sf_pred_std_samples).permute(1,0,2)

            # cprint('g', sf_pred_mu_samples.shape)
            # cprint('b', sf_pred_std_samples.shape)

            pred_mu_samples_list.append(sf_pred_mu_samples)
            pred_std_samples_list.append(sf_pred_std_samples)
        #

        return pred_mu_samples_list, pred_std_samples_list, ensembles_var


#========================================================================================#

def _train_one_mf_fno1d_ensemble_vnet(config, dataset, init_state=None):
    modes = config.model.modes
    width = config.model.width
    model = FNO1d_Emsemble_vnet(modes, width, config.data.fids_list).to(config.device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)

    # if init_state is not None:
    #     model.load_state_dict(init_state)
    # else:
    #     cprint('y', 'WARNING: NO INIT STATE FOUND.. FNO2D with new INIT')

    batch_ratio = config.training.batch_ratio
    epochs = config.training.epochs
    learning_rate = config.optim.learning_rate
    weight_decay = config.optim.weight_decay
    iters_per_epoch = int(1.0 / batch_ratio)
    iterations = epochs * iters_per_epoch

    # cprint('r', epochs)
    # cprint('r', batch_ratio)
    # cprint('b', iters_per_epoch)
    # cprint('r', iterations)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    best_rmse = np.inf
    best_model = None

    Xtr_list, ytr_list = dataset.get_train_data()
    Xte_list, yte_list = dataset.get_test_data()
    Xte, yte = Xte_list[-1], yte_list[-1]

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xte, yte),
        batch_size=config.testing.batch_size,
        shuffle=False
    )

    dist_gamma = torch.distributions.Gamma(
        torch.tensor(1.).to(config.device),
        torch.tensor(1.).to(config.device)
    )

    myloss = LpLoss(size_average=False)

    for ep in range(epochs):

        for iter in range(iters_per_epoch):

            mfid_loss = []

            optimizer.zero_grad()

            for fid, Xfid, yfid in zip(config.data.fids_list, Xtr_list, ytr_list):
                # cprint('r', fid)
                # cprint('b', Xfid.shape)
                # cprint('b', yfid.shape)

                bz = round(Xfid.shape[0] * batch_ratio)
                batch_idx = np.random.choice(a=Xfid.shape[0], size=bz, replace=False)

                # print(bz)
                # print(batch_idx)

                x, y = Xfid[batch_idx, ...], yfid[batch_idx, ...]
                x, y = x.to(config.device), y.to(config.device)
                # cprint('r', x.shape)
                # cprint('b', y.shape)

                out, rho_out = model(x, fid)

                out = out.reshape(x.shape[0], fid)

                # cprint('r', out.shape)
                # cprint('b', rho_out.shape)

                # v = torch.log1p(torch.exp(model.rho)) ** 2
                v_out = torch.log1p(torch.exp(rho_out)) ** 2

                # logprob_prec = dist_gamma.log_prob(1. / v)
                logprob_prec = dist_gamma.log_prob(1. / v_out)

                nllh_term1 = (0.5 * torch.log(v_out)).sum()
                # cprint('y', torch.square(y - out).shape)
                # cprint('y', v_out.shape)
                nllh_term2 = 0.5 * torch.sum(torch.square(y - out) / v_out)

                loss_fid = nllh_term1 + nllh_term2 - logprob_prec.sum()

                mfid_loss.append(loss_fid)
            #

            # cprint('r', mfid_loss)

            loss = sum(mfid_loss)

            # cprint('b', loss)

            loss.backward()

            optimizer.step()
            scheduler.step()
        #
        with torch.no_grad():

            pred = []

            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                # out = model(x).reshape(x.shape[0], fid)
                out, rho_out = model(x, fid)
                out = out.reshape(x.shape[0], fid)
                pred.append(out.data.cpu().numpy())
            #

            pred = np.concatenate(pred)
            np_y_test = yte.data.cpu().numpy()

            # cprint('r', pred.shape)
            # cprint('b', np_y_test.shape)

            direct_rmse = np.sqrt(((np_y_test - pred) ** 2).sum()) / np.sqrt((np_y_test ** 2).sum())

            if direct_rmse < best_rmse:
                best_rmse = direct_rmse
                best_model = copy.deepcopy(model)

            # cprint('r', 'epoch={}, v={}, rmse={}, best_rmse={}'.format(ep, torch.log1p(torch.exp(model.rho)) ** 2, direct_rmse, best_rmse))
            # cprint('r', 'epoch={}, rmse={}, best_rmse={}'.format(ep, direct_rmse, best_rmse))
        #
    #

    return best_rmse, best_model

def _safe_train_fno1d_ensembles_vet(config, dataset, num_tries=5):
    for i in range(num_tries):
        try:
            best_rmse_i, model_i = _train_one_mf_fno1d_ensemble_vnet(config, dataset)
            return best_rmse_i, model_i
        except:
            cprint('y', 'failed, try again')
        #


def train_mf_fno1d_ensembles_vnet(config, dataset, logger, path_dicts=None):
    if path_dicts is None and config.ensembles.use_disk is True:
        raise Exception('No path found to save the ensembles')

    models_ensembles = []
    models_errs = []

    # for i in trange(config.ensembles.num_ensembles, desc='train ensembles'):
    for i in range(config.ensembles.num_ensembles):

        t_start = time.time()
        # best_rmse_i, model_i = _train_one_mf_fno1d_ensemble_vnet(config, dataset)
        best_rmse_i, model_i = _safe_train_fno1d_ensembles_vet(config, dataset)
        t_end = time.time()
        logger.info('    ({:3f} secs) ensemble-learner {}, best_rmse={},'.format(t_end - t_start, i + 1, best_rmse_i))

        if config.ensembles.use_disk:
            dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
            torch.save(model_i.state_dict(), os.path.join(path_dicts, dict_name))

        models_ensembles.append(model_i)
        models_errs.append(best_rmse_i)
    #

    models_errs = np.array(models_errs)

    return models_ensembles, models_errs


def load_model_ensembles_vnet(config, path_dicts=None):
    if path_dicts is None:
        raise Exception('no valid path found')

    models_list = []
    for i in range(config.ensembles.num_ensembles):
        modes = config.model.modes
        width = config.model.width
        model = FNO1d_Emsemble_vnet(modes, width, config.data.fids_list)
        dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
        cprint('r', 'loading {} from {}'.format(dict_name, path_dicts))
        model.load_state_dict(torch.load(os.path.join(path_dicts, dict_name)))
        model.to(config.device)
        models_list.append(model)
    #

    return models_list


def eval_mf_ensembles_preds_vnet(config, X_list, models_list):

    with torch.no_grad():

        assert len(models_list) == config.ensembles.num_ensembles

        pred_mu_samples_list = []
        pred_std_samples_list = []

        ensembles_var = []

        # for model_i in models_list:
        #     ensembles_var.append(np.log1p(np.exp(model_i.rho.item())) ** 2)
        # #

        for fid, X in zip(config.data.fids_list, X_list):
            # cprint('r', fid)
            # cprint('b', X.shape)

            sf_data_loader = torch.utils.data.DataLoader(
                X,
                batch_size=config.testing.batch_size,
                shuffle=False
            )

            sf_pred_mu_samples = []
            sf_pred_std_samples = []

            for i, model_i in enumerate(models_list):

                pred_mu_i = []
                pred_std_i = []

                for x in sf_data_loader:
                    x = x.to(config.device)
                    out, out_rho = model_i(x, fid)
                    out = out.reshape(x.shape[0], fid)
                    # cprint('b', out.shape)
                    # cprint('y', out_rho.shape)

                    # pred_rho = model_i.rho
                    pred_std = torch.log1p(torch.exp(out_rho)) * torch.ones_like(out).to(config.device)
                    # print(pred_std)
                    pred_mu_i.append(out)
                    pred_std_i.append(pred_std)
                #

                pred_mu_i = torch.cat(pred_mu_i)
                pred_std_i = torch.cat(pred_std_i)
                # cprint('g', pred_mu_i.shape)
                # cprint('r', pred_std_i.shape)
                # print(pred_std_i)

                sf_pred_mu_samples.append(pred_mu_i.unsqueeze(0))
                sf_pred_std_samples.append(pred_std_i.unsqueeze(0))
            #

            # cprint('g', torch.cat(sf_pred_mu_samples).shape)
            # cprint('b', torch.cat(sf_pred_std_samples).shape)

            sf_pred_mu_samples = torch.cat(sf_pred_mu_samples).permute(1,0,2)
            sf_pred_std_samples = torch.cat(sf_pred_std_samples).permute(1,0,2)

            # cprint('g', sf_pred_mu_samples.shape)
            # cprint('b', sf_pred_std_samples.shape)

            pred_mu_samples_list.append(sf_pred_mu_samples)
            pred_std_samples_list.append(sf_pred_std_samples)
        #

        return pred_mu_samples_list, pred_std_samples_list, ensembles_var


#========================================================================================#

def _train_one_mf_fno1d_dropout(config, dataset, init_state=None):
    modes = config.model.modes
    width = config.model.width
    # model = FNO1d_Emsemble_vnet(modes, width, config.data.fids_list).to(config.device)
    model = FNO1d_Dropout(modes, width, config.data.fids_list, config.training.dropout).to(config.device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)

    # if init_state is not None:
    #     model.load_state_dict(init_state)
    # else:
    #     cprint('y', 'WARNING: NO INIT STATE FOUND.. FNO2D with new INIT')

    batch_ratio = config.training.batch_ratio
    epochs = config.training.epochs
    learning_rate = config.optim.learning_rate
    weight_decay = config.optim.weight_decay
    iters_per_epoch = int(1.0 / batch_ratio)
    iterations = epochs * iters_per_epoch

    # cprint('r', epochs)
    # cprint('r', batch_ratio)
    # cprint('b', iters_per_epoch)
    # cprint('r', iterations)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    best_rmse = np.inf
    best_model = None

    Xtr_list, ytr_list = dataset.get_train_data()
    Xte_list, yte_list = dataset.get_test_data()
    Xte, yte = Xte_list[-1], yte_list[-1]

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xte, yte),
        batch_size=config.testing.batch_size,
        shuffle=False
    )

    dist_gamma = torch.distributions.Gamma(
        torch.tensor(1.).to(config.device),
        torch.tensor(1.).to(config.device)
    )

    myloss = LpLoss(size_average=False)

    for ep in range(epochs):

        for iter in range(iters_per_epoch):

            mfid_loss = []

            optimizer.zero_grad()

            for fid, Xfid, yfid in zip(config.data.fids_list, Xtr_list, ytr_list):
                # cprint('r', fid)
                # cprint('b', Xfid.shape)
                # cprint('b', yfid.shape)

                bz = round(Xfid.shape[0] * batch_ratio)
                batch_idx = np.random.choice(a=Xfid.shape[0], size=bz, replace=False)

                # print(bz)
                # print(batch_idx)

                x, y = Xfid[batch_idx, ...], yfid[batch_idx, ...]
                x, y = x.to(config.device), y.to(config.device)
                # cprint('r', x.shape)
                # cprint('b', y.shape)

                out, rho_out = model(x, fid)

                out = out.reshape(x.shape[0], fid)

                # cprint('r', out.shape)
                # cprint('b', rho_out.shape)

                # v = torch.log1p(torch.exp(model.rho)) ** 2
                v_out = torch.log1p(torch.exp(rho_out)) ** 2

                # logprob_prec = dist_gamma.log_prob(1. / v)
                logprob_prec = dist_gamma.log_prob(1. / v_out)

                nllh_term1 = (0.5 * torch.log(v_out)).sum()
                # cprint('y', torch.square(y - out).shape)
                # cprint('y', v_out.shape)
                nllh_term2 = 0.5 * torch.sum(torch.square(y - out) / v_out)

                loss_fid = nllh_term1 + nllh_term2 - logprob_prec.sum()

                mfid_loss.append(loss_fid)
            #

            # cprint('r', mfid_loss)

            loss = sum(mfid_loss)

            # cprint('b', loss)

            loss.backward()

            optimizer.step()
            scheduler.step()
        #
        with torch.no_grad():

            pred = []

            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                # out = model(x).reshape(x.shape[0], fid)
                out, rho_out = model(x, fid)
                out = out.reshape(x.shape[0], fid)
                pred.append(out.data.cpu().numpy())
            #

            pred = np.concatenate(pred)
            np_y_test = yte.data.cpu().numpy()

            # cprint('r', pred.shape)
            # cprint('b', np_y_test.shape)

            direct_rmse = np.sqrt(((np_y_test - pred) ** 2).sum()) / np.sqrt((np_y_test ** 2).sum())

            if direct_rmse < best_rmse:
                best_rmse = direct_rmse
                best_model = copy.deepcopy(model)

            # cprint('g', 'epoch={}, rmse={}, best_rmse={}'.format(ep, direct_rmse, best_rmse))
        #
    #

    return best_rmse, best_model

def _safe_train_fno1d_dropout(config, dataset, num_tries=5):
    for i in range(num_tries):
        try:
            best_rmse_i, model_i = _train_one_mf_fno1d_dropout(config, dataset)
            return best_rmse_i, model_i
        except:
            cprint('y', 'failed, try again')
        #


def train_mf_fno1d_dropout(config, dataset, logger, path_dicts=None):
    if path_dicts is None and config.ensembles.use_disk is True:
        raise Exception('No path found to save the ensembles')

    models_ensembles = []
    models_errs = []

    # for i in trange(config.ensembles.num_ensembles, desc='train ensembles'):
    for i in range(1):

        t_start = time.time()
        # best_rmse_i, model_i = _train_one_mf_fno1d_dropout(config, dataset)
        best_rmse_i, model_i = _safe_train_fno1d_dropout(config, dataset)
        t_end = time.time()
        logger.info('    ({:3f} secs) ensemble-learner {}, best_rmse={},'.format(t_end - t_start, i + 1, best_rmse_i))

        if config.ensembles.use_disk:
            dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
            torch.save(model_i.state_dict(), os.path.join(path_dicts, dict_name))

        models_ensembles.append(model_i)
        models_errs.append(best_rmse_i)
    #

    models_errs = np.array(models_errs)

    return models_ensembles, models_errs


def load_model_dropout(config, path_dicts=None):
    if path_dicts is None:
        raise Exception('no valid path found')

    # models_list = []
    # for i in range(1):
    #     modes = config.model.modes
    #     width = config.model.width
    #     # model = FNO1d_Emsemble_vnet(modes, width, config.data.fids_list)
    #     model = FNO1d_Dropout(modes, width, config.data.fids_list)
    #     dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
    #     cprint('r', 'loading {} from {}'.format(dict_name, path_dicts))
    #     model.load_state_dict(torch.load(os.path.join(path_dicts, dict_name)))
    #     model.to(config.device)
    #     models_list.append(model)
    # #

    modes = config.model.modes
    width = config.model.width
    # model = FNO1d_Emsemble_vnet(modes, width, config.data.fids_list)
    model = FNO1d_Dropout(modes, width, config.data.fids_list, config.training.dropout)
    dict_name = 'ensemble_learner_{}.pt'.format(1)
    cprint('r', 'loading {} from {}'.format(dict_name, path_dicts))
    model.load_state_dict(torch.load(os.path.join(path_dicts, dict_name)))
    model.to(config.device)

    return model


def eval_mf_preds_dropout(config, X_list, model):

    with torch.no_grad():

        pred_mu_samples_list = []
        pred_std_samples_list = []

        ensembles_var = []

        for fid, X in zip(config.data.fids_list, X_list):
            # cprint('r', fid)
            # cprint('b', X.shape)

            sf_data_loader = torch.utils.data.DataLoader(
                X,
                batch_size=config.testing.batch_size,
                shuffle=False
            )

            sf_pred_mu_samples = []
            sf_pred_std_samples = []

            # for i, model_i in enumerate(models_list):
            for i in range(config.ensembles.num_ensembles):

                pred_mu_i = []
                pred_std_i = []

                for x in sf_data_loader:
                    x = x.to(config.device)
                    out, out_rho = model(x, fid)
                    out = out.reshape(x.shape[0], fid)
                    # cprint('b', out.shape)
                    # cprint('y', out_rho.shape)

                    # pred_rho = model_i.rho
                    pred_std = torch.log1p(torch.exp(out_rho)) * torch.ones_like(out).to(config.device)
                    # print(pred_std)
                    pred_mu_i.append(out)
                    pred_std_i.append(pred_std)
                #

                pred_mu_i = torch.cat(pred_mu_i)
                pred_std_i = torch.cat(pred_std_i)
                # cprint('g', pred_mu_i.shape)
                # cprint('r', pred_std_i.shape)
                # print(pred_std_i)

                sf_pred_mu_samples.append(pred_mu_i.unsqueeze(0))
                sf_pred_std_samples.append(pred_std_i.unsqueeze(0))
            #

            # cprint('g', torch.cat(sf_pred_mu_samples).shape)
            # cprint('b', torch.cat(sf_pred_std_samples).shape)

            sf_pred_mu_samples = torch.cat(sf_pred_mu_samples).permute(1,0,2)
            sf_pred_std_samples = torch.cat(sf_pred_std_samples).permute(1,0,2)

            # cprint('g', sf_pred_mu_samples.shape)
            # cprint('b', sf_pred_std_samples.shape)
            # print(sf_pred_mu_samples[0,:,:])

            pred_mu_samples_list.append(sf_pred_mu_samples)
            pred_std_samples_list.append(sf_pred_std_samples)
        #

        return pred_mu_samples_list, pred_std_samples_list, ensembles_var

#========================================================================================#

def _train_one_mf_fno1d_coresets(config, dataset, init_state=None):
    modes = config.model.modes
    width = config.model.width
    model = FNO1d_Coresets(modes, width, config.data.fids_list).to(config.device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)

    # if init_state is not None:
    #     model.load_state_dict(init_state)
    # else:
    #     cprint('y', 'WARNING: NO INIT STATE FOUND.. FNO2D with new INIT')

    batch_ratio = config.training.batch_ratio
    epochs = config.training.epochs
    learning_rate = config.optim.learning_rate
    weight_decay = config.optim.weight_decay
    iters_per_epoch = int(1.0 / batch_ratio)
    iterations = epochs * iters_per_epoch

    # cprint('r', epochs)
    # cprint('r', batch_ratio)
    # cprint('b', iters_per_epoch)
    # cprint('r', iterations)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    best_rmse = np.inf
    best_model = None

    Xtr_list, ytr_list = dataset.get_train_data()
    Xte_list, yte_list = dataset.get_test_data()
    Xte, yte = Xte_list[-1], yte_list[-1]

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xte, yte),
        batch_size=config.testing.batch_size,
        shuffle=False
    )

    dist_gamma = torch.distributions.Gamma(
        torch.tensor(1.).to(config.device),
        torch.tensor(1.).to(config.device)
    )

    myloss = LpLoss(size_average=False)

    for ep in range(epochs):

        for iter in range(iters_per_epoch):

            mfid_loss = []

            optimizer.zero_grad()

            for fid, Xfid, yfid in zip(config.data.fids_list, Xtr_list, ytr_list):
                # cprint('r', fid)
                # cprint('b', Xfid.shape)
                # cprint('b', yfid.shape)

                bz = round(Xfid.shape[0] * batch_ratio)
                batch_idx = np.random.choice(a=Xfid.shape[0], size=bz, replace=False)

                # print(bz)
                # print(batch_idx)

                x, y = Xfid[batch_idx, ...], yfid[batch_idx, ...]
                x, y = x.to(config.device), y.to(config.device)
                # cprint('r', x.shape)
                # cprint('b', y.shape)

                out, rho_out = model(x, fid)

                out = out.reshape(x.shape[0], fid)

                # cprint('r', out.shape)
                # cprint('b', rho_out.shape)

                # v = torch.log1p(torch.exp(model.rho)) ** 2
                v_out = torch.log1p(torch.exp(rho_out)) ** 2

                # logprob_prec = dist_gamma.log_prob(1. / v)
                logprob_prec = dist_gamma.log_prob(1. / v_out)

                nllh_term1 = (0.5 * torch.log(v_out)).sum()
                # cprint('y', torch.square(y - out).shape)
                # cprint('y', v_out.shape)
                nllh_term2 = 0.5 * torch.sum(torch.square(y - out) / v_out)

                loss_fid = nllh_term1 + nllh_term2 - logprob_prec.sum()

                mfid_loss.append(loss_fid)
            #

            # cprint('r', mfid_loss)

            loss = sum(mfid_loss)

            # cprint('b', loss)

            loss.backward()

            optimizer.step()
            scheduler.step()
        #
        with torch.no_grad():

            pred = []

            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                # out = model(x).reshape(x.shape[0], fid)
                out, rho_out = model(x, fid)
                out = out.reshape(x.shape[0], fid)
                pred.append(out.data.cpu().numpy())
            #

            pred = np.concatenate(pred)
            np_y_test = yte.data.cpu().numpy()

            # cprint('r', pred.shape)
            # cprint('b', np_y_test.shape)

            direct_rmse = np.sqrt(((np_y_test - pred) ** 2).sum()) / np.sqrt((np_y_test ** 2).sum())

            if direct_rmse < best_rmse:
                best_rmse = direct_rmse
                best_model = copy.deepcopy(model)

            # cprint('r', 'epoch={}, v={}, rmse={}, best_rmse={}'.format(ep, torch.log1p(torch.exp(model.rho)) ** 2, direct_rmse, best_rmse))
            # cprint('c', 'epoch={}, rmse={}, best_rmse={}'.format(ep, direct_rmse, best_rmse))
        #
    #

    return best_rmse, best_model

def _safe_train_fno1d_coresets(config, dataset, num_tries=5):
    for i in range(num_tries):
        try:
            best_rmse_i, model_i = _train_one_mf_fno1d_coresets(config, dataset)
            return best_rmse_i, model_i
        except:
            cprint('y', 'failed, try again')
        #


def train_mf_fno1d_coresets(config, dataset, logger, path_dicts=None):
    if path_dicts is None and config.ensembles.use_disk is True:
        raise Exception('No path found to save the ensembles')

    # models_ensembles = []
    # models_errs = []

    # # for i in trange(config.ensembles.num_ensembles, desc='train ensembles'):
    # for i in range(config.ensembles.num_ensembles):
    #
    #     t_start = time.time()
    #     # best_rmse_i, model_i = _train_one_mf_fno1d_ensemble_vnet(config, dataset)
    #     best_rmse_i, model_i = _safe_train_fno1d_coresets(config, dataset)
    #     t_end = time.time()
    #     logger.info('    ({:3f} secs) ensemble-learner {}, best_rmse={},'.format(t_end - t_start, i + 1, best_rmse_i))
    #
    #     if config.ensembles.use_disk:
    #         dict_name = 'ensemble_learner_{}.pt'.format(i + 1)
    #         torch.save(model_i.state_dict(), os.path.join(path_dicts, dict_name))
    #
    #     models_ensembles.append(model_i)
    #     models_errs.append(best_rmse_i)
    # #
    #
    # models_errs = np.array(models_errs)
    #
    # return models_ensembles, models_errs

    # for i in trange(config.ensembles.num_ensembles, desc='train ensembles'):


    t_start = time.time()
    # best_rmse_i, model_i = _train_one_mf_fno1d_ensemble_vnet(config, dataset)
    best_rmse, model = _safe_train_fno1d_coresets(config, dataset)
    t_end = time.time()
    logger.info('    ({:3f} secs) ensemble-learner {}, best_rmse={},'.format(t_end - t_start, 1, best_rmse))

    if config.ensembles.use_disk:
        dict_name = 'ensemble_learner_{}.pt'.format(1)
        torch.save(model.state_dict(), os.path.join(path_dicts, dict_name))


    return model, best_rmse

def load_model_coresets(config, path_dicts=None):
    if path_dicts is None:
        raise Exception('no valid path found')


    modes = config.model.modes
    width = config.model.width
    model = FNO1d_Coresets(modes, width, config.data.fids_list)
    dict_name = 'ensemble_learner_{}.pt'.format(1)
    cprint('r', 'loading {} from {}'.format(dict_name, path_dicts))
    model.load_state_dict(torch.load(os.path.join(path_dicts, dict_name)))
    model.to(config.device)

    return model


def eval_model_mf_embeds(config, X_list, model):

    with torch.no_grad():

        mf_embeds = []

        for fid, X in zip(config.data.fids_list, X_list):
            # cprint('r', fid)
            # cprint('b', X.shape)

            sf_data_loader = torch.utils.data.DataLoader(
                X,
                batch_size=config.testing.batch_size,
                shuffle=False
            )

            sf_embeds = []

            for x in sf_data_loader:
                x = x.to(config.device)
                out, _, z = model.forward_emb(x, fid)
                z = z.reshape(x.shape[0],-1)
                sf_embeds.append(z)
            #

            sf_embeds = torch.vstack(sf_embeds)
            mf_embeds.append(sf_embeds)
        #

        return mf_embeds

def eval_model_mf_embeds_hybrid(config, X_list, model):

    with torch.no_grad():

        mf_embeds = []

        for fid, X in zip(config.data.fids_list, X_list):
            # cprint('r', fid)
            # cprint('b', X.shape)

            sf_data_loader = torch.utils.data.DataLoader(
                X,
                batch_size=config.testing.batch_size,
                shuffle=False
            )

            sf_embeds = []

            for x in sf_data_loader:
                x = x.to(config.device)
                # print(x.shape)
                # print(config.data.target_fid)
                x = x.permute(0,2,1)
                xhf = F.interpolate(x, size=config.data.target_fid, mode='linear')
                xhf = xhf.permute(0,2,1)
                # cprint('g', xhf.shape)
                out, _, z = model.forward_emb(xhf, config.data.target_fid)
                # cprint('r', out.shape)
                # cprint('c', z.shape)

                z = z.reshape(x.shape[0],-1)
                sf_embeds.append(z)
            #

            sf_embeds = torch.vstack(sf_embeds)
            # cprint('c', sf_embeds.shape)
            mf_embeds.append(sf_embeds)
        #

        return mf_embeds

