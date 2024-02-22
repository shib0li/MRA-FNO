import os
from infras.misc import cprint


def validate_model_dicts(dicts_dir, num_ensembles=5):
    valid = True
    # print(dicts_dir)
    for i in range(num_ensembles):
        dict_fname = 'ensemble_learner_{}.pt'.format(i+1)
        if os.path.exists(os.path.join(dicts_dir, dict_fname)):
            continue
        else:
            cprint('y', '(model saved at {} is missing)'.format(os.path.join(dicts_dir, dict_fname)))
            valid = False
        #
    #
    return valid

def validate_exp_evals(evals_dir, exp_name):
    valid = True

    hist_costs_fname = 'hist_costs_{}.npy'.format(exp_name)
    if os.path.exists(os.path.join(evals_dir, hist_costs_fname)) is False:
        valid = False
        cprint('y', '(hist costs file {} is missing)'.format(os.path.join(evals_dir, hist_costs_fname)))

    hist_fids_fname = 'hist_fids_{}.npy'.format(exp_name)
    if os.path.exists(os.path.join(evals_dir, hist_fids_fname)) is False:
        valid = False
        cprint('y', '(hist fids file {} is missing)'.format(os.path.join(evals_dir, hist_fids_fname)))


    rmse_fname = 'test_rmse_{}.npy'.format(exp_name)
    if os.path.exists(os.path.join(evals_dir, rmse_fname)) is False:
        valid = False
        cprint('y', '(rmse file {} is missing)'.format(os.path.join(evals_dir, rmse_fname)))

    data_idx_fname = 'data_idx_{}.pickle'.format(exp_name)
    if os.path.exists(os.path.join(evals_dir, data_idx_fname)) is False:
        valid = False
        cprint('y', '(data idx file {} is missing)'.format(os.path.join(evals_dir, data_idx_fname)))

    return valid



def recover_exp_snapshots(meta_dir, exp_name):

    cprint('g', 'Looking for Experiment snapshots')

    exp_log_workdir = os.path.join(meta_dir, 'logs')
    exp_evals_workdir = os.path.join(meta_dir, 'evals', exp_name)
    exp_dicts_workdir = os.path.join(meta_dir, 'dicts', exp_name)

    if os.path.exists(exp_log_workdir) is False:
        cprint('y', 'log dir {} is missing. Failed to recover.'.format(exp_log_workdir))
        return None

    if os.path.exists(os.path.join(exp_log_workdir, 'log-{}.txt'.format(exp_name))) is False:
        cprint('y', 'log file {} is missing. Failed to recover.'.format(exp_log_workdir))
        return None

    if os.path.exists(exp_evals_workdir) is False:
        cprint('y', 'evals dir {} is missing. Failed to recover.'.format(exp_evals_workdir))
        return None

    if os.path.exists(exp_dicts_workdir) is False:
        cprint('y', 'dicts dir {} is missing. Failed to recover.'.format(exp_dicts_workdir))
        return None

    # print(exp_log_workdir)
    # print(exp_evals_workdir)
    # print(exp_dicts_workdir)
    #
    # print(os.path.exists(exp_log_workdir))
    # print(os.path.exists(exp_evals_workdir))
    # print(os.path.exists(exp_dicts_workdir))

    current_step = 0

    while True:

        valid_dicts = validate_model_dicts(os.path.join(exp_dicts_workdir, 'step{}'.format(current_step+1)))
        if valid_dicts:
            cprint('g', '  - successfully validate model dicts at step {}'.format(current_step+1))
        else:
            cprint('r', '  - failed validate model dicts at step {}'.format(current_step + 1))
            break

        valid_evals = validate_exp_evals(
            os.path.join(exp_evals_workdir, 'step{}'.format(current_step + 1)),
            exp_name
        )

        if valid_evals:
            cprint('g', '  - successfully validate evaluations at step {}'.format(current_step+1))
        else:
            cprint('r', '  - failed validate evaluations at step {}'.format(current_step + 1))
            break

        current_step += 1

    # cprint('g', 'Validated saved step is {}'.format(current_step))

    if current_step > 0:
        return current_step
    else:
        return None



