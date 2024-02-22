import os
from infras.fno_utilities import *
from infras.misc import cprint

#===============================================================================================================#

# class SFData2D:
#
#     def __init__(self, config):
#
#         self.target_fid = config.data.target_fid
#
#         ntrain = config.data.ntrain
#         ntest = config.data.ntest
#
#         mat_path = os.path.join(
#             config.data.path,
#             'darcy_s{}_n{}.mat'.format(config.data.target_fid, config.data.nall)
#         )
#
#         mat_loader = MatReader(mat_path)
#
#         x_data = mat_loader.read_field('input')
#         y_data = mat_loader.read_field('output')
#
#         x_train = x_data[:ntrain, ...]
#         y_train = y_data[:ntrain, ...]
#         x_test = x_data[-ntest:, ...]
#         y_test = y_data[-ntest:, ...]
#
#         self.x_normalizer = UnitGaussianNormalizer(x_data[:ntrain, ...])
#         self.y_normalizer = UnitGaussianNormalizer(y_data[:ntrain, ...])
#
#         self.x_train = self.x_normalizer.encode(x_train)
#         self.y_train = self.y_normalizer.encode(y_train)
#
#         self.x_test = self.x_normalizer.encode(x_test)
#         self.y_test = self.y_normalizer.encode(y_test)
#
#         # cprint('r', self.x_train.shape)
#         # cprint('r', self.y_train.shape)
#         #
#         # cprint('b', self.x_test.shape)
#         # cprint('b', self.y_test.shape)
#
#         self.train_idx = np.arange(config.active.init_size)
#         self.pool_idx = np.arange(config.active.init_size, config.active.pool_size+config.active.init_size)
#
#         # cprint('r', self.train_idx)
#         # cprint('b', self.pool_idx)
#
#
#     def get_train_data(self):
#
#         x = self.x_train[self.train_idx, ...]
#         y = self.y_train[self.train_idx, ...]
#
#         # cprint('r', x.shape)
#         # cprint('r', y.shape)
#
#         x = x.reshape(len(self.train_idx), self.target_fid, self.target_fid, 1)
#
#         return x, y
#
#     def get_pool_data(self):
#
#         x = self.x_train[self.pool_idx, ...]
#         y = self.y_train[self.pool_idx, ...]
#
#         x = x.reshape(len(self.pool_idx), self.target_fid, self.target_fid, 1)
#
#         return x, y
#
#     def get_test_data(self):
#
#         x = self.x_test
#         y = self.y_test
#
#         x = x.reshape(self.x_test.shape[0], self.target_fid, self.target_fid, 1)
#
#         return x, y
#
#     def update(self, queries_idx, logger=None):
#
#         nq = queries_idx.size
#
#         updated_train_idx = np.concatenate([self.train_idx, self.pool_idx[queries_idx]])
#         self.train_idx = updated_train_idx
#         updated_pool_idx = np.delete(self.pool_idx, queries_idx)
#         self.pool_idx = updated_pool_idx
#
#
#         if logger is not None:
#             logger.info(' - size sanity checking ...')
#             logger.info('    current train examples: {}'.format(self.x_train[self.train_idx].shape[0]))
#             logger.info('    current pool  examples: {}'.format(self.x_train[self.pool_idx].shape[0]))
#             logger.info('    current test  examples: {}'.format(self.x_test.shape[0]))
#         #

#===============================================================================================================#

class MFData2D:

    def __init__(self, config):

        self.target_fid = config.data.target_fid

        ntrain = config.data.ntrain
        ntest = config.data.ntest


        self.fids_list = config.data.fids_list
        self.fids_cost = config.data.fids_cost

        self.x_train_list = []
        self.y_train_list = []

        self.x_test_list = []
        self.y_test_list = []

        self.x_normalizer_list = []
        self.y_normalizer_list = []

        assert self.target_fid == self.fids_list[-1]

        for fid in self.fids_list:

            mat_path = os.path.join(
                config.data.path,
                'darcy_s{}_n{}.mat'.format(fid, config.data.nall)
            )

            mat_loader = MatReader(mat_path)

            x_data = mat_loader.read_field('input')
            y_data = mat_loader.read_field('output')

            # cprint('r', x_data.shape)
            # cprint('b', y_data.shape)

            x_train = x_data[:ntrain, ...]
            y_train = y_data[:ntrain, ...]
            # cprint('r', x_train.shape)
            # cprint('b', y_train.shape)

            x_test = x_data[-ntest:, ...]
            y_test = y_data[-ntest:, ...]
            # cprint('r', x_test.shape)
            # cprint('b', y_test.shape)

            x_normalizer = UnitGaussianNormalizer(x_train)
            y_normalizer = UnitGaussianNormalizer(y_train)
            self.x_normalizer_list.append(x_normalizer)
            self.y_normalizer_list.append(y_normalizer)

            x_train = x_normalizer.encode(x_train)
            y_train = y_normalizer.encode(y_train)
            self.x_train_list.append(x_train)
            self.y_train_list.append(y_train)

            x_test = x_normalizer.encode(x_test)
            y_test = y_normalizer.encode(y_test)
            self.x_test_list.append(x_test)
            self.y_test_list.append(y_test)

        #

        nfids = len(self.fids_list)
        self.fids_train_idx = []

        for i in range(nfids):
            self.fids_train_idx.append(np.arange(config.active.init_size))

        self.pool_idx = np.arange(config.active.init_size, config.active.pool_size+config.active.init_size)


        # cprint('r', self.fids_train_idx)
        # cprint('g', self.pool_idx)


    def get_train_data(self):

        X_list = []
        y_list = []

        for fid, train_idx, x_train, y_train in zip(self.fids_list, self.fids_train_idx, self.x_train_list, self.y_train_list):

            x = x_train[train_idx, ...]
            y = y_train[train_idx, ...]

            x = x.reshape(x.shape[0], fid, fid, 1)

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list

    def get_pool_data(self):

        X_list = []
        y_list = []

        for fid, x_train, y_train in zip(self.fids_list, self.x_train_list, self.y_train_list):

            x = x_train[self.pool_idx, ...]
            y = y_train[self.pool_idx, ...]

            x = x.reshape(x.shape[0], fid, fid, 1)

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list


    # def get_test_data(self):
    #
    #     x = self.x_test_list[-1]
    #     y = self.y_test_list[-1]
    #     fid = self.fids_list[-1]
    #
    #     x = x.reshape(x.shape[0], fid, fid, 1)
    #     # cprint('r', x.shape)
    #     # cprint('b', y.shape)
    #
    #     return x, y

    def get_test_data(self):

        X_list = []
        y_list = []

        for fid, x, y in zip(self.fids_list, self.x_test_list, self.y_test_list):

            x = x.reshape(x.shape[0], fid, fid, 1)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list


    def update(self, queries_input, queries_fid, logger=None):

        nq = queries_input.size
        assert queries_input.size == queries_fid.size

        for i in range(nq):
            queries = np.array([queries_input[i]])
            fid = queries_fid[i]
            self.fids_train_idx[fid] = np.concatenate([self.fids_train_idx[fid], self.pool_idx[queries]])
        #

        self.pool_idx = np.delete(self.pool_idx, queries_input)

        if logger is not None:
            logger.info(' - size checking ...')
            logger.info('    current train examples: ')

            for fidx, train_idx in enumerate(self.fids_train_idx):
                # print(fidx, train_idx)
                logger.info('      * fid={}, ntr={}'.format(self.fids_list[fidx], len(train_idx)))

            logger.info('    current pool examples: {}'.format(self.x_train_list[0][self.pool_idx].shape[0]))
            logger.info('    current test examples: {}'.format(self.x_test_list[0].shape[0]))
        #


class MFData2Dv2:

    def __init__(self, config):

        self.target_fid = config.data.target_fid

        ntrain = config.data.ntrain
        ntest = config.data.ntest


        self.fids_list = config.data.fids_list
        self.fids_cost = config.data.fids_cost

        self.x_train_list = []
        self.y_train_list = []

        self.x_test_list = []
        self.y_test_list = []

        self.x_normalizer_list = []
        self.y_normalizer_list = []

        assert self.target_fid == self.fids_list[-1]

        for fid in self.fids_list:

            # mat_path = os.path.join(
            #     config.data.path,
            #     'darcy_s{}_n{}.mat'.format(fid, config.data.nall)
            # )
            #
            # mat_loader = MatReader(mat_path)
            #
            # x_data = mat_loader.read_field('input')
            # y_data = mat_loader.read_field('output')
            #
            # cprint('r', x_data.shape)
            # cprint('b', y_data.shape)

            path_xdata = os.path.join(
                config.data.path,
                'diff-react-xdata-s{}-n{}.npy'.format(fid, config.data.nall)
            )

            path_ydata = os.path.join(
                config.data.path,
                'diff-react-ydata-s{}-n{}.npy'.format(fid, config.data.nall)
            )

            x_data = torch.tensor(np.load(path_xdata)).float()
            y_data = torch.tensor(np.load(path_ydata)).float()

            x_train = x_data[:ntrain, ...]
            y_train = y_data[:ntrain, ...]
            # cprint('r', x_train.shape)
            # cprint('b', y_train.shape)

            x_test = x_data[-ntest:, ...]
            y_test = y_data[-ntest:, ...]
            # cprint('r', x_test.shape)
            # cprint('b', y_test.shape)

            x_normalizer = UnitGaussianNormalizer(x_train)
            y_normalizer = UnitGaussianNormalizer(y_train)
            self.x_normalizer_list.append(x_normalizer)
            self.y_normalizer_list.append(y_normalizer)

            x_train = x_normalizer.encode(x_train)
            y_train = y_normalizer.encode(y_train)
            self.x_train_list.append(x_train)
            self.y_train_list.append(y_train)

            x_test = x_normalizer.encode(x_test)
            y_test = y_normalizer.encode(y_test)
            self.x_test_list.append(x_test)
            self.y_test_list.append(y_test)

        #

        nfids = len(self.fids_list)
        self.fids_train_idx = []

        for i in range(nfids):
            self.fids_train_idx.append(np.arange(config.active.init_size))

        self.pool_idx = np.arange(config.active.init_size, config.active.pool_size+config.active.init_size)


        # cprint('r', self.fids_train_idx)
        # cprint('g', self.pool_idx)


    def get_train_data(self):

        X_list = []
        y_list = []

        for fid, train_idx, x_train, y_train in zip(self.fids_list, self.fids_train_idx, self.x_train_list, self.y_train_list):

            x = x_train[train_idx, ...]
            y = y_train[train_idx, ...]

            x = x.reshape(x.shape[0], fid, fid, 1)

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list

    def get_pool_data(self):

        X_list = []
        y_list = []

        for fid, x_train, y_train in zip(self.fids_list, self.x_train_list, self.y_train_list):

            x = x_train[self.pool_idx, ...]
            y = y_train[self.pool_idx, ...]

            x = x.reshape(x.shape[0], fid, fid, 1)

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list


    # def get_test_data(self):
    #
    #     x = self.x_test_list[-1]
    #     y = self.y_test_list[-1]
    #     fid = self.fids_list[-1]
    #
    #     x = x.reshape(x.shape[0], fid, fid, 1)
    #     # cprint('r', x.shape)
    #     # cprint('b', y.shape)
    #
    #     return x, y

    def get_test_data(self):

        X_list = []
        y_list = []

        for fid, x, y in zip(self.fids_list, self.x_test_list, self.y_test_list):

            x = x.reshape(x.shape[0], fid, fid, 1)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list


    def update(self, queries_input, queries_fid, logger=None):

        nq = queries_input.size
        assert queries_input.size == queries_fid.size

        for i in range(nq):
            queries = np.array([queries_input[i]])
            fid = queries_fid[i]
            self.fids_train_idx[fid] = np.concatenate([self.fids_train_idx[fid], self.pool_idx[queries]])
        #

        self.pool_idx = np.delete(self.pool_idx, queries_input)

        if logger is not None:
            logger.info(' - size checking ...')
            logger.info('    current train examples: ')

            for fidx, train_idx in enumerate(self.fids_train_idx):
                # print(fidx, train_idx)
                logger.info('      * fid={}, ntr={}'.format(self.fids_list[fidx], len(train_idx)))

            logger.info('    current pool examples: {}'.format(self.x_train_list[0][self.pool_idx].shape[0]))
            logger.info('    current test examples: {}'.format(self.x_test_list[0].shape[0]))
        #

class MFData3D:

    def __init__(self, config):

        self.target_fid = config.data.target_fid

        ntrain = config.data.ntrain
        ntest = config.data.ntest


        self.fids_list = config.data.fids_list
        self.fids_cost = config.data.fids_cost

        self.x_train_list = []
        self.y_train_list = []

        self.x_test_list = []
        self.y_test_list = []

        self.x_normalizer_list = []
        self.y_normalizer_list = []

        assert self.target_fid == self.fids_list[-1]

        self.T_in = config.data.T_in
        self.T_out = config.data.T_out

        for fid in self.fids_list:

            mat_path = os.path.join(
                config.data.path,
                'ns_data_n{}_s{}_vis{}_T{}_steps{}.mat'.format(
                    config.data.nall,
                    fid,
                    config.data.viscosity,
                    config.data.T,
                    config.data.steps,
                )
            )

            cprint('r', mat_path)

            mat_loader = MatReader(mat_path)

            Uall = mat_loader.read_field('u')
            # cprint('g', Uall.shape)

            x_data = Uall[...,:config.data.T_in]
            y_data = Uall[...,config.data.T_in:config.data.T_in+config.data.T_out]
            # cprint('r', x_data.shape)
            # cprint('b', y_data.shape)

            x_train = x_data[:ntrain, ...]
            y_train = y_data[:ntrain, ...]
            # cprint('r', x_train.shape)
            # cprint('b', y_train.shape)

            x_test = x_data[-ntest:, ...]
            y_test = y_data[-ntest:, ...]
            # cprint('r', x_test.shape)
            # cprint('b', y_test.shape)

            x_normalizer = UnitGaussianNormalizer(x_train)
            y_normalizer = UnitGaussianNormalizer(y_train)
            self.x_normalizer_list.append(x_normalizer)
            self.y_normalizer_list.append(y_normalizer)

            x_train = x_normalizer.encode(x_train)
            y_train = y_normalizer.encode(y_train)
            self.x_train_list.append(x_train)
            self.y_train_list.append(y_train)

            x_test = x_normalizer.encode(x_test)
            y_test = y_normalizer.encode(y_test)
            self.x_test_list.append(x_test)
            self.y_test_list.append(y_test)

        #

        nfids = len(self.fids_list)
        self.fids_train_idx = []

        for i in range(nfids):
            self.fids_train_idx.append(np.arange(config.active.init_size))

        self.pool_idx = np.arange(config.active.init_size, config.active.pool_size+config.active.init_size)


        # cprint('r', self.fids_train_idx)
        # cprint('g', self.pool_idx)


    def get_train_data(self):

        X_list = []
        y_list = []

        for fid, train_idx, x_train, y_train in zip(self.fids_list, self.fids_train_idx, self.x_train_list, self.y_train_list):

            x = x_train[train_idx, ...]
            y = y_train[train_idx, ...]

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            x = x.reshape(x.shape[0], fid, fid, 1, self.T_in).repeat([1,1,1,self.T_out,1])

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list

    def get_pool_data(self):

        X_list = []
        y_list = []

        for fid, x_train, y_train in zip(self.fids_list, self.x_train_list, self.y_train_list):

            x = x_train[self.pool_idx, ...]
            y = y_train[self.pool_idx, ...]

            x = x.reshape(x.shape[0], fid, fid, 1, self.T_in).repeat([1,1,1,self.T_out,1])

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list



    def get_test_data(self):

        X_list = []
        y_list = []

        for fid, x, y in zip(self.fids_list, self.x_test_list, self.y_test_list):

            # x = x.reshape(x.shape[0], fid, fid, 1)

            x = x.reshape(x.shape[0], fid, fid, 1, self.T_in).repeat([1,1,1,self.T_out,1])

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list


    def update(self, queries_input, queries_fid, logger=None):

        nq = queries_input.size
        assert queries_input.size == queries_fid.size

        for i in range(nq):
            queries = np.array([queries_input[i]])
            fid = queries_fid[i]
            self.fids_train_idx[fid] = np.concatenate([self.fids_train_idx[fid], self.pool_idx[queries]])
        #

        self.pool_idx = np.delete(self.pool_idx, queries_input)

        if logger is not None:
            logger.info(' - size checking ...')
            logger.info('    current train examples: ')

            for fidx, train_idx in enumerate(self.fids_train_idx):
                # print(fidx, train_idx)
                logger.info('      * fid={}, ntr={}'.format(self.fids_list[fidx], len(train_idx)))

            logger.info('    current pool examples: {}'.format(self.x_train_list[0][self.pool_idx].shape[0]))
            logger.info('    current test examples: {}'.format(self.x_test_list[0].shape[0]))
        #

class MFData1D:

    def __init__(self, config):

        self.target_fid = config.data.target_fid

        ntrain = config.data.ntrain
        ntest = config.data.ntest


        self.fids_list = config.data.fids_list
        self.fids_cost = config.data.fids_cost

        self.x_train_list = []
        self.y_train_list = []

        self.x_test_list = []
        self.y_test_list = []

        self.x_normalizer_list = []
        self.y_normalizer_list = []

        assert self.target_fid == self.fids_list[-1]

        for fid in self.fids_list:

            mat_path = os.path.join(
                config.data.path,
                'Burgers{}.mat'.format(fid)
            )
            print(mat_path)

            mat_loader = MatReader(mat_path)

            x_data = mat_loader.read_field('input')
            y_data = mat_loader.read_field('output')

            x_train = x_data[:ntrain, ...]
            y_train = y_data[:ntrain, ...]
            # cprint('r', x_train.shape)
            # cprint('b', y_train.shape)

            x_test = x_data[-ntest:, ...]
            y_test = y_data[-ntest:, ...]
            # cprint('r', x_test.shape)
            # cprint('b', y_test.shape)

            x_normalizer = UnitGaussianNormalizer(x_train)
            y_normalizer = UnitGaussianNormalizer(y_train)
            self.x_normalizer_list.append(x_normalizer)
            self.y_normalizer_list.append(y_normalizer)

            x_train = x_normalizer.encode(x_train)
            y_train = y_normalizer.encode(y_train)
            self.x_train_list.append(x_train)
            self.y_train_list.append(y_train)

            x_test = x_normalizer.encode(x_test)
            y_test = y_normalizer.encode(y_test)
            self.x_test_list.append(x_test)
            self.y_test_list.append(y_test)

        #

        nfids = len(self.fids_list)
        self.fids_train_idx = []

        for i in range(nfids):
            self.fids_train_idx.append(np.arange(config.active.init_size))

        self.pool_idx = np.arange(config.active.init_size, config.active.pool_size+config.active.init_size)


        # cprint('r', self.fids_train_idx)
        # cprint('g', self.pool_idx)


    def get_train_data(self):

        X_list = []
        y_list = []

        for fid, train_idx, x_train, y_train in zip(self.fids_list, self.fids_train_idx, self.x_train_list, self.y_train_list):

            x = x_train[train_idx, ...]
            y = y_train[train_idx, ...]

            x = x.reshape(x.shape[0], fid, 1)

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list

    def get_pool_data(self):

        X_list = []
        y_list = []

        for fid, x_train, y_train in zip(self.fids_list, self.x_train_list, self.y_train_list):

            x = x_train[self.pool_idx, ...]
            y = y_train[self.pool_idx, ...]

            x = x.reshape(x.shape[0], fid, 1)

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list


    def get_test_data(self):

        X_list = []
        y_list = []

        for fid, x, y in zip(self.fids_list, self.x_test_list, self.y_test_list):

            x = x.reshape(x.shape[0], fid, 1)

            # cprint('r', x.shape)
            # cprint('b', y.shape)

            X_list.append(x)
            y_list.append(y)
        #

        return X_list, y_list


    def update(self, queries_input, queries_fid, logger=None):

        nq = queries_input.size
        assert queries_input.size == queries_fid.size

        for i in range(nq):
            queries = np.array([queries_input[i]])
            fid = queries_fid[i]
            self.fids_train_idx[fid] = np.concatenate([self.fids_train_idx[fid], self.pool_idx[queries]])
        #

        self.pool_idx = np.delete(self.pool_idx, queries_input)

        if logger is not None:
            logger.info(' - size checking ...')
            logger.info('    current train examples: ')

            for fidx, train_idx in enumerate(self.fids_train_idx):
                # print(fidx, train_idx)
                logger.info('      * fid={}, ntr={}'.format(self.fids_list[fidx], len(train_idx)))

            logger.info('    current pool examples: {}'.format(self.x_train_list[0][self.pool_idx].shape[0]))
            logger.info('    current test examples: {}'.format(self.x_test_list[0].shape[0]))
        #


