import ml_collections
import torch

def get_default_configs():

    config = ml_collections.ConfigDict()

    # data
    config.data = data = ml_collections.ConfigDict()
    data.normalize = True
    data.ntrain = 1000
    data.ntest = 200

    # model
    config.model = model = ml_collections.ConfigDict()
    model.modes = 16
    model.width = 64

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 20
    training.epochs = 500

    # testing
    config.testing = testing = ml_collections.ConfigDict()
    testing.batch_size = 20
    testing.samples = 20

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.learning_rate = 1e-3
    optim.weight_decay = 1e-4
    optim.scheduler = 'CosAnnealingLR'

    # logging
    config.logging = logging =  ml_collections.ConfigDict()
    logging.display = False

    # misc
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config