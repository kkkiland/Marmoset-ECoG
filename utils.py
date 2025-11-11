import os
import json
import torch
import numpy as np
import logging
from datetime import datetime
from torch.utils.data import Dataset


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)




def Setup(args):
    """
        Input:
            args: arguments object from argparse
        Returns:
            config: configuration dictionary
    """
    config = args.__dict__  # configuration dictionary
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, config['data_path'], initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    #logger.info("Stored configuration file in '{}' as a configuration.json".format(output_dir))

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def Initialization(config):
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    #logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    return device



class dataset_class(Dataset):

    def __init__(self, data, label):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, ind):

        x = self.feature[ind]
        x = x.astype(np.float32)

        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)



