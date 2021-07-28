import numpy as np
import json
import subprocess
import os
from multiprocessing import Process
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_json(json_path):
    with open(json_path, 'r') as f:
        out = json.load(f)
    return out


def save_json(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    args = parser.parse_args()

    # Data Configuration
    json_data_path = '../config/base_data.json'
    json_data = load_json(json_data_path)

    # Network Configuration
    json_network_path = '../config/base_network.json'
    json_network = load_json(json_network_path)

    # Train Configuration
    json_train_path = '../config/base_train.json'
    json_train = load_json(json_train_path)

    # Meta Configuration
    json_meta_path = '../config/base_meta.json'
    json_meta = load_json(json_meta_path)

    # Meta Configuration
    json_tune_path = '../config/base_tune.json'
    json_tune = load_json(json_tune_path)

    # Global Option
    train_prop = 1.
    val_prop = 1.

    resume = False
    mixed_precision = True

    ddp = False
    project_folder = 'module-merge'

    # Setup Configuration for Each Experiments
    if args.exp == 0:
        server = 'nipa'
        save_dir = '/home/sung/checkpoint/merge'
        data_dir = '/home/sung/dataset'
        data_type_and_num = ('imagenet', 1000)

        exp_name = 'imp'
        start = 0
        ix = 0
        comb_list = []

        w_d = 1e-4
        lr = 0.1
        epoch = 100

        batch_size = 256
        mixed_precision = True
        num_per_gpu = 1
        gpus = ['0,1,2', '0,1,2', '0,1,2']
        ddp = True

        depth = 34

        for t in gpus:
            comb_list.append([t, ix])
            ix += 1

    elif args.exp == 1:
        server = 'nipa'
        save_dir = '/home/sung/checkpoint/merge'
        data_dir = '/home/sung/dataset'
        data_type_and_num = ('cifar10', 10)

        exp_name = 'imp'
        start = 0
        ix = 0
        comb_list = []

        w_d = 1e-4
        lr = 0.1
        epoch = 200

        batch_size = 512
        mixed_precision = True
        num_per_gpu = 1
        gpus = ['0']

        depth = 18

        for t in ['0']:
            comb_list.append([t, ix])
            ix += 1

    else:
        raise('Select Proper Experiment Number')

    comb_list = comb_list * num_per_gpu
    comb_list = [comb + [index] for index, comb in enumerate(comb_list)]

    arr = np.array_split(comb_list, len(gpus))
    arr_dict = {}
    for ix in range(len(gpus)):
        arr_dict[ix] = arr[ix]

    def tr_gpu(comb, ix):
        comb = comb[ix]
        for i, comb_ix in enumerate(comb):
            exp_num = start + int(comb_ix[-1])
            os.makedirs(os.path.join(save_dir, exp_name, str(exp_num)), exist_ok=True)

            gpu = gpus[ix]

            # Modify the data configuration
            json_data['data_dir'] = data_dir
            json_data['data_type'] = data_type_and_num[0]
            json_data['num_class'] = data_type_and_num[1]
            save_json(json_data, os.path.join(save_dir, exp_name, str(exp_num), 'data.json'))

            # Modify the network configuration
            json_network['network_type'] = 'resnet%d' %int(depth)
            save_json(json_network, os.path.join(save_dir, exp_name, str(exp_num), 'network.json'))

            # Modify the train configuration
            json_train['gpu'] = str(gpu)

            json_train['lr'] = lr
            json_train['weight_decay'] = w_d

            json_train['total_epoch'] = epoch
            json_train['batch_size'] = batch_size

            json_train["mixed_precision"] = mixed_precision

            json_train["resume"] = resume

            json_train["train_prop"] = train_prop
            json_train["val_prop"] = val_prop

            json_train["ddp"] = ddp

            save_json(json_train, os.path.join(save_dir, exp_name, str(exp_num), 'train.json'))

            # Modify the meta configuration
            json_meta['server'] = str(server)
            json_meta['save_dir'] = str(save_dir)
            json_meta['project_folder'] = project_folder
            save_json(json_meta, os.path.join(save_dir, exp_name, str(exp_num), 'meta.json'))

            # Modify the tune configuration
            save_json(json_tune, os.path.join(save_dir, exp_name, str(exp_num), 'tune.json'))

            # Run !
            script = 'python ../main.py --save_dir %s --exp_name %s --exp_num %d' %(save_dir, exp_name, exp_num)
            subprocess.call(script, shell=True)


    for ix in range(len(gpus)):
        exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

    for ix in range(len(gpus)):
        exec('thread%d.start()' % ix)