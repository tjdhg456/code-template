import numpy as np
import json
import subprocess
import os
from multiprocessing import Process
import argparse


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
    parser.add_argument('--log', type=lambda x: x.lower()=='true', default=True)
    args = parser.parse_args()

    # Data Configuration
    json_data_path = '../config/base_data.json'
    json_data = load_json(json_data_path)

    # Network Configuration
    json_network_path = '../config/base_network.json'
    json_network = load_json(json_network_path)

    # Meta Configuration
    json_meta_path = '../config/base_meta.json'
    json_meta = load_json(json_meta_path)

    # Exemplar Configuration
    json_exemplar_path = '../config/base_exemplar.json'
    json_exemplar = load_json(json_exemplar_path)

    # Setup Configuration for Each Experiments
    if args.exp == 0:
        server = 'nipa'
        save_dir = '/home/sung/checkpoint/icarl'
        data_dir = '/home/sung/dataset'
        data_type_and_num = ('cifar100', 100)

        exp_name = 'parameter_control'
        start = 0
        comb_list = []

        num_per_gpu = 1
        network_type = 'resnet18'
        gpus = ['0']
        train_list = ['icarl']
        num_exemple_list = [2000]
        optimizer_type = 'sgd'

        lr_list = [0.1]
        weight_decay_list = [0.0001]
        momentum_list = [0.9]

        # Initialize
        only_init = False

        # Resume Option
        resume = False
        resume_task_id = 0
        init_path = None

        ix = 0
        for tr in train_list:
            for num_ex in num_exemple_list:
                for lr in lr_list:
                    for weight in weight_decay_list:
                        for momentum in momentum_list:
                            comb_list.append([tr, num_ex, lr, weight, momentum, ix])
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
            json_data['data_dir'] = str(data_dir)
            json_data['data_type'] = data_type_and_num[0]
            json_data['num_class'] = data_type_and_num[1]
            save_json(json_data, os.path.join(save_dir, exp_name, str(exp_num), 'data.json'))


            # Modify the network configuration
            json_network['network_type'] = network_type
            save_json(json_network, os.path.join(save_dir, exp_name, str(exp_num), 'network.json'))


            # Modify the train configuration
            train_type = str(comb_ix[0])
            json_train_path = '../config/base_train_%s.json' %train_type
            json_train = load_json(json_train_path)

            json_train['only_init'] = only_init
            json_train['resume'] = resume
            json_train['resume_task_id'] = resume_task_id
            json_train['gpu'] = str(gpu)
            json_train['num_exemplary'] = int(comb_ix[1])
            save_json(json_train, os.path.join(save_dir, exp_name, str(exp_num), 'train.json'))


            # Modify the optimizer configuration
            json_optim_path = '../config/base_optim_%s.json' %optimizer_type
            json_optim = load_json(json_optim_path)

            json_optim['lr'] = float(comb_ix[2])
            json_optim['weight_decay'] = float(comb_ix[3])
            json_optim['momentum'] = float(comb_ix[4])

            save_json(json_optim, os.path.join(save_dir, exp_name, str(exp_num), 'optim.json'))


            # Modify the meta configuration
            json_meta['server'] = str(server)
            json_meta['save_dir'] = str(save_dir)
            save_json(json_meta, os.path.join(save_dir, exp_name, str(exp_num), 'meta.json'))


            # Modify the exemplar Configuration
            json_exemplar
            save_json(json_exemplar, os.path.join(save_dir, exp_name, str(exp_num), 'exemplar.json'))


            # Run !
            if init_path is not None:
                script0 = 'cp -r %s %s' % (os.path.join('/'.join(init_path.split('/')[:-1]), 'result.txt'), os.path.join(save_dir, exp_name, str(exp_num)))
                script1 = 'python ../generate_init_set.py --save_dir %s --exp_name %s --exp_num %s --init_path %s' %(save_dir, exp_name, exp_num, init_path)
                script2 = 'python ../main.py --save_dir %s --exp_name %s --exp_num %d --log %s' % (save_dir, exp_name, exp_num, str(args.log))
                subprocess.call(script0, shell=True)
                subprocess.call(script1, shell=True)
                subprocess.call(script2, shell=True)

            else:
                script = 'python ../main.py --save_dir %s --exp_name %s --exp_num %d --log %s' %(save_dir, exp_name, exp_num, str(args.log))
                subprocess.call(script, shell=True)


    for ix in range(len(gpus)):
        exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

    for ix in range(len(gpus)):
        exec('thread%d.start()' % ix)