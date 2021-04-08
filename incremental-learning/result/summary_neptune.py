import numpy as np
import pandas as pd
import os
import json

def load_json(json_path):
    with open(json_path, 'r') as f:
        out = json.load(f)
    return out

def save_json(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f)

def get_config(config_path, option):
    json_config = load_json(config_path)
    param_list = []
    column_list = []

    for op in option:
        out = json_config[op[0]][op[1]]
        param_list.append(out)
        column_list.append(op[1])

    return param_list, column_list

def summary(result_csv, config_dir, config_name, output_column, option_column, save_name, exp_name_criterion=None):
    df = pd.read_csv(result_csv)
    df_target = df.loc[:, output_column]
    df_exp = df.loc[:, ['exp_name', 'exp_num']]

    target_list = []
    target_column = []
    for i in range(len(df_target)):
        target_output = list(df_target.iloc[i, :])
        exp_name = str(df_exp.loc[i, 'exp_name'])
        exp_num = str(df_exp.loc[i, 'exp_num'])

        if (exp_name_criterion is not None) and (exp_name != exp_name_criterion):
            continue

        config_path = os.path.join(config_dir, exp_name, str(exp_num), config_name)
        target_param, target_column = get_config(config_path, option_column)

        target_list.append(target_output + target_param + [exp_name, exp_num])

    column_list = output_column + target_column + ['exp_name', 'exp_num']

    if len(target_list) == 0:
        print('No Data!!')
    else:
        df = pd.DataFrame(np.array(target_list), columns=column_list)
        df.to_csv(save_name, index=False)


if __name__=='__main__':
    # Option
    result_file = './result_file/exp1_parameter_CIFAR100_icarl.csv'
    out_column = ['val_acc1_total', 'val_acc5_total']
    option = [('optim', 'lr'), ('optim', 'weight_decay'), ('optim', 'momentum')]
    config_dir = '/home/sung/checkpoint/icarl'
    config_name = 'task_1_config.json'

    # Save the result
    summary(result_csv=result_file, config_dir=config_dir, config_name=config_name, output_column=out_column, \
            option_column=option, save_name='./out/exp1_parameter_result3.csv', exp_name_criterion='parameter_control')







