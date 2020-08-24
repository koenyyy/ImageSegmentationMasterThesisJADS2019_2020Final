import itertools
import json
import os
import time

#in this file we automated the initiation of experiments. All preprocessing steps should be specified in the create_configurations function.
# based on these steps experiments are created in the gpu cluster every 20 seconds. No need to define each experiment by hand.

def create_configurations():
    normalization_methods = ['zscore', 'iscaling']
    otsu = [True, False]
    resampling_factors = [1, 2, 4]
	# be aware that n4bc application is not done on the fly, a special preprocessed dataset should be used. 
    n4bc = [True, False]
    random_seed = [364]

    all_combinations = list(itertools.product(normalization_methods, otsu, resampling_factors, n4bc, random_seed))
    print("combinations:", len(all_combinations), all_combinations)

    return all_combinations

def create_config_json_files(exp, dataset, train_config_loc, eval_config_loc):
    normalization_methods = exp[0]
    use_otsu = exp[1]
    resampling_factor = exp[2]
    use_n4bc = exp[3]
    random_seed = exp[4]

    # open the config json files in order to adjust it based on the experiments needed settings
    with open(train_config_loc, 'r') as train_unet_infile:
        train_unet_json = json.load(train_unet_infile)

    with open(eval_config_loc, 'r') as eval_unet_infile:
        eval_unet_json = json.load(eval_unet_infile)

    eval_unet_json['seed'] = random_seed
    train_unet_json['seed'] = random_seed

    # below we set the general variables (normalization and otsu) based on the experiment names
    if normalization_methods == 'zscore':
        eval_unet_json['technique'] = 'z-score'
        train_unet_json['technique'] = 'z-score'
    elif normalization_methods == 'iscaling':
        eval_unet_json['technique'] = 'i-scaling'
        train_unet_json['technique'] = 'i-scaling'

    if use_otsu:
        eval_unet_json['using_otsu_ROI'] = True
        train_unet_json['using_otsu_ROI'] = True
    elif not use_otsu:
        eval_unet_json['using_otsu_ROI'] = False
        train_unet_json['using_otsu_ROI'] = False

    # below we set the dependent variables (resampling factor, dataset and nifti source)
    if dataset == 'BTD':
        # set dataset first
        eval_unet_json['Dataset'] = 'BTD'
        train_unet_json['Dataset'] = 'BTD'

        # set resampling factor to either 1, 2 or 4
        if resampling_factor == 1:
            eval_unet_json['resampling_factor'] = 1
            train_unet_json['resampling_factor'] = 1
        elif resampling_factor == 2:
            eval_unet_json['resampling_factor'] = 2
            train_unet_json['resampling_factor'] = 2
        elif resampling_factor == 4:
            eval_unet_json['resampling_factor'] = 4
            train_unet_json['resampling_factor'] = 4

        # set nifti source depending on whether BC is used or not
        if not use_n4bc:
            eval_unet_json['Nifti Source'] = '/media/data/kvangarderen/BTD'
            train_unet_json['Nifti Source'] = '/media/data/kvangarderen/BTD'
        elif use_n4bc:
            eval_unet_json['Nifti Source'] = '/media/data/kderaad/BTD_N4BC'
            train_unet_json['Nifti Source'] = '/media/data/kderaad/BTD_N4BC'

    if dataset == 'LiTS':
        # set dataset first
        eval_unet_json['Dataset'] = 'LitsData'
        train_unet_json['Dataset'] = 'LitsData'

        # set nifti source depending on whether BC is used or not
        if not use_n4bc:
            # set resampling factor to either 1, 2 or 4
            if resampling_factor == 1:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes1'
                train_unet_json['resampling_factor'] = 1
                train_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes1'
            elif resampling_factor == 2:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes2'
                train_unet_json['resampling_factor'] = 1
                train_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes2'
            elif resampling_factor == 4:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes4'
                train_unet_json['resampling_factor'] = 1
                train_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes4'

        elif use_n4bc:
            # set resampling factor to either 1, 2 or 4
            if resampling_factor == 1:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes1_N4BC'
                train_unet_json['resampling_factor'] = 1
                train_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes1_N4BC'
            elif resampling_factor == 2:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes2_N4BC'
                train_unet_json['resampling_factor'] = 1
                train_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes2_N4BC'
            elif resampling_factor == 4:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes4_N4BC'
                train_unet_json['resampling_factor'] = 1
                train_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes4_N4BC'

    print(train_unet_json)
    print(eval_unet_json)
    print('\n')

    # write new json file as config
    with open(train_config_loc, 'w') as train_unet_outfile:
        json.dump(train_unet_json, train_unet_outfile, indent=4)

    with open(eval_config_loc, 'w') as eval_unet_outfile:
        json.dump(eval_unet_json, eval_unet_outfile, indent=4)

    return train_unet_json, eval_unet_json

        # print(results_loc + '/' + 'eval_nifti_2_{0}'.format(results_loc.strip('experiment_results/')) + '/config_eval_unet.json')
        # write new json file as config
        # with open(results_loc + '/' + 'eval_nifti_2_{0}'.format(results_loc.strip('experiment_results/')) + '/config_eval_unet.json', 'w') as eval_unet_outfile:
        #     json.dump(eval_unet_json, eval_unet_outfile, indent=4)

def create_run_command(train_config):
    base_cmd = 'python -m glassimaging.execution.experiment {0}_{1}_{2}Otsu_{3}BC_Res{4}_Seed{5} ' \
               'config/unet_end_to_end.json experiment_results'
    dataset = train_config['Dataset']
    norm_method = train_config['technique'].replace('-', '')
    use_otsu = 'with' if train_config['using_otsu_ROI'] else 'no'
    use_BC = 'with' if 'N4BC' in train_config['Nifti Source'] else 'no'
	# Here we specify what to do for what dataset
    if dataset == 'BTD':
        res_factor = train_config['resampling_factor']
    elif dataset == 'LitsData' or dataset == 'ErgoData':
        res_factor = int(train_config['Nifti Source'].split(os.sep)[-1].split('_')[0][-1])
    seed = train_config['seed']

    run_cmd = base_cmd.format(dataset, norm_method, use_otsu, use_BC, res_factor, seed)
    return run_cmd

def execute_run_command(command):
    os.system(command)

if __name__ == '__main__':
    combination_list = create_configurations()
    train_config_loc = '/media/data/kderaad/glassimaging/glassimaging-master/config/train_unet.json'
    eval_config_loc = '/media/data/kderaad/glassimaging/glassimaging-master/config/eval_unet.json'
    for exp in combination_list:
        train_unet_json, eval_unet_json = create_config_json_files(exp, 'BTD', train_config_loc, eval_config_loc)
        command = create_run_command(train_unet_json)
        execute_run_command(command)
        time.sleep(20)
