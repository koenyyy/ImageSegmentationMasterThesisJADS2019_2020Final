import json
import os
import time

# Code that is used to create a bunch of evaluation jobs using the BIGR framework

list_of_exp = ["/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131606_BTD_iscaling_noOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131528_BTD_zscore_noOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131633_BTD_iscaling_withOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131707_BTD_zscore_withOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131757_BTD_zscore_noOtsu_withBC_Res4/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131856_BTD_iscaling_noOtsu_withBC_Res4/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131947_BTD_iscaling_withOtsu_withBC_Res4/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530132035_BTD_zscore_withOtsu_withBC_Res4/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131528_BTD_zscore_noOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131606_BTD_iscaling_noOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131633_BTD_iscaling_withOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131707_BTD_zscore_withOtsu_withBC_Res2/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131757_BTD_zscore_noOtsu_withBC_Res4/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131856_BTD_iscaling_noOtsu_withBC_Res4/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530131947_BTD_iscaling_withOtsu_withBC_Res4/eval_nifti/result",
"/media/data/kderaad/glassimaging/glassimaging-master/experiment_results/20200530132035_BTD_zscore_withOtsu_withBC_Res4/eval_nifti/result"]

base_cmd = " python -m glassimaging.execution.jobs.jobeval {0} {1} {2}"

for exp in list_of_exp:
    results_loc = exp.split("glassimaging-master/")[1].rstrip("/eval_nifti/result")
    run_eval_cmd = base_cmd.format('eval_nifti_2_{0}'.format(results_loc.strip('experiment_results/')),
                                   results_loc + '/' + 'eval_nifti_2_{0}'.format(results_loc.strip('experiment_results/')) + '/config_eval_unet.json' , results_loc)

    # load the slurm script and add the right line to run in it (python -m etc.)
    with open('C:/Users/s145576/Documents/GitHub/master_thesis/glassimaging-master/codeTesting/run_eval_job2.sh', 'r') as infile:
        eval_job_slurm = infile.readlines()
        eval_job_slurm[-1] = run_eval_cmd

    with open('C:/Users/s145576/Documents/GitHub/master_thesis/glassimaging-master/codeTesting/run_eval_job2.sh', 'w') as outfile:
        outfile.writelines(eval_job_slurm)

    # open the config json file in order to adjust it based on the experiments neede settings
    with open('C:/Users/s145576/Documents/GitHub/master_thesis/glassimaging-master/config/eval_unet.json', 'r') as eval_unet_infile:
        eval_unet_json = json.load(eval_unet_infile)

    # below we set the general variables (normalization and otsu) based on the experiment names
    if 'noNorm' in exp:
        eval_unet_json['use_normalization'] = False
    else:
        eval_unet_json['use_normalization'] = True

    if 'zscore' in exp:
        eval_unet_json['technique'] = 'z-score'
    elif 'iscaling' in exp:
        eval_unet_json['technique'] = 'i-scaling'

    if 'withOtsu' in exp:
        eval_unet_json['using_otsu_ROI'] = True
    elif 'noOtsu' in exp or 'nohOtsu' in exp:
        eval_unet_json['using_otsu_ROI'] = False

    # below we set the dependent variables (resampling factor, dataset and nifti source)
    if 'BTD' in exp:
        # set dataset first
        eval_unet_json['Dataset'] = 'BTD'

        # set resampling factor to either 1, 2 or 4
        if 'Res1' in exp:
            eval_unet_json['resampling_factor'] = 1
        elif 'Res2' in exp:
            eval_unet_json['resampling_factor'] = 2
        elif 'Res4' in exp:
            eval_unet_json['resampling_factor'] = 4

        # set nifti source depending on whether BC is used or not
        if 'noBC' in exp:
            eval_unet_json['Nifti Source'] = '/media/data/kvangarderen/BTD'
        elif 'withBC' in exp:
            eval_unet_json['Nifti Source'] = '/media/data/kderaad/BTD_N4BC'

    if 'LiTS' in exp:
        # set dataset first
        eval_unet_json['Dataset'] = 'LitsData'

        # set nifti source depending on whether BC is used or not
        if 'noBC' in exp:
            # set resampling factor to either 1, 2 or 4
            if 'Res1' in exp:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes1'
            elif 'Res2' in exp:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes2'
            elif 'Res4' in exp:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes4'

        elif 'withBC' in exp:
            # set resampling factor to either 1, 2 or 4
            if 'Res1' in exp:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes1_N4BC'
            elif 'Res2' in exp:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes2_N4BC'
            elif 'Res4' in exp:
                eval_unet_json['resampling_factor'] = 1
                eval_unet_json['Nifti Source'] = '/media/data/kderaad/LiTSRes4_N4BC'

    print(run_eval_cmd)
    print(eval_unet_json)
    print('\n')
    print(results_loc + '/' + 'eval_nifti_2_{0}'.format(results_loc.strip('experiment_results/')) + '/config_eval_unet.json')
    # write new json file as config
    # with open(results_loc + '/' + 'eval_nifti_2_{0}'.format(results_loc.strip('experiment_results/')) + '/config_eval_unet.json', 'w') as eval_unet_outfile:
    #     json.dump(eval_unet_json, eval_unet_outfile, indent=4)
    #
    # # call the command that makes an eval job
    # os.system('sbatch run_eval_job.sh')

    time.sleep(5)
