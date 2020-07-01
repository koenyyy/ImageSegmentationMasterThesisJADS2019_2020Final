import numpy as np
import pandas as pd
from os import walk
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import wilcoxon
import itertools
from statsmodels.stats.multitest import multipletests

# Code that executes the Wilcoxon signed rank tests. Results are also visualized.


def get_result_file_list(mypath):
    files = []
    for (dirpath, dirnames, fileList) in walk(mypath):
        for filename in fileList:
            if not "old" in dirpath:
                # for lits we need to use the newly resampled materials
                if 'LiTS' in dirpath or ('BTD' in dirpath and 'BTD seed 0' in dirpath):
                    if "results_eval.csv" in filename and '_Res1' in dirpath and 'new_eval' in dirpath:
                        files.append(os.path.join(dirpath, filename))
                    elif "upsampled_eval_results.csv" in filename and 'new_eval' in dirpath:
                        files.append(os.path.join(dirpath, filename))
                else:
                    if "results_eval.csv" in filename and '_Res1' in dirpath:
                        files.append(os.path.join(dirpath, filename))
                    elif "upsampled_eval_results.csv" in filename:
                        files.append(os.path.join(dirpath, filename))

    return files


def create_single_boxplot_from_results_csv(path):
    df = pd.read_csv(path, sep=",")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    save_path = path.split(".csv")[0] + '_boxplot.png'

    plt.figure()
    # use string as class for normal results_eval
    # df[df["class"]=='1'].boxplot(column=["dice"])
    plt.ylim(0, 1)
    # use int for upsampled_eval_results
    df[df["class"] == 1].boxplot(column=["dice"])

    plt.savefig(save_path, format='png')
    plt.close()

def create_individual_boxplots(list_of_files):
    for results_file in res_file_list:
        create_single_boxplot_from_results_csv(results_file)

def create_results_df(results_file, config_number):
    df = pd.read_csv(results_file, sep=",")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df["class"] = df["class"].astype(str)

    # first we initialize the variables we need
    dataset = ''
    use_normalization = True
    technique = ''
    use_otsu = False
    use_N4BC = False
    res_factor = 1
    # below we set all the variables based on the file name
    if 'noNorm' in results_file:
        use_normalization = False
    else:
        use_normalization = True

    if 'zscore' in results_file:
        technique = ' z-score'
    elif 'iscaling' in results_file:
        technique = ' i-scaling'
    elif 'noNorm' in results_file:
        technique = 'without normalization'

    if 'withOtsu' in results_file:
        use_otsu = True
    elif 'noOtsu' in results_file or 'nohOtsu' in results_file:
        use_otsu = False

    if 'withBC' in results_file:
        use_N4BC = True
    elif 'noBC' in results_file:
        use_N4BC = False

    # set resampling factor to either 1, 2 or 4
    if 'Res1' in results_file:
        res_factor = 1
    elif 'Res2' in results_file:
        res_factor = 2
    elif 'Res4' in results_file:
        res_factor = 4

    if 'BTD' in results_file:
        dataset = 'BTD'
    elif 'LiTS' in results_file:
        dataset = 'LiTS'
    elif 'Ergo' in results_file:
        dataset = 'Ergo'

    # here we place the values that indicate preprocessing steps into
    df['dataset'] = dataset
    df['use_normalization'] = use_normalization
    df['technique'] = technique
    df['use_otsu'] = use_otsu
    df['use_N4BC'] = use_N4BC
    df['res_factor'] = res_factor
    df['config_number'] = config_number

    if use_otsu:
        otsu_cn = ' with Otsu'
    else:
        otsu_cn = ' no Otsu'

    if use_N4BC:
        n4bc_cn = ' with N4BC'
    else:
        n4bc_cn = ' no N4BC'

    if res_factor == 1:
        res_factor_cn = ' Res 1'
    elif res_factor == 2:
        res_factor_cn = ' Res 2'
    elif res_factor == 4:
        res_factor_cn = ' Res 4'

    df['config_name'] = technique + otsu_cn + n4bc_cn + res_factor_cn
    return df

def get_best_config(list_of_df):
    best_config_name = ''
    best_median = 0

    for setup in list_of_df:
        if setup[setup["class"] == '1']['dice'].median() > best_median:
            best_median = setup[setup["class"] == '1']['dice'].median()
            best_config_name = setup['config_name'][0]
    return best_config_name

def create_single_big_boxplot(list_of_files, save=False, save_path='C:\\Users\\s145576\\Desktop', compare_best=False):
    df_list = []
    for index, results_file in enumerate(list_of_files):
        res_df = create_results_df(results_file, index)
        df_list.append(res_df)

    results_df = pd.DataFrame(columns=['setting 0', 'setting 1', 'p-value'])
    # either use permutations or combinations
    permutation_list = [i for i in itertools.permutations(df_list, r=2)]
    for permutation in permutation_list:
        if not compare_best:
            w, p = wilcoxon(permutation[0][permutation[0]["class"] == '1']['dice'].to_numpy(), permutation[1][permutation[1]["class"] == '1']['dice'].to_numpy())
            results_df = results_df.append({'setting 0': permutation[0]['config_name'][0], 'setting 1': permutation[1]['config_name'][0], 'p-value':p}, ignore_index = True)
        else:
            best_config_name = get_best_config(df_list)
            if permutation[0]['config_name'][0] == best_config_name or permutation[1]['config_name'][0] == best_config_name:
                w, p = wilcoxon(permutation[0][permutation[0]["class"] == '1']['dice'].to_numpy(), permutation[1][permutation[1]["class"] == '1']['dice'].to_numpy())
                results_df = results_df.append(
                    {'setting 0': permutation[0]['config_name'][0], 'setting 1': permutation[1]['config_name'][0],
                     'p-value':p}, ignore_index = True)
            else:
                results_df = results_df.append(
                    {'setting 0': permutation[0]['config_name'][0], 'setting 1': permutation[1]['config_name'][0],
                     'p-value': 1.0}, ignore_index=True)

    # results_df.sort_values(by=['setting 0', 'setting 1'])
    # corr_p_values = multipletests(results_df['p-value'], alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
    # results_df['corr_p_values'] = corr_p_values[1]

    plt.figure(figsize=(30, 30))
    print(results_df[results_df.duplicated()])
    results_df = results_df.pivot("setting 0", "setting 1", "p-value")
    mask = np.zeros_like(results_df)
    mask[results_df > 0.05] = True
    mask[np.triu_indices_from(mask)] = True
    # g = sns.heatmap(results_df, mask=mask, cbar=False, annot=True, linewidths=0.5, fmt=".3f",cmap=ListedColormap(['#89cf55', '#fffde8']), center = 0.05)
    g = sns.heatmap(results_df, cbar=False, annot=True, linewidths=0.5, fmt=".3f", cmap=ListedColormap(['#89cf55', '#fffde8']), center=0.05)

    if save:
        if 'Ergo' in save_path:
            filename = '\\Ergo_Results_Wilcoxon_test.png'
            df_filename = '\\Ergo_Results_Wilcoxon_test.csv'
        elif 'BTD' in save_path and 'seed 0' in save_path:
            filename = '\\BTD_Results_Wilcoxon_test_seed_0.png'
            df_filename = '\\BTD_Results_Wilcoxon_test_seed_0.csv'
        elif 'BTD' in save_path and 'seed 13' in save_path:
            filename = '\\BTD_Results_Wilcoxon_test_seed_13.png'
            df_filename = '\\BTD_Results_Wilcoxon_test_seed_13.csv'
        elif 'BTD' in save_path and 'seed 364' in save_path:
            filename = '\\BTD_Results_Wilcoxon_test_seed_364.png'
            df_filename = '\\BTD_Results_Wilcoxon_test_seed_364.csv'
        elif 'LiTS' in save_path:
            filename = '\\LiTS_Results_Wilcoxon_test.png'
            df_filename = '\\LiTS_Results_Wilcoxon_test.csv'
        results_df.to_csv(save_path + df_filename, index=False)
        plt.savefig(save_path + filename, format='png')
        plt.close()
    else:
        plt.show()


def create_ranking_based_on_median(list_of_files, save=False, save_path='C:\\Users\\s145576\\Desktop'):
    df_list = []
    for index, results_file in enumerate(list_of_files):
        res_df = create_results_df(results_file, index)
        df_list.append(res_df)

    results_df = pd.DataFrame(columns=['config_name', 'median'])

    best_config_name = ''
    best_median = 0

    for setup in df_list:
        results_df = results_df.append({'config_name': setup['config_name'][0], 'median':setup[setup["class"] == '1']['dice'].median()}, ignore_index = True)

    results_df.sort_values(by='median', inplace=True, ascending=False)


    if save:
        if 'Ergo' in save_path:
            df_filename = '\\Ergo_Results_ranking.csv'
        elif 'BTD' in save_path and 'seed 0' in save_path:
            df_filename = '\\BTD_Results_ranking_seed_0.csv'
        elif 'BTD' in save_path and 'seed 13' in save_path:
            df_filename = '\\BTD_Results_ranking_seed_13.csv'
        elif 'BTD' in save_path and 'seed 364' in save_path:
            df_filename = '\\BTD_Results_ranking_seed_364.csv'
        elif 'LiTS' in save_path:
            df_filename = '\\LiTS_Results_ranking.csv'
        results_df.to_csv(save_path + df_filename, index=False)
    else:
        print(results_df)


if __name__ == "__main__":
    mypath = "C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Results\\24ExperimentsLiTS"
    # mypath = "C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Results\\72ExperimentsBTD\BTD seed 364"
    res_file_list = get_result_file_list(mypath)
    # create_individual_boxplots(res_file_list)
    # create_single_big_boxplot(res_file_list, save=True, save_path=mypath, compare_best=True)
    create_ranking_based_on_median(res_file_list, save=True, save_path=mypath)

