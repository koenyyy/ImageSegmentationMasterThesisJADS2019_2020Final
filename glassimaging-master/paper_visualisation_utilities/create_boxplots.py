import pandas as pd
from os import walk
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Code that generates boxplots for visualisation purposes

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

    df['config_name'] = technique + otsu_cn + n4bc_cn
    return df

def create_single_big_boxplot(list_of_files, save=False, save_path='C:\\Users'):
    full_df = pd.DataFrame()
    for index, results_file in enumerate(res_file_list):
        res_df = create_results_df(results_file, index)
        full_df = pd.concat([full_df, res_df], sort=True)

    full_df = full_df.sort_values(by=['config_name'])

    plt.figure()
    # use string as class for normal results_eval
    # df[df["class"]=='1'].boxplot(column=["dice"])
    plt.ylim(0, 1)
    # use int for upsampled_eval_results
    # full_df[full_df["class"] == 1].boxplot(by='config_number', column=["dice"], figsize=(20,10))
    g = sns.catplot(kind='box', x='dice', y='config_name', data=full_df[full_df["class"] == '1'],
                    row="res_factor", palette="Set3", orient="h", height=3, aspect=6, legend=True)
    g.set_axis_labels("Dice Score", "Configuration")
    g.set_titles(row_template="Resampling factor = {row_name}")

    if save:
        if 'Ergo' in save_path:
            filename = '\\Ergo_Results_Boxplots.png'
        elif 'BTD' in save_path:
            filename = '\\BTD_Results_Boxplots.png'
        elif 'LiTS' in save_path:
            filename = '\\LiTS_Results_Boxplots.png'
        plt.savefig(save_path + filename, format='png')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    mypath = "C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Results\\24ExperimentsLiTS"
    # mypath = "C:\\Users\\s145576\\Documents\\.Koen de Raad\\year19.20\\Thesis\\Erasmus MC\\Results\\72ExperimentsBTD\\BTD seed 0"
    res_file_list = get_result_file_list(mypath)
    # create_individual_boxplots(res_file_list)
    create_single_big_boxplot(res_file_list, save=False, save_path=mypath)

