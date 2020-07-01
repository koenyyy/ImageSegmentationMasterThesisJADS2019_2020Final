import os

# Short script that shows what files have not yet been processed (file was needed after script quit mid-run)

def print_unprocessed_dirs():
    all_dirs = []
    dirs_present = []
    path_all = '/media/data/kvangarderen/BTD'
    path_done = '/media/data/kderaad/BTD_N4BC_2'
    for dirName, subdirList, fileList in os.walk(path_all):
        all_dirs.append(dirName.split(os.sep)[-1])

    for dirName, subdirList, fileList in os.walk(path_done):
        dirs_present.append(dirName.split(os.sep)[-1])

    print('##1', all_dirs)
    print('##2', dirs_present)
    diff_list = list(set(all_dirs).symmetric_difference(set(dirs_present)))
    print('##3', diff_list)
    print('DONE')

if __name__ == '__main__':
    print_unprocessed_dirs()

