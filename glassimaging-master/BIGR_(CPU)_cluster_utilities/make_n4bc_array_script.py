import os

# This is made for working with the LiTS dataset and BTD and Ergo
def make_n4bc_array_script(data_loc):
    lstFilesGz = []  # create an empty list with images
    lstFilesGzSeg = []  # create an empty list with segmentations

    for dirName, subdirList, fileList in os.walk(data_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                # specify how the segmentations and the masks look
                if "t1_t2_morphology" in filename.lower():
                    lstFilesGz.append(os.path.join(dirName, filename))
                if "combined_bone_mask" in filename.lower():
                    lstFilesGzSeg.append(os.path.join(dirName, filename))
    print(lstFilesGz)

    with open("n4bc_array.sh", "w") as file:
        for image in lstFilesGz:
            input_path = image
            output_path = image.replace("Ergo_Res4", "Ergo_Res4_N4BC")

            # write n4 bc command
            file.write("n4 3 -i {0:s} -o {1:s}".format(input_path, output_path))
            file.write('\n')

    with open("create_subtree.sh", "w") as file:
        for image in lstFilesGz:
            output_path = image.replace("Ergo_Res4", "Ergo_Res4_N4BC")
            output_subfolder_tree = os.path.join(*output_path.split(os.sep)[-3:-1])

            # create mkdir file structure
            file.write("mkdir -p {0:s}".format(output_subfolder_tree))
            file.write('\n')

    with open("copy_segmentations.sh", "w") as file:
        for image in lstFilesGzSeg:
            output_path = image.replace("Ergo_Res4", "Ergo_Res4_N4BC")
            output_subfolder_tree = os.path.join(*output_path.split(os.sep)[-3:-1])

            # create mkdir file structure
            file.write("cp {0:s} {1:s}".format(image, output_path))
            file.write('\n')
if __name__ == '__main__':
    bigr_app_data_loc= "/scratch/kderaad/Ergo_Res4"
    make_n4bc_array_script(bigr_app_data_loc)
