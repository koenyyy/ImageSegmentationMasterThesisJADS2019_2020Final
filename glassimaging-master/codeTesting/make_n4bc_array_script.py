import os

# Code that generates a script which can be run on the BIGR cluster to create a list of files that need to pass through bias correction

# This is made for working with the LiTS dataset
def make_n4bc_array_script(data_loc):
    lstFilesGz = []  # create an empty list with images
    lstFilesGzSeg = []  # create an empty list with segmentations

    for dirName, subdirList, fileList in os.walk(data_loc):
        for filename in fileList:
            if ".nii" in filename.lower():  # check whether the file's .nii
                # check whether a file has a segmentation from a specific person
                if "volume" in filename.lower():
                    lstFilesGz.append(os.path.join(dirName, filename))
                if "segmentation" in filename.lower():
                    lstFilesGzSeg.append(os.path.join(dirName, filename))
    print(lstFilesGz)

    with open("n4bc_array.sh", "w") as file:
        for image in lstFilesGz:
            input_path = image
            output_path = image.replace("intermediate_data", "Lits_N4BC")

            # write n4 bc command
            file.write("n4 -i {0:s} -o {1:s}".format(input_path, output_path))
            file.write('\n')

    with open("create_subtree.sh", "w") as file:
        for image in lstFilesGz:
            output_path = image.replace("intermediate_data", "Lits_N4BC")
            output_subfolder_tree = os.path.join(*output_path.split(os.sep)[:-1])

            # create mkdir file structure
            file.write("mkdir -p {0:s}".format(output_subfolder_tree))
            file.write('\n')

if __name__ == '__main__':
    bigr_app_data_loc= "/scratch/kderaad/LiTS"
    make_n4bc_array_script("C:\\Users\\s145576\\Documents\\GitHub\\automaticSegmentationThesis\\data\\intermediate_data")
