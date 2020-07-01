import SimpleITK as sitk

# Code that was used for the validation of a connection with the BIGR cluster

def test_bigr_cluster_connection():
    img = sitk.ReadImage("/media/data/kderaad/bigr_mount/LiTS/volume-0.nii/volume-0.nii")
    print("succesfully obtained image with size:", img.GetSize())
    print("END")

if __name__ == '__main__':
    test_bigr_cluster_connection()
