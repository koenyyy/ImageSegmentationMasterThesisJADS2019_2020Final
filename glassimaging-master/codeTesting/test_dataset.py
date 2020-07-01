from glassimaging.dataloading.lipodata import LipoData, LipoDataset
import numpy as np
import pandas as pd

# Code that allows for a dataset to be loaded and check whether a new dataloader functions properly

# Windows path
# path = r'C:\Users\s145576\Documents\GitHub\master_thesis\glassimaging-master\LipoData'
# x = [r'C:\Users\s145576\Documents\GitHub\master_thesis\glassimaging-master\LipoData']

# Linux path
path = '/home/koen/AAA_Koen/AllLipoData/'
x = ['/home/koen/AAA_Koen/Master_Thesis/Code/Lipo_Data/']
y = ['string']

df = pd.DataFrame()

ld = LipoData(df)

ld.importData(path)

print(ld.df)
print(ld.patients)
