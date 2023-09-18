
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import pandas as pd
import shutil

nifti_folder = '/media/mcgoug01/nvme/ThirdYear/MastersProject/data/original_ncct/coreg_ncct/'
save_folder = '/media/mcgoug01/nvme/ThirdYear/MastersProject/data/original_ncct/over_150mAs/'
cases = os.listdir(os.path.join(nifti_folder,'images'))
cases.sort()
results = []

for file in cases:
    case = file.split('.')[0]
    exposures = []
    if file.startswith('KiTS'):
        dmccase_fold = "/media/mcgoug01/nvme/Data/C4KC-KiTS/{}".format(case)
        dmccase_fold = os.path.join(dmccase_fold,os.listdir(dmccase_fold)[0])
        dmccase_fold = os.path.join(dmccase_fold,[fold for fold in os.listdir(dmccase_fold) if 'noncontrast' in fold.lower()][0])
        num_files = len(os.listdir(dmccase_fold))
        for i in range(10):
            dcm = pydicom.dcmread(os.path.join(dmccase_fold, os.listdir(dmccase_fold)[(num_files // 2) - 5 + i]))
            exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value) / 1000)
    else:
        dmccase_fold = "/media/mcgoug01/nvme/Data/AddenbrookesRCC/NCCT/Unenhanced//{}".format(case)
        dmccase_fold = os.path.join(dmccase_fold,os.listdir(dmccase_fold)[0])
        num_files = len(os.listdir(dmccase_fold))
        for i in range(10):
            try:
                dcm = pydicom.dcmread(os.path.join(dmccase_fold, os.listdir(dmccase_fold)[(num_files // 2) - 5 + i]))
                exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value) / 1000)
            except:continue
        if len(exposures)<3:continue
    exposure = np.mean(exposures)

    if exposure>150:
        shutil.copy(os.path.join(nifti_folder,'images',file),os.path.join(save_folder,'images',file))
        shutil.copy(os.path.join(nifti_folder, 'labels', file), os.path.join(save_folder, 'labels', file))

    results.append({'case':case,'exposure':exposure})

df = pd.DataFrame(results)
exposures = df.exposure.values
print("Number of images with exposure over 200",len(exposures[exposures>150]))
interval_5 = np.percentile(exposures,5)
interval_10 = np.percentile(exposures,10)
interval_95 = np.percentile(exposures,95)
interval_90 = np.percentile(exposures,90)
print("5th percentile: {}".format(interval_5),"\n10th percentile: {}".format(interval_10),"\n90th percentile: {}".format(interval_90),"\n95th percentile: {}".format(interval_95))
plt.hist(df.exposure)
plt.xlabel('Exposure (mAs)')
plt.ylabel('Frequency')
plt.show()