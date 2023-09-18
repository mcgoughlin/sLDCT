
import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import nibabel as nib
from ts_algorithms import fbp
import pydicom
import shutil
import os

"""
Reference for noise insertion technique:
Yu, Lifeng PhD; Shiung, Maria BA; Jondal, Dayna BA; McCollough, Cynthia H. PhD. 
Development and Validation of a Practical Lower-Dose-Simulation Tool for Optimizing Computed Tomography Scan Protocols. 
Journal of Computer Assisted Tomography: July/August 2012 - Volume 36 - Issue 4 - p 477-487 
doi: 10.1097/RCT.0b013e318258e891

"""

def from_normal_to_HU(image):
    return np.maximum((image*3000)-1000,-1000)

def from_HU_to_normal(image):
    return np.maximum((image+1000)/3000,0)

def do_synthetic_maths(image,A,dose_fraction,electronic_noise_sigma=0.1,
                       is_kits=True):
    normal_image = from_HU_to_normal(image)
    expP = A(normal_image)  # A() produces non-log transformed sinogram, hence expP
    if is_kits:
        Noa = int(exposure * 820 / 250)  # this is the number of incident photons at 250mAs dose - estimated with Kits-00000
    else:
        Noa = int(exposure * 1340 / 224)  # this is the number of incident photons at 250mAs dose - estimated with RCC_049

    mu, sigma = len(expP) / 2, 5  # mean and standard deviation
    s = np.random.normal(mu, sigma, 10000)
    count, bins, ignored = plt.hist(s, num_detectors - 1, density=True)
    plt.close()
    bowtie = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
    bowtie /= bowtie.max() * 1.2
    bowtie += +0.16666666667

    # equation 11 in paper referenced at top of script
    normal = np.random.normal(size=expP.shape)
    term1 = ((1 - dose_fraction) / dose_fraction) * (expP / Noa)
    term2 = ((1 + dose_fraction) / dose_fraction) * (electronic_noise_sigma ** 2) * expP / Noa
    noise = np.sqrt(term1 * (1 + term2) + 1e-9) * normal
    filtered_noise = (bowtie * noise)
    sino_noisy = filtered_noise + expP

    return from_normal_to_HU(fbp(A, torch.Tensor(sino_noisy)).detach().cpu().numpy())

def convert_to_LDCT(CT_vol,A,dose_fraction=0.5,electronic_noise_sigma=0.1,
                    is_kits=True):
    LDCT_vol = np.zeros_like(CT_vol)
    if is_kits:
        for slice in range(CT_vol.shape[1]):
            image = CT_vol[:,slice]
            sLDCT_slice = do_synthetic_maths(image, A, dose_fraction,
                                             electronic_noise_sigma=electronic_noise_sigma)
            LDCT_vol[:,slice] = sLDCT_slice
    else:
        for slice in range(CT_vol.shape[3]):
            image = CT_vol[:, :,:, slice]
            sLDCT_slice = do_synthetic_maths(image,A,dose_fraction,
                                             electronic_noise_sigma=electronic_noise_sigma)
            LDCT_vol[:, :, :, slice] = sLDCT_slice
    return LDCT_vol[0]


nifti_folder = '/media/mcgoug01/nvme/ThirdYear/MastersProject/data/original_ncct/over_150mAs/'
save_dir = '/media/mcgoug01/nvme/ThirdYear/MastersProject/data/sLDCT/'
cases = os.listdir(os.path.join(nifti_folder,'images'))
cases.sort()

num_detectors = 1216  # num detectors taken from https://s3.amazonaws.com/sgcimages/36_37_40_41_ITN1115_Siemens.pdf
electronic_noise_sigma = 0.1  # literally just an arbitrary guess
angles = 360  # arbitrary

vg = ts.volume(shape=(1, 512, 512), size=(5, 300, 300))
pg = ts.parallel(angles=angles, shape=(1, num_detectors), size=(1, num_detectors))
A = ts.operator(vg, pg)

for target_exposure in [150,125,100,75,50,25]:
    print("Target exposure: {}".format(target_exposure))
    save_folder = os.path.join(save_dir,'target_exposure_{}'.format(target_exposure))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        os.mkdir(os.path.join(save_folder,'images'))
        os.mkdir(os.path.join(save_folder, 'labels'))

    for file in cases:
        case = file.split('.')[0]
        exposures = []
        print("Case {}".format(case))
        if file.startswith('KiTS'):
            is_kits=True
            dmccase_fold = "/media/mcgoug01/nvme/Data/C4KC-KiTS/{}".format(case)
            dmccase_fold = os.path.join(dmccase_fold,os.listdir(dmccase_fold)[0])
            dmccase_fold = os.path.join(dmccase_fold,[fold for fold in os.listdir(dmccase_fold) if 'noncontrast' in fold.lower()][0])
            num_files = len(os.listdir(dmccase_fold))
            for i in range(10):
                dcm = pydicom.dcmread(os.path.join(dmccase_fold, os.listdir(dmccase_fold)[(num_files // 2) - 5 + i]))
                exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value) / 1000)
        else:
            is_kits=False
            dmccase_fold = "/media/mcgoug01/nvme/Data/AddenbrookesRCC/NCCT/Unenhanced//{}".format(case)
            dmccase_fold = os.path.join(dmccase_fold,os.listdir(dmccase_fold)[0])
            num_files = len(os.listdir(dmccase_fold))
            for i in range(10):
                dcm = pydicom.dcmread(os.path.join(dmccase_fold, os.listdir(dmccase_fold)[(num_files // 2) - 5 + i]))
                exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value) / 1000)

        exposure = np.mean(exposures)
        dose_fraction = target_exposure / exposure  # this is the fraction of dose we want to simulate
        CT_vol,aff,header= np.expand_dims(nib.load(os.path.join(nifti_folder,'images',file)).get_fdata(),0), nib.load(os.path.join(nifti_folder,'images',file)).affine, nib.load(os.path.join(nifti_folder,'images',file)).header
        sLDCT = convert_to_LDCT(CT_vol,A,dose_fraction=dose_fraction,
                                electronic_noise_sigma=electronic_noise_sigma,is_kits=is_kits)

        sLDCT_nifti = nib.Nifti2Image(sLDCT,aff,header)
        nib.save(sLDCT_nifti,os.path.join(save_folder,'images',file))
        shutil.copy(os.path.join(nifti_folder, 'labels', file), os.path.join(save_folder, 'labels', file))
