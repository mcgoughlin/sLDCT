
import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import nibabel as nib
from ts_algorithms import fbp
import pydicom
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

nifti_folder = '/media/mcgoug01/nvme/Data/kits19_phases/noncontrast/'
cases = os.listdir(nifti_folder)
cases.sort()
target_exposure = 80 # mAs

num_detectors = 1216  # num detectors taken from https://s3.amazonaws.com/sgcimages/36_37_40_41_ITN1115_Siemens.pdf
electronic_noise_sigma = 0.1  # literally just an arbitrary guess
angles = 360  # arbitrary

vg = ts.volume(shape=(1, 512, 512), size=(5, 300, 300))
pg = ts.parallel(angles=angles, shape=(1, num_detectors), size=(1, num_detectors))
A = ts.operator(vg, pg)

def convert_to_LDCT(CT_vol,exposure=250,target_exposure=80,electronic_noise_sigma=0.1):
    LDCT_vol = np.zeros_like(CT_vol)
    for slice in range(CT_vol.shape[-1]):
        image = CT_vol[:,:,:,slice]
        normal_image = from_HU_to_normal(image)
        expP = A(normal_image)  # A() produces non-log transformed sinogram, hence expP
        Noa = int(exposure * 820 / 250)  # this is the number of incident photons at 250mAs dose
        dose_fraction = target_exposure / exposure  # this is the fraction of dose we want to simulate

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

        sLDCT_slice = from_normal_to_HU(fbp(A, torch.Tensor(sino_noisy)).detach().cpu().numpy())
        LDCT_vol[:,:,:,slice] = sLDCT_slice

    return LDCT_vol

for file in cases:
    case = file.split('.')[0]
    dmccase_fold = "/media/mcgoug01/nvme/Data/C4KC-KiTS/{}".format(case)
    dmccase_fold = os.path.join(dmccase_fold,os.listdir(dmccase_fold)[0])
    dmccase_fold = os.path.join(dmccase_fold,[fold for fold in os.listdir(dmccase_fold) if 'noncontrast' in fold.lower()][0])
    num_files = len(os.listdir(dmccase_fold))

    exposures = []
    for i in range(10):
        dcm = pydicom.dcmread(os.path.join(dmccase_fold, os.listdir(dmccase_fold)[(num_files // 2) - 5 + i]))
        exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value) / 1000)
    exposure = np.mean(exposures)

    CT_vol = np.expand_dims(np.rot90(nib.load(os.path.join(nifti_folder,file)).get_fdata(), 3), 0)

    exposures = []
    for i in range(10):
        dcm = pydicom.dcmread(os.path.join(dmccase_fold,os.listdir(dmccase_fold)[(num_files//2)-5 + i]))
        exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value)/1000)
    exposure = np.mean(exposures)

    sLDCT = convert_to_LDCT(CT_vol,exposure=exposure,target_exposure=target_exposure,electronic_noise_sigma=electronic_noise_sigma)
    # base dose reading on mid-scan slices


    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.imshow(CT_vol[0,:,:,num_files//2],vmin=-200, vmax=200)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(sLDCT[0,:,:,num_files//2],vmin=-200, vmax=200)
    plt.title('Synthetic Dose Image')
    plt.axis('off')
    plt.show(block=True)
