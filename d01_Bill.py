
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

nifti_folder = '/media/mcgoug01/nvme/ThirdYear/MastersProject/data/original_ncct/over_150mAs/images'
cases = os.listdir(nifti_folder)
cases.sort()
target_exposure = 80 # mAs

num_detectors = 1216  # num detectors taken from https://s3.amazonaws.com/sgcimages/36_37_40_41_ITN1115_Siemens.pdf
electronic_noise_sigma = 0.1  # literally just an arbitrary guess
angles = 360  # arbitrary

vg = ts.volume(shape=(1, 512, 512), size=(5, 300, 300))
pg = ts.parallel(angles=angles, shape=(1, num_detectors), size=(1, num_detectors))
A = ts.operator(vg, pg)

def convert_to_ncct(CT_vol,exposure=250,target_dose=80):
    for slice in range(CT_vol.shape[-1]):
        image[slice] = CT_vol[slice]
        normal_image = from_HU_to_normal(image)
        expP = A(normal_image)  # A() produces non-log transformed sinogram, hence expP

        air = normal_image[:, 10:30, 210:270]

        dist = np.random.normal(loc=air.mean(), scale=air.std(), size=normal_image.shape)
        expPair = A(dist)  # this is expP of an equivalent 'air' sample, taken from original image
        # we estimated Noa using an air section of an image - 820 photons at 250mAs dose - KiTS-00000.
        # If we assume number of incident photons is proportional to dose, then we can estimate Noa for any dose
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

        electronic_noise_stddev = term1 * term2
        quantum_noise_stddev = term1
        print("Case {}. Exposure: {:.1f}. Dose Fraction {:.2f}. Number of incident photons: {}.".format(case, exposure,
                                                                                                        dose_fraction,
                                                                                                        Noa))

        filtered_noise = (bowtie * noise)
        sino_noisy = filtered_noise + expP

        recon_noise = from_normal_to_HU(fbp(A, torch.Tensor(filtered_noise)).detach().cpu().numpy())[0]
        recon_og = from_normal_to_HU(fbp(A, torch.Tensor(expP)).detach().cpu().numpy())[0]
        recon_sLDCT = from_normal_to_HU(fbp(A, torch.Tensor(sino_noisy)).detach().cpu().numpy())[0]

for file in cases:
    case = file.split('.')[0]
    if not case.startswith('RCC_049'):continue
    dmccase_fold = "/media/mcgoug01/nvme/Data/AddenbrookesRCC/NCCT/Unenhanced/{}".format(case)
    dmccase_fold = os.path.join(dmccase_fold,os.listdir(dmccase_fold)[0])
    # dmccase_fold = os.path.join(dmccase_fold,[fold for fold in os.listdir(dmccase_fold) if 'noncontrast' in fold.lower()][0])
    num_files = len(os.listdir(dmccase_fold))

    exposures = []
    for i in range(10):
        dcm = pydicom.dcmread(os.path.join(dmccase_fold, os.listdir(dmccase_fold)[(num_files // 2) - 5 + i]))
        exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value) / 1000)
    exposure = np.mean(exposures)
    print(nib.load(os.path.join(nifti_folder,file)).get_fdata().max())
    image = np.expand_dims(np.rot90(nib.load(os.path.join(nifti_folder,file)).get_fdata()[:, :, -40], 3), 0)
    print(image.shape,image.max(),image.min())
    normal_image = from_HU_to_normal(image)

    exposures = []
    for i in range(10):
        dcm = pydicom.dcmread(os.path.join(dmccase_fold,os.listdir(dmccase_fold)[(num_files//2)-5 + i]))
        exposures.append(int(dcm[0x0018, 0x1150].value) * int(dcm[0x0018, 0x1151].value)/1000)
    exposure = np.mean(exposures)
    # base dose reading on mid-scan slices

    expP = A(normal_image) # A() produces non-log transformed sinogram, hence expP

    air = normal_image[:,30:50, 210:270]
    dist = np.random.normal(loc=air.mean(), scale=air.std(), size=normal_image.shape)
    expPair = A(dist) # this is expP of an equivalent 'air' sample, taken from original image
    # we estimated Noa using an air section of an image - 820 photons at 250mAs dose - KiTS-00000.
    # If we assume number of incident photons is proportional to dose, then we can estimate Noa for any dose
    Noa = 1/ np.var(np.exp(np.log(expPair+1e-6)-np.log(1e-6)))
    print(Noa)
    Noa = int(exposure * 1340 / 224)  # this is the number of incident photons at 250mAs dose
    print(Noa)
    dose_fraction = target_exposure/exposure # this is the fraction of dose we want to simulate

    mu, sigma = len(expP)/2, 5 # mean and standard deviation
    s = np.random.normal(mu, sigma, 10000)
    count, bins, ignored = plt.hist(s, num_detectors-1, density=True)
    plt.close()
    bowtie = 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) )
    bowtie /= bowtie.max()*1.2
    bowtie+=+0.16666666667

    # equation 11 in paper referenced at top of script
    normal = np.random.normal(size=expP.shape)
    term1 = ((1-dose_fraction)/dose_fraction) * (expP/Noa)
    term2 = ((1+dose_fraction)/dose_fraction) * (electronic_noise_sigma**2)*expP/Noa
    noise = np.sqrt(term1 * (1+term2)+1e-9) * normal

    electronic_noise_stddev = term1*term2
    quantum_noise_stddev = term1
    print("Case {}. Exposure: {:.1f}. Dose Fraction {:.2f}. Number of incident photons: {}.".format(case,exposure,dose_fraction,Noa))

    filtered_noise = (bowtie*noise)
    sino_noisy = filtered_noise+expP

    recon_noise = from_normal_to_HU(fbp(A,torch.Tensor(filtered_noise)).detach().cpu().numpy())[0]
    recon_og = from_normal_to_HU(fbp(A,torch.Tensor(expP)).detach().cpu().numpy())[0]
    recon_sLDCT = from_normal_to_HU(fbp(A,torch.Tensor(sino_noisy)).detach().cpu().numpy())[0]


    plt.figure(figsize=(18,6))
    plt.subplot(121)
    plt.imshow(recon_og,vmin=-200, vmax=200)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(recon_sLDCT,vmin=-200, vmax=200)
    plt.title('Synthetic {:.2f}% Dose Image'.format(dose_fraction*100))
    plt.axis('off')
    plt.show(block=True)
