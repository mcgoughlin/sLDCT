# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import AItomotools.CTtools.ct_utils as ct
import nibabel as nib
from ts_algorithms import fbp
"""
Reference for noise insertion technique:
Yu, Lifeng PhD; Shiung, Maria BA; Jondal, Dayna BA; McCollough, Cynthia H. PhD. 
Development and Validation of a Practical Lower-Dose-Simulation Tool for Optimizing Computed Tomography Scan Protocols. 
Journal of Computer Assisted Tomography: July/August 2012 - Volume 36 - Issue 4 - p 477-487 
doi: 10.1097/RCT.0b013e318258e891

"""

def from_normal_to_HU(image):
    return np.maximum((image*3000)-1000,-1000)

image = np.rot90(nib.load('/media/mcgoug01/nvme/Data/kits19_phases/noncontrast/KiTS-00000.nii.gz').get_fdata()[:,:,55],3)
normal_image = np.expand_dims(ct.from_HU_to_normal(image),0)

dose_fraction = 0.5
num_detectors = 3600
electronic_noise_sigma = 1
angles = 360

vg = ts.volume(shape=normal_image.shape, size=(5, 300, 300))
pg = ts.parallel(angles=angles, shape=(1, num_detectors), size=(1, num_detectors))
A = ts.operator(vg, pg)

sino = A(normal_image)

air = normal_image[:, 100:150, 150:350]

dist = np.random.normal(loc=air.mean(), scale=air.std(), size=normal_image.shape)
P = A(dist)
Noa = 1/np.var(np.exp(-P)) # number of incident photons! this is estimated using an air sample
print(Noa)

mu, sigma = len(sino)/2, 5 # mean and standard deviation
s = np.random.normal(mu, sigma, 10000)
count, bins, ignored = plt.hist(s, num_detectors-1, density=True)
plt.close()
bowtie = 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) )
bowtie /= bowtie.max()*1.2
bowtie+=+0.16666666667
constant = (1-dose_fraction)/dose_fraction

electronic_noise = 1+((electronic_noise_sigma**2)*constant*sino/Noa)

normal = np.random.normal(size=sino.shape)
noise = np.sqrt(constant * (sino/Noa)*electronic_noise) * normal
filtered_noise = (bowtie*noise)
sino_noisy = filtered_noise+sino

recon_noise = from_normal_to_HU(fbp(A,torch.Tensor(filtered_noise)).detach().cpu().numpy())[0]
recon_og = from_normal_to_HU(fbp(A,torch.Tensor(sino)).detach().cpu().numpy())[0]
recon_sLDCT = from_normal_to_HU(fbp(A,torch.Tensor(sino_noisy)).detach().cpu().numpy())[0]

plt.figure(figsize=(18,6))
plt.subplot(131)
plt.imshow(recon_og,vmin=-200, vmax=200)
plt.subplot(132)
plt.imshow(recon_sLDCT,vmin=-200, vmax=200)
plt.subplot(133)
plt.imshow(recon_noise)
plt.show()
