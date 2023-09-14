# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:41:55 2022

@author: mcgoug01
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon


"""
Reference for noise insertion technique:
Yu, Lifeng PhD; Shiung, Maria BA; Jondal, Dayna BA; McCollough, Cynthia H. PhD. 
Development and Validation of a Practical Lower-Dose-Simulation Tool for Optimizing Computed Tomography Scan Protocols. 
Journal of Computer Assisted Tomography: July/August 2012 - Volume 36 - Issue 4 - p 477-487 
doi: 10.1097/RCT.0b013e318258e891

"""



image = nib.load("/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/images/KiTS-00000.nii.gz").get_fdata()[200]
CT_air_calib = nib.load("/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data/raw_data/coreg_ncct/images/KiTS-00000.nii.gz").get_fdata()[600]

theta = np.linspace(0., 180., 2048, endpoint=False)
sinogram = radon(image, theta=theta)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]

#Selected air region from sinogram
# It is anticipated that this is a large source of error - this only provides a crude estimate of Noa as
#we have not actually scanned air on its own to calibrate the value of Noa
P = radon(CT_air_calib)

#indices below are meant to roughly mark the region of air scanning in this CT volume
expPair = (P[370:470,89:92]-P.min())/(P.max()-P.min())
Noa = 1/np.var(expPair) #estimating the effect of
#this is a set of fractions of the detections of incident photons, which is equal to exp(-Pair)


norm_recon= iradon(sinogram)
###estimating bowtie with above paper
gain = (3e4)/(5e3)

#peak at half-way
mu, sigma = len(sinogram)/2, 30 # mean and standard deviation
s = np.random.normal(mu, sigma, 10000)
count, bins, ignored = plt.hist(s, 511, density=True)
plt.close()
bowtie = 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) )
bowtie /= bowtie.max()*1.2
bowtie+=+0.16666666667


#normalise sinogram to make it between 0-1
norm_sino = (sinogram-sinogram.min())/(sinogram.max()-sinogram.min())

#reduce dose by this amount
a = 0.1
constant = (1-a)/a


#creating noises
normal = np.random.normal(size=sinogram.shape)
noise = np.sqrt(constant * norm_sino/Noa) * normal

LDCT_sino = (bowtie*noise.T).T+norm_sino

#project sLDCT back into sinogram magnitudes
sLDCT = LDCT_sino - LDCT_sino.mean()
sLDCT *= (sinogram.max()-sinogram.min())
sLDCT += sinogram.mean()
sLDCT = iradon(sLDCT)

### Plotting
def refactor(recon,base=image):
    # recon/=recon.max()
    recon /=recon.max()
    recon = np.where((recon==0) | ((recon* image.max())<-1024),-1024,recon * image.max())
    return recon
# fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8, 4.5))
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8, 4.5))


ax2.set_title("NCCT Sinogram")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
            aspect='auto')

ax1.set_title("Original NCCT Image")
ax1.imshow(image, cmap=plt.cm.Greys_r,
           vmax=300,
           vmin=-100)

ax3.set_title("Inverse Radon reconstruction of sLDCT")
ax3.imshow(refactor(sLDCT), cmap=plt.cm.Greys_r,
           vmax=300,
           vmin=-100)

ax4.set_title("Noise-Inserted Sinogram for sLDCT")
ax4.imshow(LDCT_sino, cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
            aspect='auto')

# plt.imshow(LDCT_sino, cmap=plt.cm.Greys_r,
#             extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
#             aspect='auto')
fig.tight_layout()
plt.show()