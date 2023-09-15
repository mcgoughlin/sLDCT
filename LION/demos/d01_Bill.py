
import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import nibabel as nib
from ts_algorithms import fbp
import pydicom
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

image = np.expand_dims(np.rot90(nib.load('/media/mcgoug01/nvme/Data/kits19_phases/noncontrast/KiTS-00000.nii.gz').get_fdata()[:,:,55],3),0)
normal_image = from_HU_to_normal(image)

dose_fraction = 0.5
num_detectors = 1216 # num detectors taken from https://s3.amazonaws.com/sgcimages/36_37_40_41_ITN1115_Siemens.pdf
electronic_noise_sigma = 0.25 # literally just an arbitrary guess - should be roughly 5% of total noise at 20% dose
angles = 360 # num angles taken from https://s3.amazonaws.com/sgcimages/36_37_40_41_ITN1115_Siemens.pdf

vg = ts.volume(shape=normal_image.shape, size=(5, 300, 300))
pg = ts.parallel(angles=angles, shape=(1, num_detectors), size=(1, num_detectors))
A = ts.operator(vg, pg)

expP = A(normal_image) # A() produces non-log transformed sinogram, hence expP


air = normal_image[:,10:30, 210:270]

dist = np.random.normal(loc=air.mean(), scale=air.std(), size=normal_image.shape)
expPair = A(dist) # this is expP of an equivalent 'air' sample, taken from original image
Noa = 1 / np.var(np.exp((np.log(expPair+1)*-1 - np.log(1)))) # number of incident photons! this is estimated using the air sino
# Need to estimate Noa based on dose characteristics within dcm rather than image,
# as it is not easy to extract an air sample from every image
print(Noa)

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
print("Electronic noise is {:.2f}% of total noise.".format(100*(electronic_noise_stddev.mean()+1e-9)/(electronic_noise_stddev.mean()+quantum_noise_stddev.mean()+1e-9)))

filtered_noise = (bowtie*noise)
sino_noisy = filtered_noise+expP

recon_noise = from_normal_to_HU(fbp(A,torch.Tensor(filtered_noise)).detach().cpu().numpy())[0]
recon_og = from_normal_to_HU(fbp(A,torch.Tensor(expP)).detach().cpu().numpy())[0]
recon_sLDCT = from_normal_to_HU(fbp(A,torch.Tensor(sino_noisy)).detach().cpu().numpy())[0]

print(recon_noise.std())

plt.figure(figsize=(18,6))
plt.subplot(121)
plt.imshow(recon_og,vmin=-200, vmax=200)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(recon_sLDCT,vmin=-200, vmax=200)
plt.title('Synthetic {}% Dose Image'.format(dose_fraction*100))
plt.axis('off')
plt.show()
