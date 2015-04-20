mapBrain
===================
A library to perform **Spherical Brain Mapping** on a 3D Brain Image. 

The **Spherical Brain Mapping** (SBM) is a framework intended to map the internal structures and features of the brain onto a 2D image that summarizes all this information, as described in [1] and [2]. 3D brain imaging, such as MRI or PET produces a huge amount of data that is currently analysed using uni or multivariate approaches. 

SBM provides a new framework that allows the mapping of a 3D brain image to a two-dimensional space by means of some statistical measures. The system is based on a conversion from 3D spherical to 2D rectangular coordinates. For each spherical coordinate pair (theta,phi), a vector containing all voxels  in the radius is selected, and a number of values are computed, including statistical values (average, entropy, kurtosis) and morphological values (tissue thickness, distance to the central point, number of non-zero blocks). These values conform a two-dimensional image that can be computationally or even visually analysed.


Installation
----------------

Usage
-----------------
The Statistical Brain Mapping is structured as a class that can be invoked from every script. The simplest approach would be using: 
```python
import mapBrain
import nibabel as nib

img = nib.load('MRIimage.nii')
sbm = mapBrain.SphericalBrainMapping()
map = sbm.doSBM(img.get_data(), measure='average', show=True)
```
To-Do
-----------------
- Add support for functions as objects
- Add support for different sampling methods

References
---------------------
1. F.J. Martinez-Murcia et al. *Projecting MRI Brain images for the detection of Alzheimer's Disease*. **Stud Health Technol Inform**. 2014; 207:225-33. 
2. F.J. Martínez-Murcia et al. *A Spherical Brain Mapping of MR Images for the Detection of Alzheimer's Disease*. To be published.
