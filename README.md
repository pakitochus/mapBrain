mapBrain
===================
[![DOI](https://zenodo.org/badge/21407/pakitochus/mapBrain.svg)](https://zenodo.org/badge/latestdoi/21407/pakitochus/mapBrain)

A library to perform **Spherical Brain Mapping** on a 3D Brain Image. 

The **Spherical Brain Mapping** (SBM) is a framework intended to map the internal structures and features of the brain onto a 2D image that summarizes all this information, as described in [1] and previously presented in [2] and [3]. 3D brain imaging, such as MRI or PET produces a huge amount of data that is currently analysed using uni or multivariate approaches. 

SBM provides a new framework that allows the mapping of a 3D brain image to a two-dimensional space by means of some statistical measures. The system is based on a conversion from 3D spherical to 2D rectangular coordinates. For each spherical coordinate pair (theta,phi), a vector containing all voxels  in the radius is selected, and a number of values are computed, including statistical values (average, entropy, kurtosis) and morphological values (tissue thickness, distance to the central point, number of non-zero blocks). These values conform a two-dimensional image that can be computationally or even visually analysed.

A new structural parametrization of MRI images has been added, using a modified hidden markov model to trace routes that follow minimal intensity change paths inside the brain, instead of the rectilinear paths used in typical SBM [4]. This file, currently only working in MATLAB, is contained in the file `hmmPaths.m`.


Installation
----------------
Copy the *.py files directly to the working directory, and import the library with `import mapBrain`. 

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
1. F.J. Martinez-Murcia et al. *A Spherical Brain Mapping of MR images for the detection of Alzheimer's Disease*. **Current Alzheimer Research** 13(5):575-88. 2016. 
2. F.J. Martinez-Murcia et al. *Projecting MRI Brain images for the detection of Alzheimer's Disease*. **Stud Health Technol Inform** 207, 225-33. 2014. 
3. F.J. Martínez-Murcia et al. *A Volumetric Radial LBP Projection of MRI Brain Images for the Diagnosis of Alzheimer’s Disease*. **Lecture Notes in Computer Science** 9107, 19-28. 2015.
4. F.J. Martinez-Murcia et al. *A Structural Parametrization of the Brain Using Hidden Markov Models-Based Paths in Alzheimer's Disease*. **International Journal of Neural Systems** 26(6) 1650024. 2016. 
