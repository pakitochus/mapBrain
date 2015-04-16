mapBrain
===================
A librrary to perform **Spherical Brain Mapping** on a 3D Brain Image. 

The **Spherical Brain Mapping** (SBM) is a framework intended to map the internal structures and features of the brain onto a 2D image that summarizes all this information, as described in [1] and [2]. 

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

References
---------------------
1. F.J. Martinez-Murcia et al. *Projecting MRI Brain images for the detection of Alzheimer's Disease*. **Stud Health Technol Inform**. 2014; 207:225-33. 
2. F.J. Martínez-Murcia et al. *A Spherical Brain Mapping of MR Images for the Detection of Alzheimer's Disease*. To be published.
