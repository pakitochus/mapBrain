#!/usr/bin/env python

"""
Spherical Brain Mapping of 3D Brain Images. 
3D brain imaging, such as MRI or PET produces a huge amount of data that is
currently analysed using uni or multivariate approaches. 
The main aim of SBM is to provide a new framework that allows the mapping  
of a 3D brain image to a two-dimensional space by means of some statistical 
measures. The system is based on a conversion from 3D spherical to 2D rectangular 
coordinates. For each spherical coordinate pair (theta,phi), a vector 
containing all voxels  in the radius is selected, and a number of values are 
computed, including statistical values (average, entropy, kurtosis) and 
morphological values (tissue thickness, distance to the central point, number of 
non-zero blocks). These values conform a two-dimensional image that can be 
computationally or even visually analysed.

The simplest approach is to use whichever measure that we want, and then apply
SBM to a brain image object, for example, imported using nibabel: 
    import nibabel as nib
    img = nib.load('MRIimage.nii')

We create an sbm object:
    import mapBrain
    sbm = mapBrain.SphericalBrainMapping()

And then, perform the SBM using 'average':
    map = sbm.doSBM(img.get_data(), measure='average', show=True)

Francisco Jesus Martinez Murcia, Spring 2015
"""

import numpy
from scipy.stats import kurtosis, skew

class SphericalBrainMapping(object):
    """ 
    Performs a Spherical Brain Mapping of a 3D Brain Image 
    """
    
    def __init__(self, resolution=1, deformation=0.0, ithreshold=0, nlayers=1):
        """
        Initializes a SBM instance, saving all parameters as attributes of the 
        instance. 
        resolution: Angle resolution at which each mapping vector is 
        computed (default 1 degree)
        deformation: Rate of unequally distributed mapping vectors, to be used 
        when the surface to be mapped is not spherical but ellipsoid (a float
        between 0-1, default 0 -> no deformation). 
        ithreshold: Intensity threshold for the projections needing it (default 0)
        nlayers: Nummber of equally distributed layers (default 1)
        """
        self.resolution = resolution
        self.deformation = deformation
        self.ithreshold = ithreshold
        self.nlayers = nlayers
        
    def setResolution(self, resolution=1):
        """ Sets the angular resolution at which the map will be computed
        :param resolution: Angular resolution at which each mapping vector 
        will be computed (default 1). 
        """
        self.resolution = resolution
        
    def setDeformation(self, deformation=0.0):
        """ Sets the deformation rate to be used in SBM. 
        :param deformation: Deformation rate (float 0-1)
        """
        self.deformation = deformation
        
    def setIThreshold(self, ithreshold=0):
        """ Sets the intensity threshold to be used in SBM.
        :param ithreshold: Intensity Threshold
        """
        self.ithreshold = ithreshold
    
    def setNLayers(self, nlayers=1):
        """ Sets the number of layers to be mapped
        :param nlayers: Nummber of equally distributed layers (default 1)
        """
        self.nlayers = nlayers
        
    def getResolution(self):
        """ Returns the resolution used in SBM. """
        return self.resolution
    
    def getDeformation(self):
        """ Returns the current deformation rate used in SBM """
        return self.deformation
        
    def getIThreshold(self):
        """ Returns the Intensity Threshold used in SBM """
        return self.ithreshold
        
    def getNLayers(self):
        """ Returns the number of layers used in SBM """
        return self.nlayers
        
    def computeMappingVectors(self):
        """ Computes the mapping vectors azim and elev
        """
        spaceVector = 1 - self.deformation*numpy.cos(numpy.deg2rad(numpy.arange(-2*180,2*180+self.resolution,self.resolution*2)))
        azim = numpy.deg2rad(numpy.cumsum(spaceVector)*self.resolution-270)
        elev = numpy.deg2rad(numpy.arange(-90, 90+self.resolution, self.resolution))
        return azim, elev
        
    def surface(self, set):
        """ Returns the surface of all mapped voxels
        :param set: Set of mapped voxels' intensity
        """
        return numpy.nanmax(numpy.argwhere(set>self.ithreshold))
        
    def thickness(self, set):
        """ Returns the thickness of the layer of mapped voxels
        :param set: Set of mapped voxels' intensity
        """
        thickness = numpy.nanmax(numpy.argwhere(set>self.ithreshold)) - numpy.nanmin(numpy.argwhere(set>self.ithreshold))
        return thickness
        
    def numfold(self, set):
        """ Returns the number of non-connected subsets in the mapped voxels
        :param set: Set of mapped voxels' intensity
        """
        return numpy.ceil(len(numpy.argwhere(numpy.bitwise_xor(set[:-1]>self.ithreshold, set[1:]>self.ithreshold)))/2.)
        
    def average(self, set):
        """ Returns the average of the sampling set
        :param set: Set of mapped voxels' intensity
        """
        return numpy.nanmean(set)     
           
    def variance(self, set):
        """ Returns the variance of the sampling set
        :param set: Set of mapped voxels' intensity
        """
        return numpy.nanvar(set)      
           
    def skewness(self, set):
        """ Returns the variance of the sampling set
        :param set: Set of mapped voxels' intensity
        """
        return skew(set, bias=False)
           
    def entropy(self, set):
        """ Returns the variance of the sampling set
        :param set: Set of mapped voxels' intensity
        """
        return sum(numpy.multiply(set[set>self.ithreshold],numpy.log(set[set>self.ithreshold])))
           
    def kurtosis(self, set):
        """ Returns the variance of the sampling set
        :param set: Set of mapped voxels' intensity
        """
        return kurtosis(set, fisher=False, bias=False)        
        
    def showMap(self, map, measure):
        """ Shows the computed maps in a window using pylab
        :param map: map or array of maps to be shown
        """
        import matplotlib.pyplot as plt
        minimum = numpy.min(map)
        maximum = numpy.max(map)
        if self.nlayers>1:
            imgplot = plt.figure()
            ncol = numpy.floor(self.nlayers/numpy.ceil(self.nlayers**(1.0/3)))
            nrow = numpy.ceil(self.nlayers/ncol)
            for nl in range(self.nlayers):
                plt.subplot(nrow,ncol,nl+1)
                plt.imshow(numpy.rot90(map[nl]),cmap='gray', vmin=minimum, vmax=maximum)
                plt.title(measure+'-SBM ('+str(nl)+')')
                plt.colorbar()
            plt.show()
                
        else:
            imgplot = plt.figure()
            plt.imshow(numpy.rot90(map[0]),cmap='gray', vmin=minimum, vmax=maximum)
            plt.title(measure+'-SBM')
            plt.colorbar()
            plt.show()
        
        return imgplot
                
            
            
    def sph2cart(self, theta, phi, r):
        """ Returns the corresponding spherical coordinates given the elevation,
        azimuth and radius
        :param theta: Azimuth angle
        :param phi: Elevation angle
        :param rad: Radius 
        """
        z = r * numpy.sin(phi)
        rcosphi = r * numpy.cos(phi)
        x = rcosphi * numpy.cos(theta)
        y = rcosphi * numpy.sin(theta)
        return x, y, z

            
    def doSBM(self, image, measure='average', show=True):
        """ Performs the SBM on the selected image and using the specified 
        measure
        :param image: Three-dimensional intensity array corresponding to a 3D 
        registered brain image. 
        :param measure: Measure used 
        :param show: Specifies whether to show the computed map (True) or not (False)
        """
        image[numpy.isnan(image)] = 0
        tam = image.shape                           # Size of the image
        puntoMedio = numpy.divide(image.shape,2)    # To compute the middle point
        lon = max(puntoMedio)                       # Compute the maximum value of the mapping vector v
        tamArr=numpy.repeat([tam],lon,0)

        # We create the mapping vectors and perform the conversion from spherical
        # coordinates to cartesian coordiantes (the ones in our 3D array). 
        azim, elev = self.computeMappingVectors()
        THETA,PHI,RAD = numpy.meshgrid(azim, elev, numpy.arange(lon))
        x,y,z = self.sph2cart(THETA,PHI,RAD)
        
        X = numpy.int32(numpy.round(x+puntoMedio[0]))
        Y = numpy.int32(numpy.round(y+puntoMedio[1]))
        Z = numpy.int32(numpy.round(z+puntoMedio[2]))
        coord = numpy.ravel_multi_index((X,Y,Z), mode='clip', dims=tam, order='F').transpose((1,0,2))

        # This is the blank map to be filled. 
        map = numpy.zeros([self.nlayers, numpy.ceil(361/self.resolution), numpy.ceil(181/self.resolution)])
        
        # Begin of the loop
        image = image.flatten('F')
        for nl in range(self.nlayers):
            intvl=numpy.int32(numpy.floor(lon/self.nlayers))
            for i in range(coord.shape[0]):
                for j in range(coord.shape[1]):
                    set = numpy.squeeze(image[coord[i][j][nl*intvl:(nl+1)*intvl]]
)
                    if measure.__class__==type('str'):
                        try:
                            map[nl][i][j] = getattr(self,measure)(set)
                        except AttributeError:
                            print "The measure "+measure+" is not supported"
                            return
                            
        # If it has been set, we display the map
        if show:
            self.showMap(map,measure)

        return map
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        