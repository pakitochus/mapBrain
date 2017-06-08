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
REFs:
[1] - F.J. Martinez-Murcia et al. Projecting MRI Brain images for the
      detection of Alzheimer's Disease. Innovation in Medicine and 
      Healthcare 207:225 2014.
[2] - F.J. Martinez-Murcia et al. A Spherical Brain Mapping of MR Images 
      for the Detection of Alzheimer's Disease. Journal
      of Current Alzheimer's Research. 13(5):575-88. 2016. 
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
        
    def vsetResolution(self, resolution=1):
        """ vsets the angular resolution at which the map will be computed
        :param resolution: Angular resolution at which each mapping vector 
        will be computed (default 1). 
        """
        self.resolution = resolution
        
    def vsetDeformation(self, deformation=0.0):
        """ vsets the deformation rate to be used in SBM. 
        :param deformation: Deformation rate (float 0-1)
        """
        self.deformation = deformation
        
    def vsetIThreshold(self, ithreshold=0):
        """ vsets the intensity threshold to be used in SBM.
        :param ithreshold: Intensity Threshold
        """
        self.ithreshold = ithreshold
    
    def vsetNLayers(self, nlayers=1):
        """ vsets the number of layers to be mapped
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
        
    def surface(self, vset):
        """ Returns the surface of all mapped voxels
        :param vset: set of mapped voxels' intensity
        """
        return numpy.nanmax(numpy.argwhere(vset>self.ithreshold))
        
    def thickness(self, vset):
        """ Returns the thickness of the layer of mapped voxels
        :param vset: set of mapped voxels' intensity
        """
        aux = numpy.argwhere(vset>self.ithreshold)
        if aux.size>0:
            thickness = numpy.nanmax(aux) - numpy.nanmin(aux)
        else:
            thickness = 0
        return thickness
        
    def numfold(self, vset):
        """ Returns the number of non-connected subvsets in the mapped voxels
        :param vset: set of mapped voxels' intensity
        """
        return numpy.ceil(len(numpy.argwhere(numpy.bitwise_xor(vset[:-1]>self.ithreshold, vset[1:]>self.ithreshold)))/2.)
        
    def average(self, vset):
        """ Returns the average of the sampling vset
        :param vset: set of mapped voxels' intensity
        """
        return numpy.nanmean(vset)     
           
    def variance(self, vset):
        """ Returns the variance of the sampling vset
        :param vset: set of mapped voxels' intensity
        """
        return numpy.nanvar(vset)      
           
    def skewness(self, vset):
        """ Returns the skewness of the sampling vset
        :param vset: set of mapped voxels' intensity
        """
        return skew(vset, bias=False)
           
    def entropy(self, vset):
        """ Returns the entropy of the sampling vset
        :param vset: set of mapped voxels' intensity
        """
        return sum(numpy.multiply(vset[vset>self.ithreshold],numpy.log(vset[vset>self.ithreshold])))
           
    def kurtosis(self, vset):
        """ Returns the kurtosis of the sampling vset
        :param vset: set of mapped voxels' intensity
        """
        return kurtosis(vset, fisher=False, bias=False)        


    def _interp_single_gray_level(self, p, imag):
        ''' Interpolates the gray level found at the point p
        using interpolation by percentage of superposition of
        pixels.
        ''' 
        # We create the array of surrounding points: 
        a = numpy.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [1, 0, 0],
                      [1, 0, 1],
                      [1, 1, 0],
                      [1, 1, 1]])
        points = (numpy.floor(p)+a).astype(int)
        
        # Extract the colours at each point:
        c = imag[points[:,0], points[:,1], points[:,2]]
        
        # Calculate the weights as distances:
        w = numpy.prod(1-numpy.abs(p-points),axis=1)
        
        # And sum the different values: 
        ci = numpy.sum(c*w)
        return ci
    
    def _interp_gray_level(self, p, imag):
        ''' Interpolates the gray level found at the point array p
        by using superposition interpolation. 
        ''' 
        if p.ndim==3:
            ci = numpy.zeros(p.shape[:2])
            for i in range(p.shape[0]):
                ci[i,:] = numpy.array([self._interp_single_gray_level(punto,imag) for punto in p[i,:,:]])
                
        elif p.ndim==2:
            ci = numpy.array([self._interp_single_gray_level(punto,imag) for punto in p])
        return ci


    def _get_points_centering(self, center, n):
        ''' 
        Gets the array of centerpoints in a given 
        direction (n), in this order:
        -------------
        | 1 | 2 | 3 |
        |------------
        | 8 | 0 | 4 |
        |-----------|
        | 7 | 6 | 5 |
        -------------
        '''
        u11 = numpy.sqrt(n[1]**2/(n[0]**2+n[1]**2))
        u12 = -numpy.sqrt(n[1]**2/(n[0]**2+n[1]**2))
        u21 = numpy.sqrt(1-u11**2)
        u22 = numpy.sqrt(1-u12**2)
        
        u1 = numpy.array([u11, u21, 0])
        u2 = numpy.array([u12, u22, 0])
        
        if numpy.dot(u1,n)<1e-10:
            u = u1
        else:
            u = u2
        
        v = numpy.cross(n,u)
        
        p0 = center
        p1 = center - u + v
        p2 = center + v
        p3 = center + u + v
        p4 = center + u
        p5 = center + u - v
        p6 = center - v
        p7 = center - u - v
        p8 = center - u
        return numpy.array([p0, p1, p2, p3, p4, p5, p6, p7, p8])
    
    def _posterizeImage(self, ndarray, numLevels = 16 ):
        ''' 
        Posterizes the image to number of levels
        '''
        
        #Gray-level resizing
        numLevels = numLevels-1 #don't touch. Logical adding issue.
        minImage = numpy.nanmin(ndarray)
        ndarray = ndarray-(minImage)
        maxImage = numpy.nanmax(ndarray)
        tempShift = maxImage/numLevels
        ndarray = numpy.floor(ndarray/tempShift)
        ndarray=ndarray + 1
        numLevels = numLevels + 1 # don't touch. Logical adding issue.
        
        return ndarray


    def computeTexture(self, p, imag, center, distances=1):
        '''
        Computes the texture around vector p
        '''
        from mahotas.features import texture
        
        origins = self._get_points_centering(center, p[1,:])
        puntos = numpy.array([p+cent for cent in origins])
        select = (puntos<numpy.array(imag.shape)-1).all(axis=2).all(axis=0)
        p_real = puntos[:,select,:]
        
        colors = self._interp_gray_level(p_real, imag)
        
        ndarray = self._posterizeImage(colors)
        
        # Prevent errors with iterative list:
        if (type(distances) is not list) and (type(distances) is not numpy.ndarray):
            distances = [distances]        
        
        glcm=[]
        for dis in distances:
            glcm.append(texture.cooccurence(ndarray.astype(int)-1, 0, distance=dis, symmetric=False))
        features = texture.haralick_features(glcm)
        labels = texture.haralick_labels
        return features, labels
    
    
    def showMap(self, map, measure, cmap='gray'):
        """ Shows the computed maps in a window using pyplot
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
                plt.imshow(numpy.rot90(map[nl]),cmap=cmap, vmin=minimum, vmax=maximum)
                plt.title(measure+'-SBM ('+str(nl)+')')
                plt.colorbar()
            plt.show()
        elif measure=='texture':
            imgplot = plt.figure()
            ncol = numpy.floor(13/numpy.ceil(13**(1.0/3)))
            nrow = numpy.ceil(13/ncol)
            for nl in range(13):
                plt.subplot(nrow,ncol,nl+1)
                plt.imshow(numpy.rot90(map[nl,:,:]),cmap=cmap, vmin=minimum, vmax=maximum)
                plt.title(measure+'-SBM ('+str(nl)+')')
                plt.colorbar()
            plt.show()
                
        else:
            imgplot = plt.figure()
            plt.imshow(numpy.rot90(map[0]),cmap=cmap, vmin=minimum, vmax=maximum)
            plt.title(measure+'-SBM')
            plt.colorbar()
            plt.show()
        
        return imgplot
                
            
            
    def sph2cart(self, theta, phi, r):
        """ Returns the corresponding spherical coordinates given the elevation,
        azimuth and radius
        :param theta: Azimuth angle (radians)
        :param phi: Elevation angle (radians)
        :param rad: Radius 
        """
        z = r * numpy.sin(phi)
        rcosphi = r * numpy.cos(phi)
        x = rcosphi * numpy.cos(theta)
        y = rcosphi * numpy.sin(theta)
        return x, y, z

            
    def doSBM(self, image, measure='average', show=True, centre=None):
        """ Performs the SBM on the selected image and using the specified 
        measure
        :param image: Three-dimensional intensity array corresponding to a 3D 
        registered brain image. 
        :param measure: Measure used 
        :param show: Specifies whether to show the computed map (True) or not (False)
        """
        image[numpy.isnan(image)] = 0
        tam = image.shape                           # Size of the image
        if centre is None:
            centre = numpy.divide(image.shape,2)    # To compute the middle point
        lon = max(centre)                       # Compute the maximum value of the mapping vector v
#        tamArr=numpy.repeat([tam],lon,0) # Residual

        # We create the mapping vectors and perform the conversion from spherical
        # coordinates to cartesian coordiantes (the ones in our 3D array). 
        azim, elev = self.computeMappingVectors()
        THETA,PHI,RAD = numpy.meshgrid(azim, elev, numpy.arange(lon))
        x,y,z = self.sph2cart(THETA,PHI,RAD)
        
        if measure=='texture':
            # Define the blank map to be filled (nlayers, features, 361, 181)
            mapa = numpy.zeros([13, numpy.ceil(361/self.resolution).astype(int), numpy.ceil(181/self.resolution).astype(int)])
#            for nl in range(self.nlayers):
#                intvl=numpy.int32(numpy.floor(lon/self.nlayers)) # no sirve para nada todavía
            # eliminamos el número de capas
            for i in range(azim.shape[0]):
                for j in range(elev.shape[0]):
                    p = numpy.vstack((x[j,i,:].flatten(),y[j,i,:].flatten(),z[j,i,:].flatten())).T
                    mapa[:,i,j], _ = self.computeTexture(p, image, centre)
            
        else:
            X = numpy.int32(numpy.round(x+centre[0]))
            Y = numpy.int32(numpy.round(y+centre[1]))
            Z = numpy.int32(numpy.round(z+centre[2]))
            coord = numpy.ravel_multi_index((X,Y,Z), mode='clip', dims=tam, order='F').transpose((1,0,2))
    
            # This is the blank map to be filled. 
            mapa = numpy.zeros([self.nlayers, numpy.ceil(361/self.resolution).astype(int), numpy.ceil(181/self.resolution).astype(int)])
            
            # Begin of the loop
            image = image.flatten('F')
            for nl in range(self.nlayers):
                intvl=numpy.int32(numpy.floor(lon/self.nlayers))
                for i in range(coord.shape[0]):
                    for j in range(coord.shape[1]):
                        vset = numpy.squeeze(image[coord[i][j][nl*intvl:(nl+1)*intvl]])
                        if measure.__class__==type('str'):
                            try:
                                mapa[nl][i][j] = getattr(self,measure)(vset)
                            except AttributeError:
                                print("The measure %s is not supported"%measure)
                                return
                            
        # If it has been vset, we display the map
        if show:
            self.showMap(mapa,measure)

        return mapa
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
