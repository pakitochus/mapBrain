# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:21:04 2016

@author: pakitochus
"""
import numpy as np

def isAinB(A,B):
    # Checks if the rows of A are in B. 
    if A.ndim==1 or B.ndim==1:
        indList = np.all(A==B,axis=1)
    else:
        indList = [np.any(np.all(a==B,axis=1),axis=0) for a in A ]
    return indList

def neighbourSet(center, radius, im):
    # This subfunction is used to compute the L2-norm support ball 
    # with radius RADIUS. 
    tam = im.shape
    xx, yy, zz = np.meshgrid(np.arange(-radius,radius+1), np.arange(-radius,radius+1), np.arange(-radius,radius+1), sparse=False)
    mask = (xx**2+yy**2+zz**2)<=radius**2    
    subset = np.column_stack((xx[mask]+center[0], yy[mask]+center[1], zz[mask]+center[2]))
    subset = subset[(subset[:,0]<tam[0])&(subset[:,1]<tam[1])&(subset[:,2]<tam[2])&(subset[:,0]>=0)&(subset[:,1]>=0)&(subset[:,2]>=0),:]    
    c = im[subset[:,0], subset[:,1], subset[:,2]]    
    return subset,c
    
def unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]
    
def extraerPuntos(azim,elev,cent,tam):
    # This subfunction extract the points that are contained in the 
    # mapping vector used in SBM. 
    radia = np.arange(0,np.max(np.array(tam)-cent))
    X = np.floor(radia*np.sin(azim*np.pi/180)*np.cos(elev*np.pi/180)+cent[0])
    Y = np.floor(radia*np.sin(azim*np.pi/180)*np.sin(elev*np.pi/180)+cent[1])
    Z = np.floor(radia*np.cos(azim*np.pi/180)+cent[2])
    seleccion = (X>tam[0])|(Y>tam[1])|(Z>tam[2])|(X<0)|(Y<0)|(Z<0)
    data = unique_rows(np.column_stack((X[~seleccion],Y[~seleccion],Z[~seleccion])))
    return data[:,0],data[:,1],data[:,2]  
    
def hmmPaths(center, im, Ith, e, azim, elev, xfin):
    """
    RBFPATHS performs a path tracing in a 3D intensity map (e.g. an MRI image)
    using Hidden Markov Models and a RBF based robability function, as in [1].
    
       POINTS = HMMPATHS(CENTER, IM, ITH, E, ...) creates radiuses starting in
       point CENTER in the 3D intensity map defined by IM, using the intensity
       threshold in ITH as a stop condition and the L2-norm support ball of
       radius E to extract the candidates in each step. 
     
       POINTS = HMMPATHS(CENTER, IM, ITH, E, AZIM, ELEV, []) uses the
       information of the radial direction coded in (AZIM,ELEV) to create an
       attractor in the limits of the image and force that direction. 
    
       POINTS = HMMPATHS(CENTER, IM, ITH, E, [], [], XFIN) forces the
       attractor in the point defined by XFIN. +
    
       [1] F.J. Martinez-Murcia et al. A Structural Parametrization of the 
       Brain using Hidden Markov Models Based Paths in Alzheimer's Disease. 
       International Journal of Neural Systems. 
     
    """
    points = center
    nspoint = center
    tam = im.shape
    iterLim = 1e3
    t=0
    # The final point of the SBM mapping vector will be the attractor
    X,Y,Z = extraerPuntos(azim, elev, center, tam)
    xfin = np.array([X[-1],Y[-1],Z[-1]]) 
    cVar = (0.05*(im.max()-im.min()))**2
    condicion = True
    while condicion:
        # Compute the neighbour set
        if t!=0:
            nspoint = points[:,t]
        subset,ci = neighbourSet(nspoint,e,im)
        select = ~isAinB(subset,points)&(ci>0)
        ci = ci[select]
        c = im[nspoint[0],nspoint[1],nspoint[2]]
        N,dim = subset.shape
        #update wi 
        wi = 1/np.sqrt(2*np.pi*cVar)*np.exp(-((c-ci)**2)/(2*cVar))




condicion = true;
#if(debug)
#    slice(double(permute(im,[2 1 3])), center(1), center(2), center(3));
#    colormap('gray');
#    shading flat
#    hold on
#    view(0,90)
#    axis([1 121 1 145 1 121]);
#    daspect([1 1 1]);
#end
cTh = Ith;
while condicion
    % Compute the neighbour set. 
    [subset,ci] = neighbourSet(points(:,t), e, im);
    ci = double(ci);
    select = ~isAinB(subset,points)&ci>0; 
    subset = subset(:,select);
    ci = ci(select);
    c = double(im(points(1,t), points(2,t), points(3,t)));
    [dim, N] = size(subset);
    if(N==0)
        return;
    end
    % update wi,
    wi = 1/sqrt(2*pi*cVar)*exp(-((c-ci).^2)/(2*cVar));

    % update RBF,
    difi = bsxfun(@minus, subset, xfin);
    g = zeros(1,N);
    pVar = 0.05*norm(xfin-points(:,t))^(2); % sets the variance as the 5% of the squared distance.
    for i=1:N
        term1 = sqrt((2*pi)^dim*pVar);
        term2 = -norm(difi(:,i))^2/(pVar*dim);
        g(i) = 1/term1*exp(term2);
    end
    [~, ind] = max(g.*wi);

    puntosRepeat= ismember(subset(:,ind)',points', 'rows'); 
    % Stop condition
    condicion =~all(subset(:,ind)==xfin)&&t<iterLim&&(~puntosRepeat)&&ci(ind)>cTh&&subset(1,ind)<=tam(1)&&subset(2,ind)<=tam(2)&&subset(3,ind)<=tam(3)&&subset(1,ind)>0&&subset(2,ind)>0&&subset(3,ind)>0;
    if(condicion)
        points(:,t+1)=subset(:,ind);
        t = t+1;
        if(debug)
            plot3(points(1,:), points(2,:), points(3,:),'ro-');
            drawnow
	    fprintf('.');
        end
    end
end
if(debug)
    fprintf('\n');
    hold off
end
end

