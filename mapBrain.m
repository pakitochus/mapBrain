function [map,azim,elev] = mapBrain(I, varargin)
%MAPBRAIN performs a Spherical Brain Mapping of a 3D brain (from a struct I 
%that contains an field img) into a two dimensional plane, using the spherical
%coordinates and different approaches.
%
%   [MAP, AZIM, ELEV] = MAPBRAIN(I) takes the struct that contains the 3D image
%   in the field img, and projects it into two dimensions. The default
%   behaviour is a projection of the surface of the image I (using the
%   distance from the center to the last voxel with an intensity greater
%   than 0), just as in [1,2].
%
%   MAP contains the 2D projected image using different approaches
%   (default, the surface of the image I).
%
%   COORD contains the 3D vectors that are used to project the image.
%
%   MAPBRAIN(..., 'resolution', R) sets the angle resolution to be used in
%   the projection. Default is 1 degree.
%
%   MAPBRAIN(..., 'deformation', D) sets the rate of unequally distributed
%   mapping vectors, to be used when the surface to be mapped is not
%   spherical but ellipsoid.
%
%   MAPBRAIN(..., 'threshold', TH) sets the threshold for the projections
%   needing it (surface, thickness, numfold, mediahigh).
%
%   MAPBRAIN(..., 'nlayers', N) sets up the layered projection, resulting
%   in a set of N projections, each one based on one of N divisions of the
%   r vector, depicting from the most internal to the most external regions
%
%   in MAPBRAIN(..., type), type is a string that specifies which operation
%   is applied to the mapping vectors to obtain the final projection. The
%   following projection types are defined in this work:
%       'sum' calculates the sum of all values in the mapping vector.
%       This is the default behaviour.
%
%       'surface' calculates the external surface of the tissue, using the
%       distance from the center of the image to the most external voxel
%       with an intensity greater than TH.
%
%       'thickness' measures the thickness of the tissue by calculating the
%       distance from the first voxel greater than a threshold TH to the
%       last voxel that acomplishes this criterion.
%
%       'average' performs the projection by averaging al the
%       values of intensity in the mapping vector.
%
%       'mediahigh' performs the projection by averaging al the
%       values of intensity greater than a threshold in the mapping vector.
%
%       'var' performs the projection by obtaining the variance
%       of the values of the intensity in the mapping vector.
%
%       'kurtosis' performs the projection by obtaining the kurtosis
%       of the values of the intensity in the mapping vector.
%
%       'skewness' performs the projection by obtaining the skewness
%       of the values of the intensity in the mapping vector.
%
%       'entropy' performs the projection by obtaining an estimate of the
%       entropy of the mapping vector.
%
%       'numfold' performs the projection by estimating the
%       number of cortex folds that the vector R crosses.
%
%       'lbp'
%
%
% created with MATLAB ver.: 7.14.0.739 (R2012a) on Linux 3.11.0-15-generic
% #25-Ubuntu SMP Thu Jan 30 17:25:07 UTC 2014 i686
%
% created by: fjesusmartinez@ugr.es
% UPDATED: Feb 20th 2015
%
% REFs:
% [1] - F.J. Martinez-Murcia et al. Projecting MRI Brain images for the
%       detection of Alzheimer's Disease. 2014.
% [2] - F.J. Martinez-Murcia et al. A Statistical Projection of MRI Brain
%       Images Approach for the Detection of Alzheimer's Disease. Journal
%       of Current Alzheimer's Disease N(V). 2015.

%Default Parameters:
res =1;                 % Angular Resolution
deformation=0;          % Deformation ratio
umbral=0;               % Threshold value
mostrarImagen = false;  % Checks if image is to be shown.
nlayers=1;              % The whole radius is considered.

% Switches for different projections
projection='sum';

%Leemos varargin
if ~isempty(varargin)
    for i=1:length(varargin)
        parametro = varargin{i};
        if(ischar(parametro))
            switch(parametro)
                case 'resolution'
                    if (length(varargin)>i)&&(~isempty(varargin{i+1}))
                        res = varargin{i+1};
                    end
                case 'deformation'
                    if (length(varargin)>i)&&(~isempty(varargin{i+1}))
                        deformation = varargin{i+1};
                    end
                case 'threshold'
                    if (length(varargin)>i)&&(~isempty(varargin{i+1}))
                        th = varargin{i+1};
                        umbral = max(I.img(:))*th;
                    end
                case {'sum','thickness','surface','average','numfold',...
                        'mediahigh','var','skewness','entropy',...
                        'kurtosis','lbp'}
                    projection = parametro;
                case 'nlayers'
                    if (length(varargin)>i)&&(~isempty(varargin{i+1}))
                        nlayers= varargin{i+1};
                    end
                case 'verbose'
                    if (length(varargin)>i)&&(~isempty(varargin{i+1}))
                        mostrarImagen = true;
                    end
                otherwise
                    error(['The option ',parametro,' is not recognized.']);
            end
        elseif(~ischar(parametro))
            if(~ischar(varargin{i-1}))
                error(['Not recognized option number ',num2str(i)]);
            end
        end
    end
end

% I.img = permute(I.img, [1 3 2]);
tam=size(I.img);
puntoMedio = ceil(size(I.img)/2);

% We create a set of vectors to map the coordinates.
spaceVector=1-deformation*cos(deg2rad(-3*180:res*2:180));
azim = deg2rad(cumsum(spaceVector)*res-270);
elev = deg2rad(90:-res:-90);
lon = length(0:max(puntoMedio));
tamArr=repmat(tam',1,lon);
[X,Y,Z]=meshgrid(azim,elev,0:max(puntoMedio));

[x,y,z] = sph2cart(X,Y,Z);
X=round(x+puntoMedio(1));
Y=round(y+puntoMedio(2));
Z=round(z+puntoMedio(3));

X(X>tamArr(1))=tamArr(1);X(X<1)=1;
Y(Y>tamArr(2))=tamArr(2);Y(Y<1)=1;
Z(Z>tamArr(3))=tamArr(3);Z(Z<1)=1;
coord=permute(sub2ind(tam,X,Y,Z), [2 1 3]);

% And map the selected vectors.
map = zeros(nlayers,size(coord,1), size(coord,2));
hbin=zeros(prod(size(coord)),1);
s=1;
for nl=1:nlayers
    intvl=floor(lon/nlayers);
    for i=1:size(coord,1)
        for j=1:size(coord,2)
            radius=squeeze(I.img(coord(i,j,(1+(nl-1)*intvl):(nl*intvl))));
            switch(projection)
                case 'sum'
                    map(nl,i,j)=sum(radius(~isnan(radius)));
                case 'average'
                    map(nl,i,j)=mean(radius(~isnan(radius)));
                case 'var'
                    map(nl,i,j)=var(radius(~isnan(radius)));
                case 'skewness'
                    map(nl,i,j)=skewness(radius(~isnan(radius)));
                case 'kurtosis'
                    map(nl,i,j)=kurtosis(radius(~isnan(radius)));
                case 'entropy'
                    map(nl,i,j)=sum(radius((~isnan(radius)&(radius>0))).*log(radius((~isnan(radius)&(radius>0)))));
                case 'mediahigh'
                    map(nl,i,j)=mean(radius((radius>umbral)&(~isnan(radius))));
                case 'thickness'
                    radius=radius(~isnan(radius));
                    map(nl,i,j)=sum(radius>umbral);
                case 'numfold'
                    indices=find(radius>umbral);
                    if(numel(indices)>1)
                        t1=radius(indices(1):indices(end)-1)>umbral;
                        t2=radius(indices(1)+1:indices(end))>umbral;
                        map(nl,i,j)=sum(xor(t1,t2));
                    else
                        map(nl,i,j)=0;
                    end
                case 'surface'
                    if(isempty(find(radius>umbral, 1, 'last' )))
                        map(nl,i,j)=0;
                    else
                        map(nl,i,j)= find(radius>umbral, 1, 'last' );
                    end
                case 'lbp'
                    radiusplanes=zeros(3,3,find(radius>umbral,1,'last'));
                    for k=1:find(radius>umbral,1,'last')
                        cylcoords=squeeze(coord(i,j,k));
                        %radius2=squeeze(coord(i,j,length(radius)));
                        [rx ry rz]=ind2sub([121 145 121],cylcoords);R1=[rx ry rz];
                        %[rx ry rz]=ind2sub([121 145 121],radius2);R2=[rx ry rz];
                        %[cx cy cz]=cylinder2P(ones(length(radius),1),5,R1,R2);
                        %Co=som_unit_coords([121 145 121],'rect');
                        %a=[cx(k,1:4);cy(k,1:4);cz(k,1:4)];   % plano k perpencicular al eje
                        %planecoord(:,:,1)=[rx-1,ry-1,rz; rx,ry-1,rz; rx+1,ry-1,rz]; % coordenadas primera fila
                        %planecoord(:,:,2)=[rx-1,ry,rz;rx,ry,rz;rx+1,ry,rz]; % coordenadas segunda fila
                        %planecoord(:,:,3)=[rx-1,ry+1,rz; rx,ry+1,rz;rx+1,ry+1,rz]; % coordenadas tercera fila
                        
                        % Bordes
                        if (rx-1)==0 rx=2; end;
                        if (rx+1)>size(I.img,1) rx=size(I.img,1)-1; end;
                        if (ry-1)==0 ry=2; end;
                        if (ry+1)>size(I.img,2) ry=size(I.img,2)-1; end;
                        if (rz-1)==0 rz=2; end;
                        if (rz+1)>size(I.img,3) rz=size(I.img,3)-1; end;
                        
                        planeimg=[I.img(rx-1,ry-1,rz) I.img(rx,ry-1,rz) I.img(rx+1,ry-1,rz); I.img(rx-1,ry,rz) I.img(rx,ry,rz) I.img(rx+1,ry,rz); I.img(rx-1,ry+1,rz) I.img(rx,ry+1,rz) I.img(rx+1,ry+1,rz)];
                        %%%
                        %scatter3(rx-1,ry-1,rz,'bo');hold on;
                        %scatter3(rx,ry-1,rz,'bo');
                        %scatter3(rx+1,ry-1,rz,'bo');
                        %
                        %scatter3(rx-1,ry,rz,'ko');
                        %scatter3(rx,ry,rz,'ko');  % Eje del cilindro
                        %scatter3(rx+1,ry,rz,'ko');
                        %
                        %scatter3(rx-1,ry+1,rz,'go');
                        %scatter3(rx,ry+1,rz,'go');
                        %scatter3(rx+1,ry+1,rz,'go');
                        
                        radiusplanes(:,:,k)=planeimg;
                        
                    end;
                    radiusplanes=double(radiusplanes);
                    % Compute LBP descriptor
                    [declbpfeat binlbpfeat]=vol_lbp2(radiusplanes,4,1);  % 4 vecinos, radio=1
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%
                    % RotateIndex = 0;
                    % FRadius = 1;
                    % TInterval = 2;
                    % NeighborPoints = 4;
                    % TimeLength = 2;
                    % BorderLength = 1;
                    % bBilinearInterpolation = 1;
                    % fHistogram = RIVLBP(radiusplanes, TInterval, FRadius, NeighborPoints, BorderLength, TimeLength, RotateIndex, bBilinearInterpolation);
                    
                    map(nl,i,j)=declbpfeat;
                    hbin=sum(binlbpfeat+1);
                    map(nl,i,j)=map(nl,i,j)./hbin;
%                     map=log(map);
            end
            %Check for changes
            %Returns the distance
        end
    end
end

if(mostrarImagen)
    set(0, 'defaultTextInterpreter', 'latex');
    for i=1:nlayers
        figure(i);
        imagesc(azim,elev,rot90(squeeze(map(i,:,:))));colormap('bone');
        xlabel('Azimuth $\Theta$');
        hl=ylabel('Elevation $\varphi$');%set(hl,'Interpreter','latex');
        axis image;
    end
end
map=squeeze(map);
end


