function points = hmmPaths(center, im, Ith, e, azim, elev, xfin)
%RBFPATHS performs a path tracing in a 3D intensity map (e.g. an MRI image)
%using Hidden Markov Models and a RBF based robability function, as in [1].
%
%   POINTS = HMMPATHS(CENTER, IM, ITH, E, ...) creates radiuses starting in
%   point CENTER in the 3D intensity map defined by IM, using the intensity
%   threshold in ITH as a stop condition and the L2-norm support ball of
%   radius E to extract the candidates in each step. 
% 
%   POINTS = HMMPATHS(CENTER, IM, ITH, E, AZIM, ELEV, []) uses the
%   information of the radial direction coded in (AZIM,ELEV) to create an
%   attractor in the limits of the image and force that direction. 
%
%   POINTS = HMMPATHS(CENTER, IM, ITH, E, [], [], XFIN) forces the
%   attractor in the point defined by XFIN. +
%
%   [1] F.J. Martinez-Murcia et al. A Structural Parametrization of the 
%   Brain using Hidden Markov Models Based Paths in Alzheimer's Disease. 
%   International Journal of Neural Systems. 
% 
%   REQUIRES MATLAB > R2012a

points = center;
debug =0; % 1 if debugging, 0 if not. 
tam = size(im);
iterLim = 1e3;
t = 1;
if(~isempty(azim)&&~isempty(elev))
    % We extract the SBM mapping vector to use its final point
    % as attractor. 
    [X, Y, Z] = extraerPuntos(azim, elev, center, tam);
    xfin = [X(end); Y(end); Z(end)];
end
cVar = (0.05*range(im(:)))^2;
condicion = true;
if(debug)
    slice(double(permute(im,[2 1 3])), center(1), center(2), center(3));
    colormap('gray');
    shading flat
    hold on
    view(0,90)
    axis([1 121 1 145 1 121]);
    daspect([1 1 1]);
end
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


function indList = isAinB(A,B)
%Checks if the rows of A are in B. 
indList = ismember(A',B','rows','R2012a')';
end


function [subset,c] = neighbourSet(center, radius, im)
% This subfunction is used to compute the L2-norm support ball 
% with radius RADIUS. 
x=-radius:radius;
y=-radius:radius;
z=-radius:radius;
[xx,yy,zz] = meshgrid(x,y,z);
mask = (sum([xx(:),yy(:),zz(:)].^2, 2)<=radius^2)';
subset = [xx(mask)+center(1); yy(mask)+center(2); zz(mask)+center(3)];
tam=size(im); 
subset = subset(:, (subset(1,:)<=tam(1))&(subset(2,:)<=tam(2))&(subset(3,:)<=tam(3))&(subset(1,:)>0)&(subset(2,:)>0)&(subset(3,:)>0));
c = im(sub2ind(size(im),subset(1,:), subset(2,:), subset(3,:)));
end

function [X, Y, Z] = extraerPuntos(azim, elev, cent, tam)
% This subfunction extract the points that are contained in the 
% mapping vector used in SBM. 
    radia = 1:max(tam);
    [X,Y,Z]=sph2cart(deg2rad(azim), deg2rad(elev), radia);
    X = ceil(X+cent(1));
    Y = ceil(Y+cent(2));
    Z = ceil(Z+cent(3));
    selection = X > tam(1) | Y > tam(2) | Z > tam(3) | X < 1 | Y < 1 | Z < 1;
    X = X(~selection);
    Y = Y(~selection);
    Z = Z(~selection);
end
