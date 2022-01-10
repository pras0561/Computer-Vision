%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EEN020 - Computer Vision 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: ComputeReprojectionError
%
% Computes the reprojection error for the camera P, 3D points U and 2D
% image points u. The value of each squared residual is in res.
% Note: infinite 2D points are interpreted as missing, and corresponding
% reprojection errors will be excluded.
%
%   inputs:    P: (3, 4) matrix
%                 camera matrix
%              U: (4, n_pts) matrix
%                 projective 3D points (homogeneous coordinates)
%              u: (2, n_pts) or (3, n_pts) matrix
%                 2D image points
%                 (homogeneous coordinates also work if last element is 1)
%
%   outputs:    err: double
%                    sum of squared errors for all residuals
%               res: (1, n_total_visible_pts) array
%                    squared residuals
%                    Note: if all points are visible, then
%                          n_total_visible_pts = n_pts
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [err,res] = ComputeReprojectionError(P,U,u)
err = 0;
res = [];
for i = 1:length(P);
    uu = u{i};
    vis = isfinite(uu(1,:));
    err = err + ...
        sum(((P{i}(1,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(1,vis)).^2) + ...
        sum(((P{i}(2,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(2,vis)).^2);
    res = [res ((P{i}(1,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(1,vis)).^2 + ...
            ((P{i}(2,:)*U(:,vis))./(P{i}(3,:)*U(:,vis)) - uu(2,vis)).^2];
end
