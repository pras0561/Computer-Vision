%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EEN020 - Computer Vision 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: update_solution
%
% Computes updated camera Pnew from the old solution
% P and the parameter increment deltav. If camera is not calibrated, K
% will still be constant throughout the update step.
%
%   inputs:    deltav: (1, 6) array
%                      parameter increment
%              P: (3, 4) matrix
%                 previous estimate of the camera matrix
%
%   outputs:    Pnew: (3, 4) matrix
%                     new estimate of the camera matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Pnew,Unew] = update_solution(deltav,P,U)
%Computes a new solution Pnew,Unew, from the old solution P,U and the 
%variable deltav

Ba = [0 1 0; -1 0 0; 0 0 0];
Bb = [0 0 1; 0 0 0; -1 0 0];
Bc = [0 0 0; 0 0 1; 0 -1 0];

dpointvar = [0; deltav(1:(3*size(U,2)-1))];
dpointvar = reshape(dpointvar, size(U(1:3,:)));
dcamvar = [0;0;0;0;0;0;deltav(3*size(U,2):end)];
dcamvar = reshape(dcamvar,[6 length(P)]);

Unew = pextend(U(1:3,:) + dpointvar);

Pnew = cell(size(P));
for i=1:length(P);
    [K,Ri] = rq(P{i});
    R0 = Ri(:,1:3);
    t0 = Ri(:,4);
    R = expm(Ba*dcamvar(1,i) + Bb*dcamvar(2,i) + Bc*dcamvar(3,i))*R0;
    t = t0 + dcamvar(4:6,i);
    Pnew{i} = K*[R t];
end
