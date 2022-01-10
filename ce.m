%% TASK a
%% RANSAC 
clc;
clear all;
close all;
n_images = 9; % number of given images
for i = 1:n_images
    data_mat = sprintf('data%d.mat', i);
    imgi = sprintf('img%d.jpg',i);
    data{i} = load(data_mat);
    img{i} = imread(imgi);
end
num_object = 7;
epoch = 300;
threshold = 5e-3;
best_count = 0;
LM_epoch = 25;
lambda = 10e-15;

for i = 1:num_object
    for j = 1:n_images
        % Pass to every image and then takes image points of one object
        x = data{j}.u{i}; 
        img_points = length(x);
        x = [x; ones(1,img_points)];
        % normalize the 3D points
        % Pass to an img and takes the 3d points of one object
        X = data{j}.U{i};
        m = size(X,2);
        X_homoge = [X; ones(1,m)];
        X_mean = mean(X,2);
        normalized3Dpoint =  X - repmat(X_mean, [1 m]);
        
        % RANSAC
        best_count = 0;
        for k = 1:epoch
            randind = [];
            for k = 1:3
                randind = [randind fix(rand*m + 1)]; 
            end
            xs = x(:,randind);
            Xs = normalized3Dpoint(1:3,randind);
            % Minimal solver camera pose gives the calibrated solutions [R t]
            min_sol = minimalCameraPose(xs,Xs);
            num_sol = length(min_sol);
            num_inliers = [];
            % Checking which solution that has more inliers
            for l = 1:num_sol
                P{l} = min_sol{l};
                [K{l}, R{l}] = rq(P{l});
                
                x_proj{l} = pflat(P{l}*X_homoge);
                err{l} = (sum(x_proj{l} - x).^2).^0.5; 
                inliers{l} = err{l} <= threshold;
                num_inliers = [num_inliers sum(inliers{l})];
                [most_inliers, maxind] = max(num_inliers);
            end


            most_in = max(num_inliers);
            disp('image:');
            disp(j);
            disp('object:');
            disp(i);
            disp('Maximum number of inliers:');
            disp(most_in);

            % Saving best solution
            if most_inliers > best_count
                best_count = most_inliers;
                largest_X = X_homoge(:,inliers{maxind});
                % gather the outlier free correspondances
                largest_x = x(:,inliers{maxind}); 
                P_ransac =P{maxind};
            end
        end
           disp('outlier free correspondances:');
           disp(largest_x);
           disp('Pose estimation')
           disp(P_ransac);
        % Minimal solver solution on inliers only
        P_est{i,j} = P_ransac;
        Xinliers{i,j} = largest_X;
        xinliers{i,j} = largest_x;
        
    end
end
save('project_1a_data1.mat', 'P_est', 'Xinliers', 'xinliers');


% Ground Truth & Bounding boxes
for i = 1:num_object
    for j = 1:n_images
        P_gts{i,j} = data{j}.poses{i};
        bounding_boxes{i,j} = data{j}.bounding_boxes{i};
    end
end
%% TASK b)
%% Levenberg Marquardt

LevenbergMarquardt = cell(size(P_est));

for i = 1:num_object
    for j = 1:n_images
        for k = 1:LM_epoch
            P = {P_est{i,j}};
            U = Xinliers{i,j};
            u = {xinliers{i,j}};
            [err_LM(k), res] = ComputeReprojectionError(P,U,u);
            [r,J] = LinearizeReprojErr(P,U,u);
            C = J'*J + lambda*speye(size(J,2));
            c=J'*r;
            deltav = -C\c;
            [Pnew{i,j}] = update_solution(deltav,P,U);
            LevenbergMarquardt{i,j} = cell2mat(Pnew{i,j});
        end
        disp({i,j});
    end
end
      
%% Evaluation & Table of RANSAC and LevenbergMarquardt Scores

Ransac = cell(1,9);
scores_LevenbergMarquardt = cell(1,9);
for i=1:9
    disp("Image "+ i + ":");
 
    Ransac{i} = eval_pose_estimates(P_gts(:,i), P_est(:,i),bounding_boxes(:,i));
    scores_LevenbergMarquardt{i} = eval_pose_estimates(P_gts(:,i), LevenbergMarquardt(:,i),bounding_boxes(:,i));
end

Ransac_tab = zeros(num_object,n_images);
LevenbergMarquardt_tab = zeros(num_object,n_images);
for i = 1:n_images
    Ransac_tab(:,i) = cell2mat(Ransac{i})';
    LevenbergMarquardt_tab(:,i) = cell2mat(scores_LevenbergMarquardt{i})';
end

RANSAC_total_avg = sum(Ransac_tab(:))/numel(Ransac_tab)
LM_total_avg = sum(LevenbergMarquardt_tab(:))/numel(LevenbergMarquardt_tab)

Ransac_avg = mean(Ransac_tab,1);
LM_avg = mean(LevenbergMarquardt_tab,1);

Ransac_tab = round([Ransac_tab; Ransac_avg],1);
LevenbergMarquardt_tab = round([LevenbergMarquardt_tab; LM_avg],2);
Ransac = array2table(Ransac_tab,'VariableNames',{'Img1','Img2','Img3','Img4','Img5','Img6','Img7','Img8','Img9'},...
    'RowNames',{'Ape';'Can';'Cat';'Duck';'Eggbox';'Glue';'Holepuncher';'Average'});


LM = array2table(LevenbergMarquardt_tab,'VariableNames',{'Img1','Img2','Img3','Img4','Img5','Img6','Img7','Img8','Img9'},...
    'RowNames',{'Ape';'Can';'Cat';'Duck';'Eggbox';'Glue';'Holepuncher';'Average'});

Ransac

LM


%% Bounding Boxes for RANSAC + Minimal Solver

for i = 1:n_images
     draw_bounding_boxes(img{i},P_gts(:,i),P_est(:,i),bounding_boxes(:,i));
end
%% Bounding Boxes for RANSAC + Minimal Solver + LM

for i = 1:n_images
     draw_bounding_boxes(img{i},P_gts(:,i),LevenbergMarquardt(:,i),bounding_boxes(:,i));
end

%% Histogram for each method

figure(19);
subplot(2,2,1);
histogram(Ransac_tab,50);
xlabel('Evaluation Score');
ylabel('Number of Pose estimation');
title('RANSAC + Minimal Solver');

subplot(2,2,2);
histogram(LevenbergMarquardt_tab,50);
xlabel('Evaluation Score');
ylabel('Number of Pose estimation');
title('RANSAC + Minimal Solver + LM');


