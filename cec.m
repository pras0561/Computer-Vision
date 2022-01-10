%% Task c
load("data1.mat")

P_all = cell(9, 7);
P_update = cell(9, 7);

b_boxes = cell(1, 9);
poses_true = cell(1, 9);

for data_ind = 1:9
    data = load(strcat('data', int2str(data_ind), '.mat'));
    poses_true{data_ind} = data.poses;
    b_boxes{data_ind} = data.bounding_boxes;

    for ind = 1:length(data.U)
        total_pts = length(data.U{ind});
        X = [data.U{ind}; ones(1, total_pts)];
        x = [data.u{ind}; ones(1, total_pts)];
    
        rand_ind = randsample(1: total_pts, 3);
        P_all{data_ind, ind}  = minimalCameraPose(x(:, rand_ind), data.U{ind}(:, rand_ind));

        [err_p1, ~] = ComputeReprojectionError1(P_all{data_ind, ind}{1}, X, x);
        [err_p2, ~] = ComputeReprojectionError1(P_all{data_ind, ind}{2}, X, x);
        if (err_p1 > err_p2)
            P_update{data_ind, ind} = P_all{data_ind, ind}{2};
        else
            P_update{data_ind, ind} = P_all{data_ind, ind}{1};
        end
    end
end

err_scores = evaluate_plot_hist(poses_true, P_update, b_boxes);
mean(err_scores)


function err_scores = evaluate_plot_hist(poses_true, P_update, b_boxes)
    err_scores = zeros(1, 9 * 7);
    for i = 1:9
        scores = eval_pose_estimates(poses_true{i}, P_update(i, :), b_boxes{i});
        start_ind = ((i - 1) * 7) + 1;
        err_scores(1, start_ind: (start_ind + 6)) = cell2mat(scores);
    end

    figure()
    histogram(err_scores, 20);
end
