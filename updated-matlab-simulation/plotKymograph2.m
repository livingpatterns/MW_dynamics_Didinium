function plotKymograph2(delta_vec_list_mid, forceMatrix,t_axis,a)
% Plot force magnitude by site and time; reversal signs are discarded.
% Position history sets the array size; t_axis supplies time labels.
% The x labels approximate arc length at the default density of one site per um.
    forceMatrix = abs(forceMatrix);
    [N_step, n_mid, ~] = size(delta_vec_list_mid);
    
    dataKymo = zeros(N_step, n_mid);

    for i = 1:N_step
        for j = 1:n_mid
            dataKymo(i,j) = forceMatrix(i,j);
        end

    end

    figure;
    imagesc(dataKymo);
    axis xy;  % time increases upward

    yticks(1:N_step);         
    nTicks = 10;
    tickIdx = round(linspace(1, N_step, nTicks));
    tickIdx_x = 0:50:2*pi*a;

    yticks(tickIdx);
    xticks(tickIdx_x);
    yticklabels(string(t_axis(tickIdx)));
    xticklabels(string(tickIdx_x));
    ylabel('Time (s)');   
    xlabel('dx (um)');
    colormap('winter');
   end
