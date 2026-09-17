function make_my_video(delta_vec_list_mid, delta_vec_list_ant, ...
                       f_mag_list_mid, f_mag_list_ant,videoFilename)
% Animate the two bands, with positive forces in red and negative forces in blue.
% Position arrays are [steps x sites x 3]; force arrays are [steps x sites].
% Saves an MPEG-4 file at 24 fps; playback is not scaled to simulation time.

    skipFrames   = 1;                % increase to render fewer frames
    fps          = 24;               % frames per second in final video
    
    N_step = size(delta_vec_list_mid, 1);
    if size(delta_vec_list_ant, 1) ~= N_step
        error('mid and ant must have the same N_step.');
    end

    allX = [delta_vec_list_mid(:,:,1), delta_vec_list_ant(:,:,1)];
    allY = [delta_vec_list_mid(:,:,2), delta_vec_list_ant(:,:,2)];
    allZ = [delta_vec_list_mid(:,:,3), delta_vec_list_ant(:,:,3)];
    
    xMin = min(allX(:));  xMax = max(allX(:));
    yMin = min(allY(:));  yMax = max(allY(:));
    zMin = min(allZ(:));  zMax = max(allZ(:));

    dx = 0.1*(xMax - xMin + eps);
    dy = 0.1*(yMax - yMin + eps);
    dz = 0.1*(zMax - zMin + eps);

    allForces = [f_mag_list_mid(:); f_mag_list_ant(:)];
    fMin = min(allForces);
    fMax = max(allForces);
    if fMin >= 0
        % All forces are nonnegative? We'll treat negative range as zero
        fMin = 0;
    end
    if fMax <= 0
        % All forces are nonpositive? We'll treat positive range as zero
        fMax = 0;
    end

    vWriter = VideoWriter(videoFilename, 'MPEG-4');
    vWriter.FrameRate = fps;
    open(vWriter);

    fig = figure('Color','w');
    ax  = axes('Parent', fig);
    hold(ax,'on');
    view(3)

    hMid = scatter3(ax, nan, nan, nan, 36, 'filled', ...
        'MarkerEdgeColor','none', 'DisplayName','mid');
    hAnt = scatter3(ax, nan, nan, nan, 36, 'filled', ...
        'MarkerEdgeColor','none', 'DisplayName','ant');
    
    axis([xMin-dx, xMax+dx, yMin-dy, yMax+dy, zMin-dz, zMax+dz]);
    daspect([1 1 1]);  % equal scaling
    grid on;
    xlabel('X'); ylabel('Y'); zlabel('Z');

    for i = 1:skipFrames:N_step

        xMid = delta_vec_list_mid(i,:,1);
        yMid = delta_vec_list_mid(i,:,2);
        zMid = delta_vec_list_mid(i,:,3);
        fMid = f_mag_list_mid(i,:);
        
        xAnt = delta_vec_list_ant(i,:,1);
        yAnt = delta_vec_list_ant(i,:,2);
        zAnt = delta_vec_list_ant(i,:,3);
        fAnt = f_mag_list_ant(i,:);

        cMid = forceToRGB(fMid, fMin, fMax);  % returns Nx3
        cAnt = forceToRGB(fAnt, fMin, fMax);  % returns Nx3

        set(hMid, 'XData', xMid, 'YData', yMid, 'ZData', zMid, 'CData', cMid);
        set(hAnt, 'XData', xAnt, 'YData', yAnt, 'ZData', zAnt, 'CData', cAnt);

        title(sprintf('Frame %d / %d', i, N_step));

        drawnow limitrate;
        frame = getframe(fig);
        writeVideo(vWriter, frame);
    end

    close(vWriter);
    close(fig);

    fprintf('Video saved to %s\n', videoFilename);

end

function C = forceToRGB(forceVals, fMin, fMax)
% Map signed forces to blue-white-red using a fixed range across all frames.

    N = numel(forceVals);
    C = zeros(N,3);

    % Avoid division by zero when only one force sign is present.
    denomNeg = abs(fMin);
    if denomNeg < 1e-12, denomNeg = 1e-12; end
    denomPos = fMax;
    if denomPos < 1e-12, denomPos = 1e-12; end

    whiteRGB = [1,1,1];
    blueRGB  = [0,0,1];
    redRGB   = [1,0,0];

    for n = 1:N
        f = forceVals(n);

        if f > 0
            alpha = min(max(f / denomPos, 0), 1);  
            C(n,:) = (1-alpha)*whiteRGB + alpha*redRGB;

        elseif f < 0
            alpha = min(max(abs(f) / denomNeg, 0), 1);
            C(n,:) = (1-alpha)*whiteRGB + alpha*blueRGB;

        else
            C(n,:) = whiteRGB;
        end
    end
end
