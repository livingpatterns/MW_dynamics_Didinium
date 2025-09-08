%% starup
set(groot, 'defaultTextInterpreter',   'latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(0,'DefaultFigureColormap',viridis);

%% define variables 

%cell parameters
U=1300; %swimming velocity [um/s]
omega_cell = 3.8; %cell spin rate [rad/s]
mu = 1e-9; %viscosity of water [um^2/s]

omega_ant=17; %anterior ring angular velocity [rad/s]
omega_mid=26; %medial ring angular velocity [rad/s]
a_mid = 54; %anterior ring radius 
a_ant = 34.7; %medial ring radius
lambda_mid = 28; %anterior ring mw wavelength
lambda_ant = 25; %medial ring mw wavelength

omega = mean([omega_ant,omega_mid]); %average mw angular velocity
a = mean([a_mid,a_ant]); %average radius
lambda = mean([lambda_mid,lambda_ant]); %average mw wavelength
Theta = atan((4*omega_cell*a)/(3*U)); %angle at which force is applied
F = U*6*pi*mu*a*cos(Theta); %total force exerted by the celia to balance drag

%simulation parameters 
cilia_density=1; % # of point forces per micron
dt=(a/U)/100; %simulation time step
%% sample trajectory fig 2
T=1; %total simulation time
data = table();
figure()

subplot (2,2,1) %omega = 20, k = 0.3 (lambda = 21)
pos_20_03 = straight_meta_fn(U,0,20,20,a,...
    a,mu,dt,21,21,cilia_density,T);
pretty_helix_plot(pos_20_03,0)
title("$\omega=$"+20+", k="+0.3)

%plot(pos(:,1),pos(:,3),'LineWidth',5)

subplot (2,2,2) %omega = 20, k = 0.05 (lambda = 125)
pos_20_005 = straight_meta_fn(U,0,20,20,a,...
    a,mu,dt,125,125,cilia_density,T);
pretty_helix_plot(pos_20_005,0)
title("$\omega=$"+20+", k="+0.05)
%plot(pos(:,1),pos(:,3),'LineWidth',5)

subplot (2,2,3) %omega = 2, k = 0.3 (lambda = 21)
pos_2_03 = straight_meta_fn(U,0,2,2,a,...
    a,mu,dt,21,21,cilia_density,T);
pretty_helix_plot(pos_2_03,0)
title("$\omega=$"+2+", k="+0.3)

%plot(pos(:,1),pos(:,3),'LineWidth',5)

subplot (2,2,4) %omega = 2, k = 0.05 (lambda = 125)
pos_2_005 = straight_meta_fn(U,0,2,2,a,...
    a,mu,dt,175,175,cilia_density,T);
pretty_helix_plot(pos_2_005,0)
title("$\omega=$"+2+", k="+0.05)

%plot(pos(:,1),pos(:,3),'LineWidth',5)

T = table(pos_20_03, pos_20_005, pos_2_03, pos_2_005, ...
          'VariableNames', {'pos_20_03','pos_20_005','pos_2_03','pos_2_005'});
writetable(T,'sample_trajectory.csv')
%% sample trajetories for different omega_mw and lambda_mw

T=1; %total simulation time

figure()
hold on

%large omega, small k
pos = straight_meta_fn(10,0,10,10,1,...
    1,1,dt,0.1,0.1,cilia_density,T);

plot(pos(:,1)+10,pos(:,3),'LineWidth',5)

%large omega, large k
pos = straight_meta_fn(10,0,10,10,1,...
    1,1,dt,1.9*pi,1.9*pi,cilia_density,T);

plot(pos(:,1)+10,pos(:,3)+12,'LineWidth',5)

%small omega, large k
pos = straight_meta_fn(10,0,0.1,0.1,1,...
    1,1,dt,1.9*pi,1.9*pi,cilia_density,1);

plot(pos(:,1),pos(:,3)+12,'LineWidth',5)

%small omega, small k
pos = straight_meta_fn(10,0,0.1,0.1,1,...
    1,1,dt,0.1,0.1,cilia_density,T);

plot(pos(:,1),pos(:,3),'LineWidth',5)


exportgraphics(gcf,'sample_trajectories.png','Resolution',600)

%% pitch and helical radius as a function of lever arm displacement 

loop_N = 1000; 

R0_list = linspace(0,a,loop_N);
omega_list = [3.5,14,28];
r_mat = zeros(length(R0_list),length(omega_list));
P_mat = r_mat;
for ii = 1:length(omega_list) %loop over omega
    for jj =1:length(R0_list) %loop over R0
    A = F*R0_list(jj)./(8*pi*mu*a^3);
    r_mat(jj,ii) = (U*A/(omega_list(ii).^2+A^2));
    P_mat(jj,ii) = ((2*pi*U*omega_list(ii))./(omega_list(ii).^2+A^2));
    end
end

data = readtable('all_tracks_helix_measurements.csv'); %read experiment data
exp_helix_radius = data.radius*1000; %convert length unit to micron
exp_helix_angle = data.helix_angle_degrees_;

figure
hold on
helix_angle = rad2deg(atan(P_mat./(r_mat*2*pi)));
c = [R0_list,flip(R0_list)]./a;
patch([r_mat(:,1);flip(r_mat(:,3))],[helix_angle(:,1);flip(helix_angle(:,3))],c,'FaceAlpha',.2,'LineStyle','none');

x = r_mat(:,1);
y = helix_angle(:,1);
y(end)=NaN;
c = R0_list./a;

patch(x,y,c,'EdgeColor','interp','marker','o','MarkerFaceColor','flat')
plot(exp_helix_radius,exp_helix_angle,'k+','markerSize',8)

cb = colorbar;
cb.Label.String = '$\chi$';
cb.Label.Interpreter = 'latex';

xlabel('radius ($\mu m$)')
ylabel('helix angle (degrees) ')
set(gca,'FontSize',18)
exportgraphics(gca,'radius_pitch_scatter.png','Resolution',600)

%save simulated data
writematrix(helix_angle,'helix_angle.csv')
writematrix(r_mat,'helix_radius.csv')

%figure showing the percentage of data within simulation bounds 
figure
poly = polyshape([r_mat(:,1);flip(r_mat(:,3))],[helix_angle(:,1);flip(helix_angle(:,3))]);
[x,y] = boundary(poly);
inpoly = inpolygon(exp_helix_radius,exp_helix_angle,x,y);
in_percent = sum(inpoly)/numel(exp_helix_radius);
plot(poly)
hold on
plot(exp_helix_radius,exp_helix_angle,'k+')
plot(exp_helix_radius(inpoly),exp_helix_angle(inpoly),'r+')
exportgraphics(gca,'data_in_range.png','Resolution',600)

%% calculate distribution of estimated omega and chi
data = readtable('all_tracks_helix_measurements.csv'); %experiment data

r_data = data.radius*1000; %convert length unit to microns 
pitch_data = data.pitch*1000;

figure
plot(r_data,pitch_data,'+')

A_data = 4*pi^2*U.*r_data./(pitch_data.^2+4*pi^2.*r_data.^2);
chi_data = A_data/22;
omega_data = 2*pi*U.*pitch_data./(pitch_data.^2+4*pi^2.*r_data.^2);

figure
histogram(chi_data);
xline(mean(chi_data),'--','LineWidth',3)
xlabel('estimated $\chi$')
ylabel('counts')
set(gca,'FontSize',18)

figure
histogram(omega_data);
xline(mean(omega_data),'--','LineWidth',3)
xlabel('estimated $\omega$')
ylabel('counts')
set(gca,'FontSize',18)

figure
plot(omega_data,chi_data,'+')
xlabel('$\omega$')
ylabel('$\chi$')
set(gca,'FontSize',18)

%% helix due to only non-integer number of nodes in metachronal wave

%N = (5:20)+0.5;
%loop_N = numel(N);
loopN = 100;
loop_omega = 100;
% lambda_list = 2*pi*a./N;
lambda_list = linspace(0.5*lambda,1.5*lambda,loop_N);

omega_list = linspace(0.5*omega,1.5*omega,loop_omega);
n=100;
angles = linspace(0,2*pi,n)';
r_mat = zeros(loop_omega,loop_N);
P_mat = zeros(loop_omega,loop_N);
delta_vec = [cos(angles),sin(angles)];

R0_list = zeros(loop_omega,loop_N);
for ii = 1:loop_omega %loop over omega
    for jj =1:loop_N %loop over lambda
        omega_temp = omega_list(ii);
        lambda_temp = lambda_list(jj);
        f_mag = F.*(sin((2*pi*a/lambda_temp).*(angles))+1)./(n);
        R0 = a.*(sum(f_mag.*delta_vec,1))/F;
        R0 = norm(R0);
    A = F*R0./(8*pi*mu*a^3*omega_temp);
    r_mat(ii,jj) = (U*A/(omega_temp.^2+A^2));
    P_mat(ii,jj) = ((2*pi*U*omega_temp)./(omega_temp.^2+A^2));
    R0_list(ii,jj) = R0;
    end
end

% figure
% plot(N,R0_list(1,:)./a,'.-')
% xlabel('number of nodes')
% ylabel('$\chi$')
% set(gca,'FontSize',20)
% exportgraphics(gca,'N_vs_chi.png','Resolution',600)
% data = table;
% data.N =N;
% data.chi = R0_list(1,:)./a;
% writetable(data, 'N_vs_chi.csv')

figure %heat map of helix radius as a function of omega and k
h1 = surf((2*pi)./lambda_list, omega_list, r_mat./a);
shading interp 
view(2)
xlim('tight')
ylim('tight')
colormap(viridis)                     
cb = colorbar;
cb.Label.String = 'helix radius ($\mu m$)';
cb.Label.Interpreter = 'latex';
cb.FontSize = 20;
xlabel('$k$')
ylabel('$\omega$')
set(gca,'FontSize',20)
ax = gca;
ax.Units = 'normalized';
ax.Position = [0.15 0.15 0.5 0.75];   
exportgraphics(gca,'r_helix_heatmap.png','Resolution',600)


figure %heat map of helix angle as a function of omega and k
helix_angle_mat = rad2deg(atan(P_mat./(r_mat*2*pi)));
h1 = surf((2*pi)./lambda_list, omega_list, helix_angle_mat);
shading interp 
view(2)
xlim('tight')
ylim('tight')
colormap(viridis)                     
cb = colorbar;
cb.Label.String = 'helix angle (deg)';
cb.Label.Interpreter = 'latex';
cb.FontSize = 20;
xlabel('$k$')
ylabel('$\omega$')
set(gca,'FontSize',20)
ax = gca;
ax.Units = 'normalized';
ax.Position = [0.15 0.15 0.5 0.75];  
exportgraphics(gca,'helix_angle_heatmap.png','Resolution',600)


% 
% figure
% h1 = surf((2*pi*a)./lambda_list, omega_list, R0_list./a);    
% shading interp 
% view(2)
% xlim('tight')
% ylim('tight')
% 
% colormap(viridis)                     
% cb = colorbar;
% cb.Label.String = '$\chi$';
% cb.Label.Interpreter = 'latex';
% 
% 
% xlabel('N')
% ylabel('$\omega$')
% xticks(2:2:25)
% set(gca,'TickDir','out')
% set(gca,'FontSize',20)
% exportgraphics(gca,'r_helix_heatmap.png','Resolution',600)

%% deflection angle due to rapid reversal 

start_time = 0.01;
dt = 0.0001;

n=10;
reversal_time_list=linspace(0.01,1,n);
lag_time_mat = zeros(n,n);
theta_list = zeros(n,n);
figure
hold on
view(3)
for ii = 1:length(reversal_time_list)
    reversal_time = reversal_time_list(ii);
    lag_time_list = linspace(0,1,n);
    lag_time_mat(ii,:) = lag_time_list;
    for jj = 1:n
        lag_time = lag_time_list(jj); 
        [pos,p_hat_list] =  reversal_w_spin_lag_cont_fn(U,omega_cell,omega_mid,omega_ant,reversal_time,lag_time,a_mid,a_ant,mu,start_time,dt,lambda_mid,lambda_ant,0);
        plot3(pos(:,1),pos(:,2),pos(:,3),'DisplayName',"\tau_{reverse}="+reversal_time)
        idx1 = round(start_time/dt);
        idx2 = round((start_time+reversal_time+lag_time)/dt);
        px = p_hat_list(:,1);
        py = p_hat_list(:,2);
        pz = p_hat_list(:,3);
        theta_rad = acos(pz);
        phi_rad   = atan2(py, px);

        theta_list(ii,jj) = mean(rad2deg(theta_rad(idx2:end)));
    end
end
%% heat map of deflection angle as a function of reversal times and lag time

figure
surf(repmat(reversal_time_list',[1,n]),lag_time_mat,theta_list)
 shading interp
 view(2)
 c=colorbar;
 c.Label.String = 'deflection (degrees)';
c.Label.Interpreter = 'latex';

ylabel('$t_{lag} (s) $')
xlabel('$t_{reverse} (s)$')
set(gca,'FontSize',20)
xlim([0.02,1])
ylim([0,1])
exportgraphics(gca,'deflection_w_lag.png','Resolution',600)

%%
n=40;
start_time = 0.01;

reversal_time_list=linspace(0.01,0.3,n);
theta_list_linear = zeros(1,n);
theta_list_desync = zeros(1,n);
theta_list_step = zeros(1,n);
omega1 = omega_mid;
omega2 = omega_ant;
for jj = 1:n
        reversal_time = reversal_time_list(jj);
        [~,p_hat_list] =  reversal_w_spin_lag_cont_fn(U,omega_cell,omega1,omega2,reversal_time,0,a_mid,a_ant,mu,start_time,dt,lambda_mid,lambda_ant,0);
        idx1 = round(start_time/dt);
        idx2 = round((start_time+reversal_time)/dt);
        pz = p_hat_list(:,3);
        theta_rad = acos(pz);

        theta_list_linear (jj) = mean(rad2deg(theta_rad(idx2:end)));

        [~,p_hat_list] =  reversal_w_spin_desync_fn(U,omega_cell,omega1,omega2,reversal_time,a_mid,a_ant,mu,start_time,dt,lambda_mid,lambda_ant,0);
        pz = p_hat_list(:,3);
        theta_rad = acos(pz);

        theta_list_desync (jj) = mean(rad2deg(theta_rad(idx2:end)));

        [~,p_hat_list] =  reversal_w_spin_step_fn(U,omega_cell,omega1,omega2,reversal_time,a_mid,a_ant,mu,start_time,dt,lambda_mid,lambda_ant,0);
        pz = p_hat_list(:,3);
        theta_rad = acos(pz);

        theta_list_step (jj) = mean(rad2deg(theta_rad(idx2:end)));
end

figure
hold on
plot(reversal_time_list,theta_list_linear,'LineWidth',2)
plot(reversal_time_list,theta_list_desync,'LineWidth',2)
plot(reversal_time_list,theta_list_step,'LineWidth',2)
legend('linear','asynchronous','Heaviside step','Location','northwest')
xlabel('$t_{reverse}$')
ylabel('deflection angle (degrees)')

set(gca,'FontSize',20)
exportgraphics(gca,'deflection_angle_models.png','Resolution',600)
%% make kymograph of reversal 

reversal_w_spin_step_fn(U,omega_cell,omega1,omega2,0.05,a_mid,a_ant,mu,start_time,dt,lambda_mid,lambda_ant);
xlabel('$dx (\mu m)$')
ylabel('Time (s)')
set(gca,'FontSize',20)

exportgraphics(gca,'kymo_step.png','Resolution',600)
%% Gaussian noise in node location 

N = floor((2*pi*a)./lambda);
sigma_upper_limit = pi/N;
sigma = linspace(0,sigma_upper_limit);
A = (3*U/(4.*a)).*sigma.*sqrt(pi/(4*N));

r_helix = U.*A./(omega^2+A.^2);
P_helix = 2*pi*U*omega./(omega^2+A.^2);
angle_helix = rad2deg(atan(P_helix./(2*pi.*r_helix)));
figure
hold on
yyaxis left
plot(sigma,r_helix,'-','HandleVisibility','off')
xlabel('$\sigma$ (rad)')
ylabel('helix radius $(\mu m)$')

yyaxis right
plot(sigma,angle_helix,'-','HandleVisibility','off')
ylabel('helix angle (deg)')
set(gca,'FontSize',20)

% figure
% plot(sigma,sqrt(pi.*sigma.^2./(2*N)))
% 

%%
T=1;
n_sample = 10;
n_pts = 20;
sim_data_radius = zeros(n_pts,n_sample);
sim_data_angle = zeros(n_pts,n_sample);
sigma_sample = linspace(0,sigma_upper_limit,n_sample);
for jj = 1:n_sample
for ii = 1:n_pts
    theta = 2*pi*(0:(N-1))/N;
    eps = sigma_sample(jj)*randn(1,N);
    x = cos(theta + eps);
    y = sin(theta + eps);
    R0 = sqrt(mean(x)^2+mean(y)^2); 
    [~,r,p] = straight_meta_fn_sp(U,omega,a,mu,dt,R0,T);

    % [r,p] = straight_meta_noise_fn(U,omega_cell,omega,a,...
    % mu,dt,lambda,cilia_density,T,sigma_sample(jj));
    sim_data_radius(ii,jj) = r;
    sim_data_angle(ii,jj)=rad2deg(atan(p/(2*pi*r)));
end
end


yyaxis left
plot(sigma_sample,mean(sim_data_radius,1),'--','HandleVisibility','off')
yyaxis right
plot(sigma_sample,mean(sim_data_angle,1),'--','HandleVisibility','off')
plot(NaN,NaN,'k-','DisplayName','theory')
plot(NaN,NaN,'k--','DisplayName','simulation')
legend show
exportgraphics(gca,'gaussian_noise.png','Resolution',600)
%%
node_gaussian_noise_theory = table;
node_gaussian_noise_theory.sigma_theory = sigma';
node_gaussian_noise_theory.r_thoery = r_helix';
node_gaussian_noise_theory.p_theory = P_helix';
writetable(node_gaussian_noise_theory, 'node_gaussian_noise_theory.csv')

node_gaussian_noise_sim = table;
node_gaussian_noise_sim.sigma_sim = sigma_sample';
node_gaussian_noise_sim.r_sim = mean(sim_data_radius,1)';
node_gaussian_noise_sim.p_sim = mean(sim_data_angle,1)';
writetable(node_gaussian_noise_sim, 'node_gaussian_noise_sim.csv')

%% example trajectories 

chi_list = [0.1,0.32,0.545];
omega_list = [11.5,11.6,8.63];

track1 = zeros(100,3);
track2 = zeros(100,3);
track3 = zeros(100,3);

figure
pos = straight_meta_fn_sp(U,omega_list(1),a,mu,dt,chi_list(1),5);
track1(1:length(pos),:) = pos;

pretty_helix_plot(pos,0)
writematrix(pos,'track1.csv')


figure
pos = straight_meta_fn_sp(U,omega_list(2),a,mu,dt,chi_list(2),5);
track1(1:length(pos),:) = pos;

pretty_helix_plot(pos,0)
writematrix(pos,'track2.csv')


figure
pos = straight_meta_fn_sp(U,omega_list(3),a,mu,dt,chi_list(3),5);
track1(1:length(pos),:) = pos;

pretty_helix_plot(pos,0)
writematrix(pos,'track3.csv')


function pretty_helix_plot(pos,y_disp)


pos_c  = pos - mean(pos,1);           % translate to the centroid
[coeff,~,~] = pca(pos_c);             % principal-component axes
v_axis = coeff(:,1);                  % direction with largest variance

% make sure the axis points roughly in +x 
if v_axis(1) < 0
    v_axis = -v_axis;
end
ex      = [1;0;0];
k       = cross(v_axis,ex);           % rotation axis
k_norm  = norm(k);

if k_norm < 1e-12                    
    R = eye(3);
else
    k       = k/k_norm;               
    theta   = acos(dot(v_axis,ex));   
    K       = [  0   -k(3)  k(2);
               k(3)    0   -k(1);
              -k(2)  k(1)    0 ];
    R = eye(3) + sin(theta)*K + (1-cos(theta))*K*K;   % Rodrigues
end
pos_rot = (R * pos_c.').';            

N  = size(pos_rot,1);
t  = linspace(0,1,N).';               

scatter3(pos_rot(:,1),pos_rot(:,2),pos_rot(:,3)+y_disp, ...
         25, t, 'filled');            
colormap(turbo);
grid on
axis equal;
set(gca,'ztick',[])
view(3);
fontsize(gca,18,"points")
axis off;
end


