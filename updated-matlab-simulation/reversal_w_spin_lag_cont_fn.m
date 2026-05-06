
function [pos,p_hat_list] = reversal_w_spin_lag_cont_fn(U,omega_cell,omega1,omega2,reversal_time,lag_time,a_mid,a_ant,mu,start_time,dt,lambda_mid,lambda_ant,video,beta_mid,beta_ant,gamma_mid,gamma_ant)

omega_mw_mid_mag = omega1;
omega_mw_ant_mag = omega2;

if nargin < 15 || isempty(beta_mid),  beta_mid  = 1; end
if nargin < 16 || isempty(beta_ant),  beta_ant  = beta_mid; end
if nargin < 17 || isempty(gamma_mid), gamma_mid = 1; end
if nargin < 18 || isempty(gamma_ant), gamma_ant = gamma_mid; end

%n_mid = floor((2*pi*a_mid)./lambda_mid);
%n_ant = floor((2*pi*a_ant)./lambda_ant);
cilia_density = 1;
n_mid = floor(2*pi*a_mid*cilia_density);
n_ant = floor(2*pi*a_ant*cilia_density);


T=start_time+reversal_time+lag_time+start_time;

R_body = a_mid;              % cell-body radius used in mobility denominators
varrho_mid = a_mid;          % radial distance of medial ring from center
h=R_body;
beta_weight = (beta_mid*n_mid + beta_ant*n_ant)/(n_mid+n_ant);
F = U*6*pi*mu*R_body/beta_weight;
angles_mid = (0:(2*pi/(n_mid)):(2*pi-2*pi/(n_mid)))';
angles_ant = (0:(2*pi/(n_ant)):(2*pi-2*pi/(n_ant)))';

omega_mw_mid = omega_mw_mid_mag.*ones([n_mid,1]); 
omega_mw_ant = omega_mw_ant_mag.*ones([n_ant,1]);
% f_mag_mid = F*ones(n_mid,1)./(n_mid+n_ant);
% f_mag_ant = F*ones(n_ant,1)./(n_mid+n_ant);
    
    f_mag_mid = F.*(sin((2*pi*a_mid/lambda_mid).*(angles_mid))+1)./(n_mid+n_ant);
    f_mag_ant = F.*(sin((2*pi*a_ant/lambda_ant).*(angles_ant))+1)./(n_mid+n_ant);
t=0; 
N_step = floor(T/dt);
p_hat =[0,0,1];
x_vec = [0,0,0];

v = [0; 0; 1];
if abs(dot(p_hat,v)) > 0.99
    v = [1; 0; 0];
end
e1 = cross(p_hat, v);
e1 = e1 / norm(e1);
e2 = cross(p_hat, e1);
e2 = e2 / norm(e2); 
delta_vec_mid = zeros(n_mid, 3);
delta_vec_ant = zeros(n_ant, 3);
angle_ant = atan(a_ant/h);
varrho_ant = sqrt(a_ant.^2+h.^2);
for i = 1:floor(n_mid)
    delta_vec_mid(i,:) = cos(angles_mid(i))*e1 + sin(angles_mid(i))*e2 ;
end

for i = 1:floor(n_ant)
    delta_vec_ant(i,:) = sin(angle_ant).*cos(angles_ant(i))*e1 + sin(angle_ant).*sin(angles_ant(i))*e2 +cos(angle_ant).*p_hat;
end

idx1 = round(start_time/dt);
idx2 = round((start_time+lag_time)/dt);

mask_mid = false(n_mid,1);
mask_ant = false(n_ant,1);

pos = zeros(N_step,3);
delta_vec_list_mid = zeros(N_step,n_mid,3);
delta_vec_list_ant = zeros(N_step,n_ant,3);

p_hat_list = zeros(N_step,3);
net_torque_list = zeros(N_step,3);
v0_list = zeros(N_step,1);
f_mag_list_mid = zeros(N_step,n_mid);
f_mag_list_ant = zeros(N_step,n_ant);

mask_list_mid = zeros(N_step,n_mid);
mask_list_ant = zeros(N_step,n_ant);

for ii = 1:N_step+1

    v0 = (beta_mid*sum(f_mag_mid) + beta_ant*sum(f_mag_ant))/(6*pi*mu*R_body);

    w_para_mid = (omega_cell*(v0/U)).*p_hat;
    w_para_ant = (omega_cell*(v0/U)).*p_hat;

    torque_mid = sum(cross(varrho_mid.*delta_vec_mid,f_mag_mid.*repmat(p_hat,[n_mid,1])),1);
    torque_ant = sum(cross(varrho_ant.*delta_vec_ant,f_mag_ant.*repmat(p_hat,[n_ant,1])),1);
    
    w_perp = (gamma_mid*torque_mid + gamma_ant*torque_ant)./(8*pi*mu*R_body^3);
    
    w0 =  w_perp;
    w0_norm = norm(w0);
    n_hat = p_hat;
    if w0_norm~=0
    n_hat = w0./w0_norm;
    end
    w0_mid = w_para_mid + w_perp;
    w0_mid_norm = sqrt(sum(w0_mid.^2));
    n_hat_mid = p_hat;
    if w0_mid_norm~=0
    n_hat_mid = w0_mid./w0_mid_norm;
    end

    w0_ant = w_para_ant + w_perp;
    w0_ant_norm = sqrt(sum(w0_ant.^2,2));
    n_hat_ant = p_hat;
    if w0_ant_norm~=0
    n_hat_ant = w0_ant./w0_ant_norm;
    end
    
    theta = norm(w0).*dt;
    theta_mid = w0_mid_norm.*dt;
    theta_ant = w0_ant_norm.*dt;

    p_hat = p_hat.*cos(theta)+cross(n_hat,p_hat).*sin(theta)+n_hat.*dot(n_hat,p_hat).*(1-cos(theta));
    p_hat = p_hat/norm(p_hat);

    delta_vec_mid = delta_vec_mid.*cos(theta_mid)+cross(repmat(n_hat_mid,[n_mid,1]),delta_vec_mid).*sin(theta_mid)+n_hat_mid.*sum(n_hat_mid.*delta_vec_mid,2).*(1-cos(theta_mid));
    delta_vec_ant = delta_vec_ant.*cos(theta_ant)+cross(repmat(n_hat_ant,[n_ant,1]),delta_vec_ant).*sin(theta_ant)+n_hat_ant.*sum(n_hat_ant.*delta_vec_ant,2).*(1-cos(theta_ant));

    x_vec = x_vec + v0.*p_hat.*dt;
    
    f_mag_mid = F.*(sin((2*pi*a_mid/lambda_mid).*(angles_mid-omega_mw_mid*t))+1)./(n_mid+n_ant);
    f_mag_ant = F.*(sin((2*pi*a_ant/lambda_ant).*(angles_ant-omega_mw_ant*t))+1)./(n_mid+n_ant);

    % if ii==idx1
    %     f_mag_mid=zeros(n_mid,1);
    %     f_mag_ant=zeros(n_ant,1);
    % end
    if ii>= idx1
    mask_mid = ((2*pi/(reversal_time))*(t-start_time))>angles_mid;
    end
    if ii>= idx2
    mask_ant = ((2*pi/(reversal_time))*(t-start_time-lag_time))>angles_ant;
    end
    % 
    % if ii>=idx1 && ii<=idx2
    %     % mask_mid = angles_mid>pi;
    %     % mask_ant = angles_ant>pi;
    %     % mask_mid = ((2*pi/(reversal_time))*(t-start_time))>angles_mid;
    %     % mask_ant = ((2*pi/(reversal_time))*(t-start_time))>angles_ant;
    %     counter_mid = ceil((t-start_time)/(reversal_time/n_mid));
    %     mask_mid = logical([ones(counter_mid,1);zeros(n_mid-counter_mid,1)]);
    %     counter_ant = ceil((t-start_time)/(reversal_time/n_ant));
    %     mask_ant = logical([ones(counter_ant,1);zeros(n_ant-counter_ant,1)]);
    %     if abs(dot(p_hat,v)) > 0.99
    %         v = [1; 0; 0];
    %     end
    %     e1 = cross(p_hat, v);
    %     e1 = e1 / norm(e1);
    %     e2 = cross(p_hat, e1);
    %     e2 = e2 / norm(e2);
    %     if counter_mid>1
    %     delta_vec_mid(counter_mid,:) = cos(angles_mid(counter_mid))*e1 + sin(angles_mid(counter_mid))*e2 ;
    %     end
    %     if counter_ant >1
    %     delta_vec_ant(counter_ant,:) = sin(angle_ant).*cos(angles_ant(counter_ant))*e1 + sin(angle_ant).*sin(angles_ant(counter_ant))*e2 +cos(angle_ant).*p_hat;
    %     end
    % 
    % end
    % 
    f_mag_mid(mask_mid) = -abs(f_mag_mid(mask_mid));
    f_mag_ant(mask_ant) = -abs(f_mag_ant(mask_ant));
    omega_mw_mid(mask_mid) =  -omega_mw_mid_mag;
    omega_mw_ant(mask_ant) = -omega_mw_ant_mag;
    
    % f_mag_mid(mask_mid) = abs(f_mag_mid(mask_mid));
    % f_mag_ant(mask_ant) = abs(f_mag_ant(mask_ant));
    % omega_mw_mid(mask_mid) =  omega_mw_mid_mag;
    % omega_mw_ant(mask_ant) = omega_mw_ant_mag;
    
    t = t+ dt;

    pos(ii,:) = x_vec;
    % delta_vec_list_mid(ii,:,:)=x_vec + a_mid.*delta_vec_mid;
    % delta_vec_list_ant(ii,:,:)=x_vec + varrho_ant.*delta_vec_ant;
    delta_vec_list_mid(ii,:,:)= a_mid.*delta_vec_mid;
    delta_vec_list_ant(ii,:,:)= varrho_ant.*delta_vec_ant;

    p_hat_list(ii,:) = p_hat;
    net_torque_list(ii,:) = gamma_mid*torque_mid + gamma_ant*torque_ant;
    v0_list(ii) = v0;
    f_mag_list_mid(ii,:) = f_mag_mid;
    f_mag_list_ant(ii,:) = f_mag_ant;

    mask_list_mid(ii,:) = mask_mid;
    mask_list_ant(ii,:) = mask_ant;
end
if video
make_my_video(delta_vec_list_mid, delta_vec_list_ant, ...
                       f_mag_list_mid, f_mag_list_ant,'lag_cont.mp4')
end 

end


