function pos = straight_meta_fn(U,omega_cell,omega_mw_mid_mag,omega_mw_ant_mag,a_mid,...
    a_ant,mu,dt,lambda_mid,lambda_ant,cilia_density,T,beta_mid,beta_ant,gamma_mid,gamma_ant)
% Motion of a self–propelled sphere with propelled by two bands of point forces.
%
% U                 swim speed  
% omega_cell        body-axis rotation rate 
% omega_mw_mid_mag  medial band metachronal-wave rotation rate 
% omega_mw_ant_mag  anterior band metachronal-wave rotation rate 
% a_mid             medial band radius
% a_ant             anterior band radius
% mu                dynamic viscosity
% dt                time step
% lambda_mid        medial band metachronal-wave wave-length
% lambda_ant        anterior band metachronal-wave wave-length
% cilia_density     # of point forces per micron
% T                 total time
% beta_mid          translational mobility correction for medial band
% beta_ant          translational mobility correction for anterior band
% gamma_mid         rotational mobility correction for medial band
% gamma_ant         rotational mobility correction for anterior band

if nargin < 13 || isempty(beta_mid),  beta_mid  = 1; end
if nargin < 14 || isempty(beta_ant),  beta_ant  = beta_mid; end
if nargin < 15 || isempty(gamma_mid), gamma_mid = 1; end
if nargin < 16 || isempty(gamma_ant), gamma_ant = gamma_mid; end

R_body = a_mid;              % cell-body radius used in mobility denominators
varrho_mid = a_mid;          % radial distance of medial ring from the center

n_mid = floor(2*pi*a_mid*cilia_density);
n_ant = floor(2*pi*a_ant*cilia_density);
beta_weight = (beta_mid*n_mid + beta_ant*n_ant)/(n_mid+n_ant);
F = U*6*pi*mu*R_body/beta_weight;


h=R_body;

angles_mid = (0:(2*pi/(n_mid)):(2*pi-2*pi/(n_mid)))';
angles_ant = (0:(2*pi/(n_ant)):(2*pi-2*pi/(n_ant)))';

omega_mw_mid = omega_mw_mid_mag.*ones([n_mid,1]); 
omega_mw_ant = omega_mw_ant_mag.*ones([n_ant,1]);
    
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

pos = zeros(N_step,3);

% == time integration =====================================================
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

    t = t+ dt;

    pos(ii,:) = x_vec;

end
