function [pos,helix_radius,helix_pitch] = straight_meta_fn_sp(U,omega,a,mu,dt,R0,T,beta,gamma)
% Motion of a self–propelled sphere with an off-centre thrust.
%
% U               swim speed  
% omega_cell      body-axis rotation rate 
% omega_mw        metachronal-wave rotation rate 
% a               sphere radius
% mu              dynamic viscosity
% dt              time step
% R0              distance of force from COM
% T               total time
% beta            translational mobility correction (optional; default 1)
% gamma           rotational mobility correction (optional; default 1)

if nargin < 8 || isempty(beta),  beta  = 1; end
if nargin < 9 || isempty(gamma), gamma = 1; end

% == initial state ========================================================
p_hat   = [0 0 1];              % swimming direction (unit)
d_hat   = [1 0 0];              % unit vector from COM to force location
x_cm    = [0 0 0];              % COM position

N   = floor(T/dt);
pos = zeros(N+1,3);
pos(1,:) = x_cm;

sixpi_mu_a  = 6*pi*mu*a;
eightpi_mu_a3 = 8*pi*mu*a^3;

% == time integration =====================================================
for ii = 1:N
    F_mag   = sixpi_mu_a*U/beta;
    F_vec   = F_mag * p_hat;
    torque  = cross(a*R0*d_hat , F_vec);

    w_para  = (omega) * p_hat;      % prescribed spin about p̂
    w_perp  = gamma*torque / eightpi_mu_a3;            
    w_tot   = w_para + w_perp;                     

    w_norm  = norm(w_tot);
    if w_norm > 0
        n_hat  = w_tot / w_norm;
        theta  = w_norm * dt;

        p_hat  = p_hat * cos(theta) + ...
                 cross(n_hat , p_hat) * sin(theta) + ...
                 n_hat * dot(n_hat , p_hat) * (1 - cos(theta));

        d_hat  = d_hat * cos(theta) + ...
                 cross(n_hat , d_hat) * sin(theta) + ...
                 n_hat * dot(n_hat , d_hat) * (1 - cos(theta));

        p_hat = p_hat / norm(p_hat);
        d_hat = d_hat / norm(d_hat);
    end

    x_cm = x_cm + U * p_hat * dt;

    pos(ii+1,:) = x_cm;
end

% Estimate the helix axis using PCA
centroid = mean(pos, 1);      
[coeff, ~, ~] = pca(pos);      
axis_dir = coeff(:,1);        
axis_dir = axis_dir / norm(axis_dir);  
% plot3([0,10*axis_dir(1)],[0,10*axis_dir(2)],[0,10*axis_dir(3)],'--')
pos0 = pos - centroid;
v = axis_dir;
s = pos0 * v;  
w = [1;0;0];
if abs(dot(w,v)) > 0.9
    w = [0;1;0];
end
u1 = cross(v, w);
u1 = u1 / norm(u1);
u2 = cross(v, u1);
xproj = pos0 * u1;       % projection onto u1
yproj = pos0 * u2;       % projection onto u2
phi   = atan2(yproj, xproj);      % Nx1 angles in (−π,π]
phi_unwrap = unwrap(phi);
p = polyfit(phi_unwrap, s, 1);   
P = 2*pi * p(1);
r = pos - centroid;  
proj = (r * axis_dir) * axis_dir';
diff = r - proj;
distances = sqrt(sum(diff.^2, 2));

avg_radius = mean(distances);
A = gamma*F_mag*R0/(8*pi*mu*a^2);
disp(gamma*F_mag/(8*pi*mu*a^2))
r2 = U*A/(omega^2+A^2);
helix_radius = mean(distances);
helix_pitch = P;

% pitch = (2*pi*U*omega)./(omega^2+A^2);

% Display the result
% fprintf('Estimated radius:  %.4f \n', avg_radius);
% fprintf('perdicted radius  %.4f \n', r2);
% 
% fprintf('Estimated mean pitch: %.4f \n', P);
% fprintf('perdicted pitch:  %.4f \n\n', pitch);
end
