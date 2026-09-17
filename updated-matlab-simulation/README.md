# Didinium swimming simulations

MATLAB code accompanying *Metachronal wave dynamics encode multimodal swimming in ciliated unicellular predators*. The models explore how coordination of the two ciliary bands produces directed swimming, helical trajectories, and reorientation during wave reversal.

The band models represent ciliary forcing as rings of point forces with traveling sinusoidal amplitudes. Net force determines translation, and off-center forcing generates a steering torque. A reduced model replaces the bands with a single off-center thrust to relate force asymmetry to helix radius and pitch.

## Simulation functions

- `straight_meta_fn.m`: two bands of point forces with traveling metachronal waves. Returns the xyz trajectory `pos`. Used for Fig. 3.
- `straight_meta_fn_sp.m`: a sphere driven by a single off-center thrust. Returns `pos`, helix radius, and signed pitch estimated using PCA. Used for Fig. 3 and Fig. S5. Its `R0` argument is the offset divided by the sphere radius, corresponding to chi in the manuscript; the physical lever arm is `a*R0`.
- `reversal_w_spin_lag_cont_fn.m`: a reversal front propagates around each band at a constant rate. `lag_time` delays the anterior band relative to the medial band. This is the “linear” protocol in Fig. S8.
- `reversal_w_spin_desync_fn.m`: forcing is switched off at unreversed sites while a reversed wave is rebuilt sequentially around each band. This is the “asynchronous” protocol in Fig. S8.
- `reversal_w_spin_step_fn.m`: half of each band reverses abruptly during the reversal interval. This is the “Heaviside step” protocol in Fig. S8.

All three reversal functions return `pos` and `p_hat_list`, the position and unit body axis at each update. `start_time` sets the initial forward-swimming interval and the duration simulated after reversal. Total duration is `2*start_time + reversal_time`, with `lag_time` added for the linear protocol.

Lengths are in micrometers, times in seconds, and angular rates in radians per second. `mu` is dynamic viscosity, with the runner using kg/(um s). `omega1` and `omega2` in the reversal functions are the medial and anterior wave rates, respectively. The optional `beta` and `gamma` parameters scale translational and rotational mobility. They default to 1 in the reduced model; in the band models, medial values default to 1 and anterior values default to their medial counterparts.

The band models return `floor(T/dt)+1` rows recorded after position updates, without the initial origin. The reduced model also returns `floor(T/dt)+1` rows, but its first row is the initial origin.

## Running the code

Open this folder as the MATLAB current folder. The manuscript reports MATLAB R2024b. The helix measurements and trajectory plots require Statistics and Machine Learning Toolbox for `pca`.

`make_figures.m` is the runner. Start with the startup and parameter sections, then run the analysis sections in order. They cover sample trajectories, helix geometry and experimental comparisons, wave-parameter sweeps, reversal protocols, and random wavefront offsets. Later sections reuse variables from earlier ones. Figures and CSV files are written to the current folder, replacing files with the same names. The random-offset section uses `randn`; set `rng` beforehand for repeatable samples.

For a short standalone example of the reduced model:

```matlab
[pos, radius, pitch] = straight_meta_fn_sp(1300, 17, 54, 1e-9, 1e-4, 0.2, 1);
plot3(pos(:,1), pos(:,2), pos(:,3));
axis equal
xlabel('x (um)'); ylabel('y (um)'); zlabel('z (um)');
```

## Plotting helpers

- `plotKymograph2.m`: plots medial-band force magnitude over time. 
- `make_my_video.m`: animates the two bands, coloring positive forces red and negative forces blue. Set `video = 1` in the linear or asynchronous reversal function to save `lag_cont.mp4` or `desync.mp4`.
- `pretty_helix_plot`: a local function at the end of `make_figures.m` that aligns a trajectory with its principal axis for plotting.
- `viridis.m`: supplies the colormap and retains its original author attribution.

## Data

- `all_tracks_helix_measurements.csv`: experimental helix measurements, with track and segment identifiers, radius, radius standard deviation, pitch, and helix angle. The runner converts radius and pitch from millimeters to micrometers; angles are in degrees.
- `sample_trajectory.csv`: four simulated trajectories from the first trajectory section. Each group of three columns contains x, y, and z positions in micrometers.
- `helix_radius.csv` and `helix_angle.csv`: outputs of the analytical lever-arm sweep, without headers. Rows follow `R0_list`; columns follow `omega_list` (currently 3.5, 14, and 28 rad/s). Radius is in micrometers and angle is in degrees.

The runner also exports PNG figures, noise predictions and sampled results in `node_gaussian_noise_*.csv`, and three reduced-model trajectories in `track1.csv` through `track3.csv`. In the noise exports, `p_theory` stores pitch in micrometers, while the existing `p_sim` column stores helix angle in degrees.
