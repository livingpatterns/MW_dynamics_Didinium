import numpy as np
import pandas as pd
import seaborn as sns
import pims
import btrack
import os
from scipy.signal import savgol_filter
import tifffile as tiff
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import io
import matplotlib.animation as animation
from scipy.fft import fft2, fftshift, ifft2, ifftshift, fft, ifft


#these functions deal with quality control fo particle identification:

def quality_control1(features, path, n_bins=10):
    """
    Plots the probability distribution of the number of particles found per frame

    Parameters:
    features (dataframe): Data frame with the list of positions of particles found in each frame.
    path (str): Path where the histogram will be saved.
    n_bins (int): Number of bins for the histogram. The bin width is considered 1 since particles are discrete

    Returns:
    nothing, only saves the histogram as a png image in the specified path.

    """
    savefolder=os.path.join(path,'plots')
    os.makedirs(savefolder, exist_ok=True)

    max_frame = features['frame'].max()
    num_part_frame = np.zeros(max_frame + 1)
    for k in range(len(num_part_frame)):
        num_part_frame[k] = features['frame'].eq(k).sum()

    plt.plot(num_part_frame)
    plt.savefig(os.path.join(savefolder, 'n_particles_frame.png'))
    plt.close()
    counts, bin_edges = np.histogram(num_part_frame, bins=np.arange(0, n_bins, 1))
    plt.bar(bin_edges[0:-1] - 0.5, counts / len(num_part_frame), align='edge', width=1)
    plt.xticks(bin_edges, bin_edges.astype(str))
    plt.xlabel('Number of particles per frame')
    plt.savefig(os.path.join(savefolder,'n_particles_found.png'))
    plt.close()


def quality_control2(features, frames, path):
    """
    Plot particle positions stored in the features dataframe on top of the image frames they were found

    Parameters:
    features (dataframe): Data frame with the list of positions of particles found in each frame.
    frames (pims image sequence): TPims image frame that corresponds to the data where the particles were found.
    path (str): Path where the plots will be saved.

    Returns:
    nothing, only saves the plots as a sequence of png images in the specified path.

    """
    grouped = features.groupby('frame')
    images = []

    # Iterate over the groups
    for frame, group in grouped:
        # Check if the group is not empty
        if not group.empty:
            # Get the first row of the group
            # row = group.iloc[0]
            y = group['y']
            x = group['x']

            # Plot the frame and the particle
            plt.imshow(frames[frame])
            plt.scatter(x, y, facecolors='none', edgecolors='r')

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Append the image data to images
            images.append(imageio.imread(buf))

            plt.close()

    # Save images as an mp4 video
    imageio.mimsave(path + '/' + 'particle_check.mp4', images, fps=30)



#this function runs btrack:
def run_btracking(tracking_features, tracking_updates,tracking_objs,config_file_path,max_disp):
    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        tracker.configure(config_file_path + '/cell_config.json')
        tracker.max_search_radius = max_disp  # this is the max displacement
        tracker.tracking_updates = ["MOTION"]

        # append the objects to be tracked
        tracker.append(tracking_objs)

        # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
        tracker.volume = ((0, 990), (0, 1340))
        tracker.features = tracking_features

        # track them (in interactive mode)
        tracker.track(step_size=100,tracking_updates=tracking_updates)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks as a python list
        tracks2 = tracker.tracks
    return tracks2



#convert the tracks object into a data frame for easier handling:
def tracks_to_dataframe(tracks):
    """
    Converts a list of tracks into a DataFrame suitable for input into the connect_tracks function.

    Parameters:
        tracks (list): List of track objects.

    Returns:
        DataFrame: DataFrame containing track information.
    """
    data = []
    for track in tracks:
        for t, x, y in zip(track.t, track.x, track.y):
            data.append({'particle': track.ID, 'frame': t, 'x': x, 'y': y})

    df = pd.DataFrame(data)
    return df



#these functions deal with importing the trajectories, visualizing them and sorting them for analysis:
def get_tracks(track_files):
    "This function reads the tracking data from the .pkl files and organizes it into numpy arrays."
    # Read tracking data from .pkl files
    tracks = [pd.read_pickle(f) for f in track_files]

    # Find out many tracks are in total from all files and what is the maximum frame number
    n_tracks = sum(len(track.groupby('particle')['particle'].unique().index.values) for track in tracks)
    t_max = max(track['frame'].max() for track in tracks) + 1  # maximum frame number

    all_x = np.full((t_max, n_tracks), np.nan)
    all_y = np.full((t_max, n_tracks), np.nan)

    ntrack = 0
    for track in tracks:
        grouped = track.groupby('particle')
        particles = grouped['particle'].unique().index.values

        for i in range(0, len(particles)):
            df = grouped.get_group(particles[i])[['x', 'y', 'frame']].copy()
            frame_indices = np.array(df['frame'].astype(int))

            # fill the arrays with the values of the tracks
            all_x[frame_indices, ntrack] = df.loc[frame_indices]['x'].values
            all_y[frame_indices, ntrack] = df.loc[frame_indices]['y'].values
            ntrack = ntrack + 1
    return all_x, all_y

def filter_tracks_by_distance(tracks, min_distance, max_distance):
    """
    Filters tracks based on a specified range of end-to-end distances.

    Parameters:
        tracks (pd.DataFrame): DataFrame containing track data with 'particle', 'x', and 'y' columns.
        min_distance (float): Minimum end-to-end distance to keep.
        max_distance (float): Maximum end-to-end distance to keep.

    Returns:
        pd.DataFrame: Filtered DataFrame containing tracks within the specified distance range.
    """
    # Calculate end-to-end distance for each track
    end_to_end_distances = tracks.groupby('particle')[['x', 'y']].apply(
        lambda group: np.sqrt((group['x'].iloc[-1] - group['x'].iloc[0]) ** 2 + (group['y'].iloc[-1] - group['y'].iloc[0]) ** 2)
    ).reset_index()

    # Rename columns for clarity
    end_to_end_distances.columns = ['track_number', 'end_to_end_distance']

    # Identify tracks outside the specified distance range
    trash_tracks = end_to_end_distances[
        (end_to_end_distances['end_to_end_distance'] < min_distance) |
        (end_to_end_distances['end_to_end_distance'] > max_distance)
    ]['track_number'].tolist()

    # Filter tracks within the specified distance range
    filtered_tracks = tracks[~tracks['particle'].isin(trash_tracks)]

    return filtered_tracks

def filter_tracks_by_particle_id(tracks, particle_ids):
    """
    Remove tracks with specific particle IDs from the DataFrame.

    Parameters:
    - tracks: DataFrame containing the tracks.
    - particle_ids: List of particle IDs to remove.

    Returns:
    - DataFrame with the specified particle IDs removed.
    """
    return tracks[~tracks['particle'].isin(particle_ids)]

def plot_trajectories(all_x,all_y, plots_per_row=4, xlim=(-1000, 1000), ylim=(-1000, 1000), figsize=(20, 5)):
    """
    Plot the trajectories of particles with a maximum number of subplots per row.

    Parameters:
        all_x (numpy array): 2D numpy array where each column represents the x-coordinates of a trajectory.
        all_y (numpy array): 2D numpy array where each column represents the y-coordinates of a trajectory.
        plots_per_row (int): Maximum number of subplots per row.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        figsize (tuple): Size of the figure.

    Returns:
        None
    """
    all_x=np.copy(all_x)
    all_y=np.copy(all_y)
    num_plots = all_x.shape[1]
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(figsize[0], figsize[1] * num_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i in range(0,num_plots):
        x, y =all_x[:,i], all_y[:, i]
        x -=  x[np.where(~np.isnan(x))[0][0]]
        y -=  y[np.where(~np.isnan(y))[0][0]]
        axs[i].plot(x, y)
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        axs[i].set_title(f'Trajectory {i}')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def timecolorplot_n(trajectory_x, trajectory_y, plots_per_row=4, xlim=(-800, 800),
                                ylim=(-800, 800), figsize=(20, 5)):
    """
    Plot time-color trajectories of particles with a maximum number of subplots per row.

    Parameters:
        trajectory_x (numpy array): 2D numpy array where each column represents the x-coordinates of a trajectory.
        trajectory_y (numpy array): 2D numpy array where each column represents the y-coordinates of a trajectory.
        time (numpy array): 1D numpy array representing the time steps.
        plots_per_row (int): Maximum number of subplots per row.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        figsize (tuple): Size of the figure.

    Returns:
        None
    """
    trajectory_x=np.copy(trajectory_x)
    trajectory_y=np.copy(trajectory_y)
    time = np.arange(np.shape(trajectory_x)[0])
    num_plots = trajectory_x.shape[1]
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(figsize[0], figsize[1] * num_rows))
    axs = np.array(axs).flatten()  # Flatten in case of multiple rows

    for i in range(num_plots):
        x, y = trajectory_x[:, i], trajectory_y[:, i]
        x -=  x[np.where(~np.isnan(x))[0][0]]
        y -= y[np.where(~np.isnan(y))[0][0]]

        xy = np.column_stack((x, y))
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])

        norm = plt.Normalize(time.min(), time.max())
        coll = LineCollection(segments, cmap=plt.cm.plasma, norm=norm)
        coll.set_array(time)
        coll.set_linewidth(1)

        ax = axs[i]
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'Trajectory {i}')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma), cax=cax)
        cax.set_title('Time')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def av_speed(all_x,all_y):
    #calculate the instantaneous velocity of the particles
    vel_mag=np.sqrt(np.diff(all_x,axis=0)**2+np.diff(all_y,axis=0)**2)
    #calculate the average velocity of the particles
    av_vel=np.nanmean(vel_mag,axis=0)
    return av_vel

def timecolorplot(xy, param):
    """
    Function that plots plots a trajectory color coded by value of a parameter
    ------------------------
    <parameters>:
    xy :    Np array with x(t),y(t) coordinates of a particle. Format should be such that xy.shape=[max(t),2]
    param:   1D np array or list that has all the values of the parameter of interest

    <Returns>:
    fig: Matplotlib figure with trajectory plot color coded by time
    """
    xy = xy.reshape(
        -1, 1, 2
    )  # Reshape things so that we have a sequence of: [[(x0,y0),(x1,y1)],[(x0,y0),(x1,y1)],...]
    segments = np.hstack([xy[:-1], xy[1:]])
    norm = plt.Normalize(param.min(), param.max())
    coll = LineCollection(segments, cmap=plt.cm.plasma, norm=norm)
    coll.set_array(param)
    coll.set_linewidth(1)

    fig, ax = plt.subplots(1, 1)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.invert_yaxis()

    # Calculate the center of the x and y coordinates
    x_center = (xy[:, 0, 0].min() + xy[:, 0, 0].max()) / 2
    y_center = (xy[:, 0, 1].min() + xy[:, 0, 1].max()) / 2
    # Set axis limits to be 10mm in length centered around the calculated center
    ax.set_xlim(x_center - 8, x_center + 8)
    ax.set_ylim(y_center - 8, y_center + 8)

    plt.xlabel('mm', fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma), cax=cax)
    cax.set_title('time (min)')
    return fig


#These functions deal with trajectory analysis
def filter_trajectory_savgol(trajectory: pd.DataFrame, window_length) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to smooth trajectory.

    Inputs:
        trajectory: The input trajectory data.
        window_length: The length of the filter window.

    Returns:
        Smoothed trajectory.
    """
    return pd.DataFrame(
        {
            't': trajectory['time'].to_numpy(),
            'x': savgol_filter(trajectory['x'].to_numpy(), window_length, 2),
            'y': savgol_filter(trajectory['y'].to_numpy(), window_length, 2)
        }
    )




def compute_kinematics_values(trajectory):
    """
    Compute kinematics values from trajectory data.

    Inputs:
        trajectory: The input trajectory data.
        dt: 1/fps of the video.
        scale: scale of how many pixels correspong to 1mm.

    Returns:
        Computed kinematics values.
    """
    xy = np.array(trajectory[['x', 'y']])
    t = trajectory['t'].to_numpy()

    dx=np.diff(xy, axis=0)
    dt=np.diff(t)

    linear_velocity = dx/dt[:, None]
    linear_velocity_norm = np.linalg.norm(linear_velocity, ord=2, axis=1)

    cosine_of_angle = np.sum(linear_velocity[:-1, :] * linear_velocity[1:, :], axis=1) / np.clip(
        (np.linalg.norm(linear_velocity[:-1, :], axis=1) * np.linalg.norm(linear_velocity[1:, :], axis=1)), a_min=1e-18,
        a_max=None)
    angular_velocity = np.arccos(cosine_of_angle)
    angular_velocity = np.nan_to_num(angular_velocity, nan=0.0)

    acceleration = np.diff(linear_velocity_norm)/dt[:-1]

    return linear_velocity, linear_velocity_norm[:-1], acceleration, angular_velocity

def probability_distribution_plot(value,
                                  x_label,
                                  title=None,
                                  y_label="Probability",
                                  bins=150,
                                  plot_size=(10, 6)
                                  ):
    """
    Plot probability distribution of a single variable.

    Inputs:
        value: Data representing the variable.
        valuename: Name of the variable.
        title: Title of the plot.

    Returns:
        figure
    """
    if title is None:
        title = f"Distribution of {x_label}"

    fig = plt.figure(figsize=plot_size)

    sns.histplot(value, kde=True, stat="probability", bins=bins)
    plt.title(title, fontsize=18)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    return fig


def joint_probability_distribution_plot(x,
                                        y,
                                        xaxis,
                                        yaxis,
                                        title,
                                        plot_size=(10, 6)
                                        ):
    """
    Plot joint probability distribution of two variables.

    Inputs:
        x: Data for the x-axis variable.
        y: Data for the y-axis variable.
        xaxis: Label for the x-axis.
        yaxis: Label for the y-axis.
        title: Title of the plot.

    Returns:
        figure
    """
    fig = plt.figure(figsize=plot_size)

    sns.jointplot(x=x, y=y, kind='kde')
    plt.suptitle(title, fontsize=18)
    plt.xlabel(xaxis, fontsize=18)
    plt.ylabel(yaxis, fontsize=18)
    plt.tight_layout()

    return fig


def autocorrelation2D_fft(c_intensity):
    """
    Calculates the autocorelation in time and space using fourier transforms.
    """

    # implementing of Wiener-Khinchin theorem
    c = np.real(fftshift(ifft2(fft2(c_intensity) * np.conj(fft2(c_intensity)))))
    # normalizing
    c = c / np.max(c)

    return c
def autocorrelation(x):
    """
    Autocorrelation function from:
    https://stackoverflow.com/questions/47850760/using-scipy-fft-to-calculate-autocorrelation-of-a-signal-gives-different-answer
    """
    xp = ifftshift((x - np.average(x)) / np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n // 2], np.zeros_like(xp), xp[n // 2:]]
    f = fft(xp)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    return np.real(pi)[:n // 2] / (np.arange(n // 2)[::-1] + n // 2)


def make_movie(pos_df, movie_path, analysis_path):
    savename = os.path.basename(analysis_path) + "_trajectory_check2.mp4"

    # Open video using pims
    video = pims.open(movie_path)

    # Get figure and axis
    fig, ax = plt.subplots()
    img_display = ax.imshow(video[0], cmap='gray')
    line, = ax.plot([], [], marker="o", linestyle="-", color="g",markerfacecolor='none')

    def update(frame):
        if frame >= len(video):
            return
        ax.clear()
        ax.imshow(video[frame], cmap="gray")

        # Select relevant trajectory data
        past_frames = max(0, frame - 10)
        traj = pos_df.loc[past_frames:frame]
        ax.plot(traj["y"], traj["x"], marker="o", linestyle="-", color="g",markerfacecolor='none')

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(video), repeat=False)

    # Save animation as MP4
    ani.save(os.path.join(analysis_path, savename), fps=30, extra_args=["-vcodec", "libx264"])

    print(f"{savename} movie saved")

def smooth_and_plot_tracks(path, filter_window, polyorder):
    """
    Processes all folders in the given path to smooth tracks, plot them on images, and save the of all videos in a single dataframe.

    Parameters:
        path (str): Path to the directory containing folders with '_analysis' in their names.

    Returns:
        pd.DataFrame: DataFrame containing smoothed tracks from all folders.
    """

    def plot_smoothed_tracks_on_image(smoothed_tracks, im, current_folder):
        """
        Plots all smoothed tracks on top of the image and saves the plot.

        Parameters:
            smoothed_tracks (pd.DataFrame): DataFrame containing smoothed track data.
            im (np.ndarray): Image array to plot the tracks on.
            current_folder (str): Path to the current folder for saving the plot.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(im, cmap="gray")  # Display the image
        plt.axis('off')  # Remove axes
        plt.title('Smoothed Tracks on Max Projection Image')

        # Plot each track
        for particle_id in smoothed_tracks['particle'].unique():
            track = smoothed_tracks[smoothed_tracks['particle'] == particle_id]
            plt.plot(track['x'], track['y'], lw=1, label=f'Particle {particle_id}')

        plt.tight_layout()
        plots_folder = os.path.join(current_folder, 'plots')
        os.makedirs(plots_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_folder, 'smoothed_tracks_on_image.png'))
        plt.show()

    # Get folders in path that contain '_analysis' in their names
    folders = sorted([os.path.join(path, f) for f in os.listdir(path) if '_analysis' in f])

    # Initialize an empty list to store smoothed tracks from all folders
    all_smoothed_tracks = []
    max_particle_id = 0  # Track the maximum particle ID across folders

    for current_folder in folders:
        # Get the image name and read the max projection image
        im_name = os.path.basename(current_folder).replace('processed_', '').replace('_analysis',
                                                                                     '') + '_max_projection.tif'
        im = tiff.imread(os.path.join(os.path.dirname(current_folder), im_name))

        # Read the filtered trajectory file
        tracks = pd.read_pickle(os.path.join(current_folder, 'filtered_tracks_tp.pkl'))
        tracks = tracks[['x', 'y', 'frame', 'particle']].copy()

        # Change the labels so they are ascending from one to N and ensure unique IDs
        tracks['particle'] = pd.factorize(tracks['particle'])[0] + 1 + max_particle_id
        max_particle_id = tracks['particle'].max()  # Update the maximum particle ID
        grouped_tracks = tracks.groupby('particle')

        # Initialize an empty list to store smoothed data for the current folder
        smoothed_tracks = []
        for particle_id, current_track in grouped_tracks:
            # Perform linear interpolation if there are NaN values
            if current_track['x'].isnull().any():
                current_track['x'] = current_track['x'].interpolate(method='linear')
                current_track['y'] = current_track['y'].interpolate(method='linear')

            # Apply Savitzky-Golay filter
            smooth_track_x = savgol_filter(current_track['x'], window_length=filter_window, polyorder=polyorder,
                                           mode='nearest')
            smooth_track_y = savgol_filter(current_track['y'], window_length=filter_window, polyorder=polyorder,
                                           mode='nearest')
            smooth_track_time = current_track['frame'].values

            # Create a DataFrame for the smoothed data of the current track
            smoothed_data = pd.DataFrame({
                'particle': particle_id,
                'x': smooth_track_x,
                'y': smooth_track_y,
                'frame': smooth_track_time
            })

            # Append the smoothed data to the list
            smoothed_tracks.append(smoothed_data)

        # Concatenate the smoothed tracks for the current folder
        smoothed_tracks = pd.concat(smoothed_tracks, ignore_index=True)
        all_smoothed_tracks.append(smoothed_tracks)
        plot_smoothed_tracks_on_image(smoothed_tracks, im, current_folder)

    # Concatenate smoothed tracks from all folders into a single DataFrame
    all_smoothed_tracks_df = pd.concat(all_smoothed_tracks, ignore_index=True)

    # Save the smoothed tracks to a pkl file in the same folder
    all_smoothed_tracks_df.to_pickle(
        os.path.join(path, os.path.basename(os.path.normpath(path)) + '_all_smoothed_tracks_tp.pkl'))
    #save parameters used for smoothing in a text file
    with open(os.path.join(path, 'parameters_smoothing.txt'), 'w') as f:
        f.write(f'filter_window={filter_window}\n')
        f.write(f'polyorder={polyorder}\n')

    return all_smoothed_tracks_df