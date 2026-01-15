import trackpy as tp
import os
import pandas as pd
from btrack.io import localizations_to_objects
from tracking_functions import tracks_to_dataframe, run_btracking


# Specify the path with your .pkl files with particle detections are stored
path = '/Users/guillermina/Desktop/didinium_trajectories/filtered_videos/small/'


#list of folders that end in _analysis
analysis_folders = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_analysis')]
#analysis_folders = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_analysis') and '2530' in f]

video_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])
#video_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif') and '2530' in f])

#to test things one video at a time
#video_files=path+'processed_CC620_01.tif'
#make video files a list of length 1
#video_files=[video_files]
#analysis_folders=path+'processed_cc124_10x 001_analysis'
#analysis_folders=[analysis_folders]

for n_video in range(0,len(video_files)):
#n_video=0#video to be analyzed for testing
#name of video dropping the word processed and the last word analysis
    analysis_path=analysis_folders[n_video]
    video_name = os.path.basename(analysis_path.split('_', 1)[1].rsplit('_', 1)[0])  # name of the video
    print(video_name)
    #Tracking with trackpy
    #import .pkl file with particle positions
    track_me=pd.read_pickle(analysis_path+'/all_raw_particle_positions.pkl')

    #Track the particles using trackpy
    search_range=100#maximum distance in pixels that a particle can move between frames
    memory=5#maximum number of frames a particle can be lost and still be tracked
    threshold=50#minimum number of frames a particle has to be tracked to be considered good

    tp.quiet()  # turn off progress reports for now
    tracks = tp.link(track_me, search_range=search_range, memory=memory)
    pruned_tracks = tp.filter_stubs(tracks, threshold=threshold)#remove tracks that are shorter than 800 frames
    tp.plot_traj(pruned_tracks)
    if not pruned_tracks.empty:
        ax=tp.plot_traj(pruned_tracks)
        fig=ax.get_figure()
        #save plotted tracks in analysis folder
        fig.savefig(analysis_path+'/plots/all_tracks_tp.png')
        #save the tracks if dataframe is not empty

        pruned_tracks.to_pickle(analysis_path+'/all_tracks_tp.pkl')

    #save the parameters used for tracking in a text file
    with open(analysis_path+'/parameters_tracking_tp.txt', 'w') as f:
        f.write('search_range='+str(search_range)+'\n')
        f.write('memory='+str(memory)+'\n')
        f.write('threshold='+str(threshold)+'\n')
        f.close()

