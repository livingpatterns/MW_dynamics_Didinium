import numpy as np
import pims
import multiprocessing
import matplotlib.pyplot as plt
import os
import trackpy as tp
import pandas as pd
from skimage import filters, morphology
from tracking_functions import quality_control1

#use pims pipeline to convert image to grayscale taking only one color
@pims.pipeline
def gray(image):
    temp= image[:, :, 0]#taking only one color makes things significantly faster and doesn't affect the particle identification
    return temp#for 20X sigma=5 works well


multiprocessing.set_start_method('fork')#needed this state ment to be able to parallelize trackpy particle locating in mac

# Specify the path with your microscopy images
path = '/Users/guillermina/Desktop/didinium_trajectories/filtered_videos/small/'
#list of videos in the path sorted by name
#video_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])
video_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif') and 'small' in f])


#video_files=path+'processed_CC620_01.tif'
#make video files a list of length 1
#video_files=[video_files]

n_video=0#video to be analyzed for testing
video=pims.open(video_files[n_video])
#video=gray(pims.open(video_files[n_video]))


n_frame=0#frame to be analyzed for testing
diameter = 27
minmass = 3000#3000


test_frame=video[n_frame].copy()

test = tp.locate(test_frame, diameter=diameter, minmass=minmass,invert=True)
tp.annotate(test, test_frame,plot_style={'markersize': diameter})

#video_name=os.path.basename(video_files[n_video])#name of the video

#do this for all videos in video_files
for n_video in range(len(video_files)):
    video_name=os.path.basename(video_files[n_video])#name of the video
    analysis_path = os.path.join(os.path.dirname(path), video_name[:-4]+'_analysis')#path were results will be saved
    os.makedirs(analysis_path, exist_ok=True)#create the folder if it doesn't exist
    #video=gray(pims.open(path+video_name)) #test video
    video=pims.open(path+video_name)
    #the following parameters are defined to test the particle locating function
    #test parameters to locate particles

    # Convert the generator to a list before passing it to tp.batch
    frames = list(video)

    # Once I am happy with the parameters I can batch locate the particles in the entire video
    tp.quiet()  # turn off progress reports for now
    particles = tp.batch(frames, diameter=diameter, minmass=minmass, invert=True,processes='auto')

    #sanity check for particle identification:
    quality_control1(particles, analysis_path,n_bins=10)  # Make a histogram of the number of particles per frame found
    particles.to_pickle(os.path.join(analysis_path, 'all_raw_particle_positions.pkl'))  # save dataframe with all particle positions

    #Save parameters used for particle identification in a text file
    with open(analysis_path+'/parameters_part_id.txt', 'w') as f:
        f.write('diameter='+str(diameter)+'\n')
        f.write('minmass='+str(minmass)+'\n')
        f.write('Invert=True'+'\n')
        f.close()


#If you are happy with your quality control plots proceed to the particle_tracking.py script