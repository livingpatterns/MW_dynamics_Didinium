import os
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import imageio
from skimage import filters


# Define constants
CHUNK_SIZE = 10000
MAX_WORKERS = 2
VIDEO_PATH = '/Users/guillermina/Desktop/didinium_swimming/'


SAVE_PATH=os.path.join(VIDEO_PATH, 'bgd_subs_'+str(CHUNK_SIZE))
#don't create the folder if it already exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Function to read video frame-by-frame (lazy loading)
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale on the fly
    cap.release()

# Function to compute median background in chunks
def calculate_background(frames):
    frames_list = list(frames)
    temp= np.mean(frames_list, axis=0)
    #apply a median filter to the mean image
    temp=filters.median(temp, np.ones((10, 10)))
    return temp

# Function to process a single frame without gaussian filtering
def process_frame(frame, background):
    return np.uint8(np.abs(frame.astype(float) - background.astype(float)))

# Process video file
def process_video(video_file):
    start_time = time.time()  # Start timing
    video_path = os.path.join(VIDEO_PATH, video_file)
    frames = read_video_frames(video_path)

    # Compute number of chunks
    frame_list = list(frames)  # Load frames lazily
    num_chunks = (len(frame_list) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Compute background frames in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunks = [frame_list[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
        bgds = list(executor.map(calculate_background, chunks))

    # Process frames and write output
    output_path = os.path.join(SAVE_PATH, "processed_" + video_file)
    with imageio.get_writer(output_path, fps=30) as writer:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for chunk, bgd in zip(chunks, bgds):
                processed_chunk = list(executor.map(lambda frame: process_frame(frame, bgd), chunk))

                for frame in processed_chunk:
                    writer.append_data(frame)

    end_time = time.time()  # End timing
    print(f"Processing time for {video_file}: {end_time - start_time:.2f} seconds")

# Process all videos in the directory
start_time_all = time.time()  # Start timing for all videos
video_files = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(process_video, video_files)

end_time_all = time.time()  # End timing for all videos
print(f"Total execution time: {end_time_all - start_time_all:.2f} seconds")

#write a text file with the parameters used to process the video and save it to the same directory as the video
file_name = os.path.join(SAVE_PATH, 'parameters.txt')
with open(file_name, 'w') as f:
   f.write(f'CHUNK_SIZE = {CHUNK_SIZE}\n')
   f.write(f'MAX_WORKERS = {MAX_WORKERS}\n')
   f.write(f'VIDEO_PATH = {VIDEO_PATH}\n')
   f.write(f'VIDEO_FILES = {video_files}\n')
   f.close()