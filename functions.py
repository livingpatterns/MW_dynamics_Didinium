
import os
import cv2
import csv
import tifffile
import pandas as pd

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft2, fftshift, ifft2
from scipy.optimize import leastsq

from skimage.draw import disk
from skimage.morphology import disk as disk_vel
import numpy.matlib as matlib
import scipy.ndimage as ndimage
from scipy.ndimage import median_filter

import polarTransform
from tqdm import tqdm

def open_tif(file_path):
    """
    Given a tif file path, it stores the timelapse in the array 'image' and 
    stores its dimensions.  
    """
  
    image        = tifffile.imread(file_path)
    num_frames   = image.shape[0]  # number of frames
    frame_height = image.shape[1]  # height of frame
    frame_width  = image.shape[2]  # width of frame
   
    return image, frame_width, frame_height, num_frames


def show_frame_no_axes(image, frame_index):
    """ 
    Displays a single frame of a given timelapse, requested by the user.
    Axes are hidden.
    """
    
    if frame_index > image.shape[0] - 1:
        print("The tiff file only has a total of " + str(image.shape[0]) + " frames!")
    else:         
        plt.figure()
        plt.axis('off')
        plt.title("Frame " + str(frame_index))
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.imshow(image[frame_index,:,:], cmap="gray")
    
    return


def show_frame_pixels(image, frame_index):
    """ 
    Displays a single frame of a given timelapse, requested by the user.
    Axis given in pixel.
    """
    
    if frame_index > image.shape[0] - 1:
        print("The tiff file only has a total of " + str(image.shape[0]) + " frames!")
    else:         
        plt.figure()
        plt.title("Frame " + str(frame_index))
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')
        plt.imshow(image[frame_index,:,:])
    
    return


def show_frame_in_microns(image, frame_index, pixel_size):
    """ 
    Displays a single frame of a given time-lapse, requested by the user.
    Axis given in microns.
    """
    # Calculate image dimensions in microns
    height_pixels = image.shape[1]
    width_pixels = image.shape[2]
    height_microns = height_pixels * pixel_size
    width_microns = width_pixels * pixel_size
        
    plt.figure()
    plt.title("Frame " + str(frame_index))
    plt.xlabel('x (\u03BCm)')
    plt.ylabel('y (\u03BCm)')
        
    # Define extent to reflect microns
    extent = [0, width_microns, height_microns, 0]
        
    plt.imshow(image[frame_index,:,:], extent=extent)


def img_norm_simple(timelapse):
    """
    Normalizes the intensity of each pixel based on its min and max values through time.
    """
    timelapse_min  = np.min(timelapse, axis = 0)[np.newaxis,:, :] # minimum of a pixel for all times
    timelapse_max  = np.max(timelapse, axis = 0)[np.newaxis,:, :] # maximum of a pixel for all times
    timelapse_norm = (timelapse - timelapse_min)/(timelapse_max - timelapse_min)
    
    return timelapse_norm


def uneven_illumination_fix(images, kernel_size):
    """
    Fixes uneven illumination that creates intensity artifacts.
    """
    corrected_images = []
    background_backup = []
    kernel_size = 31

    for i, image in enumerate(images):
        image_float = image.astype(np.float32)
        background = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 0)
        background[background == 0] = 1
        corrected_image = image_float / background
        corrected_images.append(corrected_image)
        background_backup.append(background)
        
    corrected_images = np.array(corrected_images)
    background_backup = np.array(background_backup)
    
    corrected_images_normalized = cv2.normalize(corrected_images, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return corrected_images_normalized



def select_points_perimeter_and_save(image_preview, output_file):
    """
    Allows manual selection of points on cell perimeter for the 
    determination of the radius and the cell center.
    """
    
    points = []

    def onclick(event):
        nonlocal points
        if event.inaxes is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            points.append((x, y))
            ax.plot(x, y, 'ro', markersize=3)
            fig.canvas.draw()

    def save_points(event):
        nonlocal points
        if len(points) == 0:
            print('No points selected to save.')
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])  # Write header
            for point in points:
                writer.writerow(point)

        for point in points:
            ax.plot(point[0], point[1], 'ro', markersize=3)
        fig.savefig(output_file.replace('.csv', '.png'))
        plt.close(fig)

        print(f"Points saved to {output_file}")


    fig, ax = plt.subplots()
    ax.imshow(image_preview)
    ax.set_title('Click to select points')
    ax.set_aspect('equal')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.connect('key_press_event', save_points)

    plt.show()

def select_points_line_and_save(image_preview, output_file):
    """
    Allows selection of linear segment on which to track image
    intensity for the kymograph. Segment should be drawn along the 
    ciliary band, at the middle height of the cilia and  perpendicular 
    to the cell axis.
    Used in the directed swimming analysis.
    """
    points = []
    drawing = False  
    initial_point = None 
    current_line = None  

    def on_press(event):
        nonlocal drawing, initial_point, current_line
        if event.inaxes is not None:
            drawing = True
            initial_point = (int(event.xdata), int(event.ydata))
            points.append(initial_point)
            ax.plot(initial_point[0], initial_point[1], 'ro', markersize=3)
            fig.canvas.draw()

    def on_motion(event):
        nonlocal drawing, initial_point, current_line
        if drawing and event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            
            if current_line is not None:
                current_line.pop(0).remove()
            
            current_line = ax.plot([initial_point[0], x], [initial_point[1], y], 'b-')
            fig.canvas.draw()

    def on_release(event):
        nonlocal drawing, initial_point, current_line
        if drawing and event.inaxes is not None:
            final_point = (int(event.xdata), int(event.ydata))
            points.append(final_point)
            ax.plot([initial_point[0], final_point[0]], [initial_point[1], final_point[1]], 'b-')  # Finalize the line
            fig.canvas.draw()
            drawing = False
            initial_point = None
            current_line = None  # Reset the current line

    def save_points(event):
        nonlocal points
        if len(points) == 0:
            print('No points selected to save.')
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])  # Write header
            for point in points:
                writer.writerow(point)

        for point in points:
            ax.plot(point[0], point[1], 'ro', markersize=3)
        fig.savefig(output_file.replace('.csv', '.png'))
        plt.close(fig)

        print(f"Points and drawing saved to {output_file}")

    fig, ax = plt.subplots()
    ax.imshow(image_preview)
    ax.set_title('Click and hold to draw straight lines, press any key to save points and drawing')
    ax.set_aspect('equal')

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', save_points)

    plt.show()

def read_line_borders_to_array(output_file):
    """
    Reads the manually chosen points definign the linear segment 
    on which to track image intensity for the kymograph.
    Used in the directed swimming analysis.
    """
    
    with open(output_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        points = np.array([[int(row[0]), int(row[1])] for row in reader])
    
    return points

def save_size_csv(output_file, x_c, y_c, r_c):
    """
    Save the size and center of the cell in a csv file.
    """
    # Check if the file exists
    file_exists = os.path.exists(output_file)

    with open(output_file, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
    
        # Write header if the file is newly created
        if not file_exists:
            writer.writerow(['x_c (pixel)', 'y_c (pixel)', 'r_c (pixel)'])
    
        # Write data to CSV
        writer.writerow([x_c, y_c, r_c])
    return
    
def print_manual_perimeter_points(output_file):
    """
    Displays the points chosen for validation.
    """
    
    with open(output_file, 'r', newline='') as f:
        reader = csv.reader(f)
        saved_points = list(reader)
    
    print("Saved Points:")
    for point in saved_points:
        print(', '.join(point))
        
def read_manual_perimeter_to_array(output_file):
    """
    Reads manually chosen points along the cell perimeter.
    """
    
    with open(output_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        points = np.array([[int(row[0]), int(row[1])] for row in reader])
    
    return points        

# Bresenham's line algorithm to find the pixels on the line
def bresenham(x1, y1, x2, y2):
    """
    Implementation of the Bresenham's line algorithm to find the pixel coordinates
    matching the drawn line.
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points


def fit_circle(points):
    """
    Fits a circle using the manually determined points along the cell perimeter.
    """
    # Calculate the initial guess for circle center and radius
    x = points[:, 0]
    y = points[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center, ier = leastsq(f_2, center_estimate)
    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    return xc, yc, R

def plot_detected_circle_from_manual_points(image, xc, yc, rc, output_path):
    """
    Plots the detected circle found in the fit_circle function using the 
    manually determined points along the cell perimeter.
    """
    fig, ax = plt.subplots()
    ax.imshow(image)
    circle = plt.Circle((xc, yc), rc, color='r', fill=False, linewidth=1)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    plt.title("Manual size detection")
    plt.axis('off')
    plt.savefig(output_path + "manual_size_detection.png", dpi=300)
    plt.show()





def mask_ring(image, x_c, y_c, r_in, r_out):
    """
    Creates a mask in the shape of a ring isolating the cilary band of the 
    detected Didinium cell.
    """
    mask_in = np.zeros_like(image[0,:,:])
    mask_out = np.zeros_like(image[0,:,:])

    rows_in, columns_in = disk((y_c, x_c), r_in)
    mask_in[rows_in, columns_in] = 1

    rows_out, columns_out = disk((y_c, x_c), r_out)
    mask_out[rows_out, columns_out] = 1

    mask_ring = mask_out - mask_in
    
    return mask_ring



def polar_transform(image, x_c, y_c, r_c):
    """
    Applies a polar transform on a time-lapse using a given central point.
    """
    num_frames = image.shape[0]
    range_of_transform = 330
    initialRadius = int(0)
    finalRadius = int(range_of_transform)
    
    angleSize = int(2 *np.pi *r_c)
    
    polarImage, ptSettings = polarTransform.convertToPolarImage(image[0], center =(y_c, x_c), initialRadius=initialRadius,
                                                            finalRadius=finalRadius, initialAngle=0 * np.pi,
                                                            finalAngle=2 * np.pi, hasColor=True, angleSize=angleSize)
    m, n = polarImage.shape
    num_frames   = image.shape[0]
    polarImgs = np.zeros((num_frames, m, n))
    polarImgs[0,:,:] = polarImage

    for i in tqdm(range(1, num_frames)):
        polarImgs[i,:,:], ptSettings = polarTransform.convertToPolarImage(image[i], center =(y_c, x_c), initialRadius=initialRadius,
                                                            finalRadius=finalRadius, initialAngle=0 * np.pi,
                                                            finalAngle=2 * np.pi, hasColor=True, angleSize=angleSize)
    polar_trans_image = polarImgs
    
    return polar_trans_image



# def kymograph_display(x_new):
#     """
#     Plots a given kymograph.
#     """
     
#     plt.figure()
#     plt.imshow(x_new, aspect = 1/5, origin = 'lower')
#     plt.gca().invert_yaxis()
#     plt.title('Kymograph of image intensity', fontsize = 10) 
#     plt.xlabel(r'$\tilde{x}$'' (pixel)', fontsize = 10)
#     plt.ylabel('t (frames)', fontsize = 10)
    
#     return

def kymograph_display_units(c_intensity):
    """
    Plots a given kymograph with rescaled axes in μm and sec. 
    c_intensity should be rescaled beforehand.
    """
     
    plt.figure()
    plt.imshow(c_intensity,  cmap='gray', aspect = 500, origin = 'lower', extent=[c_intensity.columns[0], c_intensity.columns[-1], c_intensity.index[0], c_intensity.index[-1]])
    plt.title('Kymograph of image intensity', fontsize = 13)
    plt.gca().invert_yaxis()
    plt.xlabel('dx (\u03BCm)', fontsize = 10)
    plt.ylabel('dt (sec)', fontsize = 10)

    return



def autocorrelation2D_fft_units(c_intensity):
    """
    Calculates the autocorelation in time and space using the Wiener–Khinchin theorem.
    """
    c_intensity = c_intensity - c_intensity.values.mean()
    
    f = fft2(c_intensity)
    c = np.real(ifft2(f * np.conj(f)))
 
    c = c / c[0,0]
    c = c[:c.shape[0]//2, :c.shape[1]//2]
    
    return c



def autocorrelation2D_visualization_units(c):
    """
    Displays the autocorrelation in space (μm) and time (sec).
    Aspect ratio is adjusted considering that usual fps is in the magnitude of 1000.
    
    """
    
    plt.figure()
    plt.imshow(c, aspect = 1000/1, origin = 'lower', extent=[c.columns[0], c.columns[-1], c.index[0], c.index[-1]], cmap = mpl.colormaps.get_cmap('jet'))
    plt.colorbar()
    plt.title('2D Autocorrelation', fontsize = 13)
    plt.gca().invert_yaxis()
    plt.xlabel('dx (\u03BCm)', fontsize = 10)
    plt.ylabel('dt (sec)', fontsize = 10)

    return

def vertical_artifacts_correction(kymograph, filter_size, output_path):
    """
    Applies median filtering along the t-axis to identify vertical line artifacts,
    and subtracts the filtered image from the original to obtain a corrected kymograph.

    """

    filtered_kymograph = median_filter(kymograph, size=(filter_size, 1))
    corrected_kymograph = kymograph - filtered_kymograph

    # Plot original, filtered, and corrected images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(kymograph, cmap='gray', aspect='auto')
    ax1.set_title('Original Kymograph')

    ax2.imshow(filtered_kymograph, cmap='gray', aspect='auto')
    ax2.set_title(f'Vertical artifacts')

    ax3.imshow(corrected_kymograph, cmap='gray', aspect='auto')
    ax3.set_title('Corrected Kymograph')

    plt.tight_layout()
    plt.savefig(output_path + "filtered_kymograph.png")
    plt.show()

    corrected_kymograph_min  = np.min(corrected_kymograph)
    corrected_kymograph_max  = np.max(corrected_kymograph)
    corrected_kymograph_norm = (corrected_kymograph - corrected_kymograph_min) / (corrected_kymograph_max - corrected_kymograph_min)

    pd.DataFrame(corrected_kymograph_norm).to_pickle(output_path + "corrected_kymograph_norm.pkl")
    return corrected_kymograph
    

def output_CBF_average(correlation, temporal_sampling_freq):
    """
    Calculation of the CBF using the autocorrelation pattern by averaging over all lags in space.
    Errors are estimated by the full width at half maximum (FWHM).
    """
    
    num_frames = correlation.shape[0]
    num_dx = correlation.shape[1]

    # For space_lag = 0
    space_lag = 0
    frequencies_0, psd_t_0 = signal.welch(correlation.iloc[:, space_lag], fs=temporal_sampling_freq, nperseg=num_frames)

    size_psd = psd_t_0.shape[0]
    
    # For all space lags
    frequencies_all = np.zeros((num_dx, size_psd))
    psd_t_all = np.zeros((num_dx, size_psd))

    frequencies_all[0,:] = frequencies_0
    psd_t_all[0,:] = psd_t_0
    
    # Loop over space_lag values from 1 to num_dx - 1
    for space_lag in range(1, num_dx):

        frequencies, psd_t = signal.welch(correlation.iloc[:, space_lag], fs=temporal_sampling_freq, nperseg=num_frames)

        frequencies_all[space_lag, :] = frequencies
        psd_t_all[space_lag, :] = psd_t
    
    # Average along the rows
    average_frequencies = np.mean(frequencies_all, axis=0)
    average_psd_t = np.mean(psd_t_all, axis=0)
    
    # std along the rows
    std_frequencies = np.std(frequencies_all, axis=0)
    std_psd_t = np.std(psd_t_all, axis=0)
    
    # normalization:
    average_psd_t = average_psd_t / np.max(average_psd_t)
    std_psd_t     = std_psd_t / np.max(average_psd_t)
    
    peaks, _ = signal.find_peaks(average_psd_t)
    
    # identify the primary frequency that should correspond to the CBF:
    sorted_peaks = peaks[np.argsort(average_psd_t[peaks])][::-1]
    primary_peak = sorted_peaks[1] if sorted_peaks[0] < 0 else sorted_peaks[0] 
    CBF = average_frequencies[primary_peak]
    
    # Calculate FWHM
    half_max = average_psd_t[primary_peak] / 2
    left_idx = np.where(average_psd_t[:primary_peak] <= half_max)[0] 
    right_idx = np.where(average_psd_t[primary_peak:] <= half_max)[0]

    if len(left_idx) > 0 and len(right_idx) > 0:
        left_half_max = average_frequencies[left_idx[-1]]
        right_half_max = average_frequencies[primary_peak + right_idx[0]]
        fwhm = right_half_max - left_half_max
        CBF_error = fwhm / 2 
    else:
        CBF_error = 0  # Default to 0 if FWHM cannot be determined
    
    return average_frequencies, average_psd_t,std_frequencies, std_psd_t, CBF, CBF_error
    



def plot_CBF(frequencies, psd_t_norm, psd_t_norm_std, CBF, CBF_error):
    """
    Plots the CBF on the PSD plot.
    """
    
    # Plot the power spectral density
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, psd_t_norm)
    plt.fill_between(frequencies, psd_t_norm - psd_t_norm_std, psd_t_norm + psd_t_norm_std, alpha=0.5)
    plt.axvline(CBF, color='r', linestyle='--', label=f'CBF: {CBF:.2f} Hz')
    plt.axvline(CBF - CBF_error, color='g', linestyle='--', label=f'CBF error: {CBF_error:.2f} Hz')
    plt.axvline(CBF + CBF_error, color='g', linestyle='--')
    
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(False)
    
    return



def output_wavelength_average(correlation, spatial_sampling_freq):
    """
    Calculation of the wavelength using the autocorrelation pattern by averaging over all lags in time.
    Errors are estimated by the full width at half maximum (FWHM).
    """
    
    N = correlation.shape[1]
    num_dt = correlation.shape[0]
    
    # For time_lag = 0
    time_lag = 0
    wavenumbers_0, psd_x_0 = signal.welch(correlation.iloc[time_lag, :], fs=spatial_sampling_freq, nperseg=N)
    
    size_psd = psd_x_0.shape[0]
    
    wavenumbers_all = np.zeros((num_dt, size_psd))
    psd_x_all = np.zeros((num_dt, size_psd))

    wavenumbers_all[0,:] = wavenumbers_0
    psd_x_all[0,:] = psd_x_0
    
    for time_lag in range(1, num_dt):
        wavenumbers, psd_x = signal.welch(correlation.iloc[time_lag, :], fs=spatial_sampling_freq, nperseg=N)

        wavenumbers_all[time_lag, :] = wavenumbers
        psd_x_all[time_lag, :] = psd_x
    
    # Average along the rows
    average_wavenumbers = np.mean(wavenumbers_all, axis=0)
    average_psd_x = np.mean(psd_x_all, axis=0)

    # std along the rows
    std_wavenumbers = np.std(wavenumbers_all, axis=0)
    std_psd_x = np.std(psd_x_all, axis=0)
    
    # normalization:
    average_psd_x = average_psd_x / np.mean(average_psd_x)
    std_psd_x     = std_psd_x /  np.mean(average_psd_x)
    
    peaks, _ = signal.find_peaks(average_psd_x)
    sorted_peaks = peaks[np.argsort(average_psd_x[peaks])][::-1]
    primary_peak = sorted_peaks[1] if sorted_peaks[0] <= 5 else sorted_peaks[0]
    primary_wavenumber = average_wavenumbers[primary_peak]
    
    # Calculate FWHM
    half_max = average_psd_x[primary_peak] / 2
    left_idx = np.where(average_psd_x[:primary_peak] <= half_max)[0] 
    right_idx = np.where(average_psd_x[primary_peak:] <= half_max)[0]
    
    if len(left_idx) > 0 and len(right_idx) > 0:
        left_half_max = average_wavenumbers[left_idx[-1]]
        right_half_max = average_wavenumbers[primary_peak + right_idx[0]]
        fwhm = right_half_max - left_half_max
        wavenumber_error = fwhm / 2  
    else:
        wavenumber_error = 0  # Default to 0 if FWHM cannot be determined
    
    wavelength = 1 / primary_wavenumber
    wavelength_error = (1 / primary_wavenumber)**2 * wavenumber_error
    
    return average_wavenumbers, average_psd_x, std_wavenumbers, std_psd_x, wavelength, wavelength_error, primary_wavenumber, wavenumber_error 



def plot_wavelength(wavenumbers, psd_x_norm, psd_x_norm_std, primary_wavenumber, wavenumber_error,wavelength, wavelength_error):
    """
    Plots the wavelength on the PSD plot.
    """
    
    # Plot the power spectral density
    plt.figure(figsize=(10, 5))
    plt.plot(wavenumbers, psd_x_norm)
    plt.fill_between(wavenumbers, psd_x_norm - psd_x_norm_std, psd_x_norm + psd_x_norm_std, alpha=0.5)
    
    plt.axvline(primary_wavenumber, color='r', linestyle='--', label=f'Wavelength: {wavelength:.2f} \u03BCm')
    plt.axvline(primary_wavenumber - wavenumber_error, color='g', linestyle='--', label=f'Wavelength error: {wavelength_error:.2f}  \u03BCm')
    plt.axvline(primary_wavenumber + wavenumber_error, color='g', linestyle='--')
    
    plt.title('Power Spectral Density')
    plt.xlabel('Wavenumber (1/\u03BCm)')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(False)
    
    return



def velocity_visualization(correlation, velocity, fps, pixel_size, wavelength):
    """
    Quick visualization of the slope found with FT of autocorrelation for visual inspection.
    """
    t_dim = correlation.shape[0]   # frames
    x_dim = correlation.shape[1]   # pixels
    
    x= np.arange(0, x_dim*pixel_size)
    c = correlation
    
    plt.figure()
    plt.title("v = " + str(round(velocity,3)) + " \u03BCm/sec")
    plt.imshow(c, aspect = 1000/1, origin = 'lower', extent=[c.columns[0], c.columns[-1], c.index[0], c.index[-1]], cmap = mpl.colormaps.get_cmap('jet'))
    plt.gca().invert_yaxis()
    
    num_lines = 20
    line_spacing = wavelength  # in micrometers
    
    # To stay in the borders  autocorrelation plot:
    threshold_above = t_dim / fps
    threshold_below = 0
    
    for i in range(1, num_lines + 1):
        offset = i * line_spacing
        line = 1 / velocity * (x - offset)
        
        # Crop the lines to fit within the image boundaries
        mask = (line <= threshold_above) & (line >= threshold_below)
        x_cropped = x[mask]
        line_cropped = line[mask]
            
        plt.plot(x_cropped, line_cropped, c="gold", linewidth = 0.8 , linestyle= "--")  
    
    return


def velocity_by_aft(correlation, pixel_size, fps, num_windows, output_path, sigma=5.0, overlap = 1,intensity_thresh = 0.8, eccentricity_thresh = 0.4, visualize_results = True):
    """ 
    Alignment by Fourier Transform (AFT):
    The followign function calculates the velocity of the MW by averaging over the
    local slopes detected in the 2D autocorrelation by means of a window search.
    Windows with intensities below an input threshold are omitted to avoid noise. 
    Note: follows the approach of: https://github.com/OakesLab/AFT-Alignment_by_Fourier_Transform
    """
    im_aft = correlation.to_numpy()
    im_aft = ndimage.gaussian_filter(im_aft, sigma=sigma)
    N_aft_rows, N_aft_cols = im_aft.shape
    window_size = int(min(N_aft_rows, N_aft_cols)/num_windows)
    
    # make window size odd if it isn't already
    if window_size % 2 == 0:
        window_size += 1
    # define the radius of the window
    radius = int(np.floor((window_size) / 2))

    # make a list of the rows and columns positions for the windows:
    rpos = np.arange(radius,N_aft_rows-radius,int(window_size * overlap)) # for overlap = 1 : no overlap
    cpos = np.arange(radius,N_aft_cols-radius,int(window_size * overlap))

    # make a structuring element to filter the mask
    bpass_filter = disk_vel(radius * .5)
    # make window mask
    window_mask = np.zeros((window_size, window_size))
    window_mask[int(np.floor(window_size/2)), int(np.floor(window_size/2))] = 1

    # filter the mask with the structuring element to define the ROI
    window_mask = cv2.filter2D(window_mask, -1, bpass_filter)
    window_mask = np.rint(window_mask) == 1

    im_mask = np.ones_like(im_aft).astype('bool')
    
    # make x and y coordinate matrices (for window)
    xcoords, ycoords = np.meshgrid(np.arange(0,window_size) , np.arange(0,window_size))
    # length of displaying orientation vector
    arrow_length = radius / 2
    
    # make empty arrays
    im_theta = np.array([])
    im_ecc = np.array([])
    x, y = [], []
    u = np.array([])
    v = np.array([])
    
    for r in rpos:
        for c in cpos:        
            # store the row and column positions
            x.append(c)
            y.append(r)
            # check to see if point is within image mask
            if im_mask[r,c] == True:
                # define the window to analyze
                im_window = im_aft[r-radius:r+radius+1,c-radius:c+radius+1]
                # check that it's above the intensity threshold
                if np.mean(im_window) > intensity_thresh:
                    #### periodic decomposition:
                    im_window = im_window.astype('float32')
                    # find the number of rows and cols
                    N_rows, N_cols = im_window.shape
                    # create an zero matrix the size of the image
                    fix_borders = np.zeros((N_rows,N_cols))
                    # deal with edge discontinuities
                    fix_borders[0,:] = im_window[0,:] - im_window[-1,:]
                    fix_borders[-1,:] = -fix_borders[0,:]
                    fix_borders[:,0] = fix_borders[:,0] + im_window[:,0] - im_window[:,-1]
                    fix_borders[:,-1] = fix_borders[:,-1] - im_window[:,0] + im_window[:,-1]
                    # calculate the frequencies of the image
                    fx = matlib.repmat(np.cos(2 * np.pi * np.arange(0,N_cols) / N_cols),N_rows,1)
                    fy = matlib.repmat(np.cos(2 * np.pi * np.arange(0,N_rows) / N_rows),N_cols,1).T
                    # set the fx[0,0] to 0 to avoid division by zero
                    fx[0,0] = 0
                    ## calculate the smoothed image component
                    # filter applied in frequency domain to emphasize low-frequency components 
                    # and de-emphasizing high-frequency component
                    im_window_smooth = np.real(ifft2(fft2(fix_borders) * 0.5 / (2 - fx - fy)))
                    im_window_periodic = im_window - im_window_smooth
                    
                    # take the FFT of the periodic component
                    im_window_fft = fftshift(fft2(im_window_periodic))
                    
                    # find the image norm and mulitply by the mask
                    im_window_norm = np.sqrt(np.real(im_window_fft * np.conj(im_window_fft)))
                    im_window_fft_norm = im_window_norm * window_mask
                    N_rows, N_cols = im_window_fft_norm.shape
                    
                    #calculate the moments
                    # zero-order moment represents the total intensity (or mass) of the image
                    M00 = np.sum(im_window_fft_norm)
                    # first-order moments are used to find the centroid (center of mass)
                    # weighted sum of the pixel coordinates
                    M10 = np.sum(im_window_fft_norm * xcoords)
                    M01 = np.sum(im_window_fft_norm * ycoords)
                    # second-order moments: spread and orientation of the image
                    # correlation between x and y
                    M11 = np.sum(im_window_fft_norm * xcoords * ycoords)
                    # variance in x direction
                    M20 = np.sum(im_window_fft_norm * xcoords * xcoords)
                    # variance in y direction
                    M02 = np.sum(im_window_fft_norm * ycoords * ycoords)
                    
                    # center of mass
                    xave = M10 / M00
                    yave = M01 / M00
    
                    # calculate the central moments
                    mu20 = M20/M00 - xave**2
                    mu02 = M02/M00 - yave**2
                    mu11 = M11/M00 - xave*yave
                    
                    # angle of axis
                    theta = 0.5 * np.arctan2((2 * mu11),(mu20 - mu02))
    
                    # multiply by -1 to correct for origin being in top left corner instead of bottom right
                    theta = -1 * theta
                    
                    # find eigenvectors of covariance matrix
                    lambda1 = (0.5 * (mu20 + mu02)) + (0.5 * np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
                    lambda2 = (0.5 * (mu20 + mu02)) - (0.5 * np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
        
                    # calculate the eccentricity (e.g. how oblong it is)
                    eccentricity = np.sqrt(1 - lambda2/lambda1)
                    
                    # correct for real space
                    theta = theta + np.pi/2
    
                    # map everything back to between -pi/2 and pi/2
                    if theta > np.pi/2:
                        theta -= np.pi
                        
                    # filter based on eccentricity
                    if eccentricity < eccentricity_thresh:
                        eccentricity = np.nan
                        theta = np.nan
    
                    # add the values 
                    im_theta = np.append(im_theta, theta)
                    im_ecc = np.append(im_ecc, eccentricity)
                    u = np.append(u, np.cos(theta) * arrow_length)
                    v = np.append(v, np.sin(theta) * arrow_length)
                else:
                    im_theta = np.append(im_theta, np.nan)
                    im_ecc = np.append(im_ecc, np.nan)
                    u = np.append(u, np.nan)
                    v = np.append(v, np.nan)
    x = np.array(x)
    y = np.array(y)        
    im_theta = np.reshape(im_theta,(len(rpos),len(cpos)))
    im_ecc = np.reshape(im_ecc,(len(rpos),len(cpos)))
    
    velocity = 1/np.tan(np.nanmean(im_theta)) * pixel_size * fps
    velocity_error = (1/(np.nanmean(im_theta ))**2 )*(np.nanstd(im_theta))* pixel_size * fps 
    
    # optional visualization:
    if visualize_results == True:
        plt.figure()
        plt.imshow(im_aft, cmap='Greys_r')
        plt.quiver(x,y,u,v, color='yellow', pivot='mid', scale_units='xy', scale=overlap/2, headaxislength=0, headlength=0, width=0.005)
        plt.title('Overlay')
        plt.show()
        print("velocity (\u03BCm/sec)          : ", velocity)
        print("error in velocity (\u03BCm/sec) : ", velocity_error)

    return velocity, velocity_error



def save_csv_file(output_path, headers, data):
    """
    Saves fileparameters or results in csv file.
    """
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(data)
    return

