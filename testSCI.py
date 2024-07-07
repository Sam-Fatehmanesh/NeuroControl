import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import imageio
from tqdm import tqdm

# Parameters
frame_size = 480
neuron_diameter_pixels = 5
radius = neuron_diameter_pixels // 2
gaussian_sigma = 1
decay_factor = 0.01

# def optical_properties(image):
#     blurred_image = gaussian_filter(image, sigma=gaussian_sigma)
#     enhanced_noise_factor = 10  # Increase this factor to increase noise level
#     noise = np.random.poisson(lam=blurred_image * 10) - blurred_image
#     noisy_blurred_image = blurred_image + noise
#     depth_effect = np.exp(-decay_factor * noisy_blurred_image)
#     return noisy_blurred_image * depth_effect

def optical_properties(image, baseline_noise=5):
    # Apply Gaussian blurring for light scattering simulation
    blurred_image = gaussian_filter(image, sigma=1)
    
    # Introduce global Gaussian noise as baseline noise
    global_noise = np.abs(np.random.normal(loc=0, scale=baseline_noise, size=image.shape))
    
    # Adding photon shot noise
    noise = np.random.poisson(lam=np.clip(blurred_image + global_noise, 0, 255)) - blurred_image
    
    # Combine blurred image with noise and simulate depth effect
    noisy_blurred_image = blurred_image + noise
    depth_effect = np.exp(-0.01 * noisy_blurred_image)
    
    brightness_factor = 2
    return np.clip((noisy_blurred_image * depth_effect)*brightness_factor, 0, 255)

def create_frames(data, positions, total_time, frame_rate):
    time_step = 1.0 / frame_rate
    num_frames = int(total_time * frame_rate)
    frames = []

    # Neuron representation as a circle
    structure = np.zeros((neuron_diameter_pixels, neuron_diameter_pixels))
    for i in range(neuron_diameter_pixels):
        for j in range(neuron_diameter_pixels):
            if (i-radius)**2 + (j-radius)**2 <= radius**2:
                structure[i, j] = 1

    for _ in range(num_frames):
        frames.append(np.zeros((frame_size, frame_size)))  # Initialize frames at target resolution

    for i, time in tqdm(enumerate(data['times']), total=len(data['times'])):
        frame_index = int(time // time_step)
        if frame_index < num_frames:
            neuron_index = data['senders'][i]
            x, y = positions[neuron_index]
            x_idx, y_idx = int(x * frame_size), int(y * frame_size)

            # Boundary checks and placement adjustments
            x_start = max(x_idx - radius, 0)
            y_start = max(y_idx - radius, 0)
            x_end = min(x_idx + radius + 1, frame_size)
            y_end = min(y_idx + radius + 1, frame_size)

            x_struct_start = max(0, radius - (x_idx - x_start))
            y_struct_start = max(0, radius - (y_idx - y_start))
            x_struct_end = neuron_diameter_pixels - max(0, (x_idx + radius + 1) - x_end)
            y_struct_end = neuron_diameter_pixels - max(0, (y_idx + radius + 1) - y_end)

            frames[frame_index][x_start:x_end, y_start:y_end] = structure[x_struct_start:x_struct_end, y_struct_start:y_struct_end] * 255

    frames = [optical_properties(frame) for frame in tqdm(frames)]
    return [np.clip(frame, 0, 255).astype(np.uint8) for frame in tqdm(frames)]

def create_video(frames, filename='neuron_activity.mp4'):
    writer = imageio.get_writer(filename, fps=1)
    for frame in tqdm(frames):
        writer.append_data(frame)
    writer.close()

# Generating data and positions
num_neurons = 128
num_events = 1000
time = 100
data = {
    'senders': np.random.randint(0, num_neurons, size=num_events),
    'times': np.sort(np.random.uniform(0, time, size=num_events))
}
positions = np.random.rand(num_neurons, 2)

frames = create_frames(data, positions, total_time=time, frame_rate=1)
create_video(frames)
