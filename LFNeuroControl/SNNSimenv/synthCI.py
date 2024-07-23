import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
from tqdm import tqdm
import pdb


# Constants
frame_size = 280
neuron_diameter_pixels = 12  # Adjusted for scaling
radius = neuron_diameter_pixels // 2
gaussian_sigma = 1
optical_depth_decay_factor = 0.01
decay_factor_per_ms = 0.5  # Exponential decay factor per millisecond

def group_spikes_camera_fps_adjust(spikes, fps):
    if fps >= 1000:
        return spikes

    ms_per_frame = int(1000/fps)
    first_index = 0
    for i in range(1, len(spikes['times'])):
        if spikes['times'][i] > spikes['times'][first_index] + ms_per_frame:
            spikes['times'][first_index:i] = spikes['times'][first_index]
            first_index = i
    
    return spikes


# def optical_properties(image, baseline_noise=5*10):
#     baseline_noise=1
#     blurred_image = gaussian_filter(image, sigma=gaussian_sigma)
#     global_noise = np.random.normal(loc=0, scale=baseline_noise, size=image.shape)
#     noisy_image = np.clip(blurred_image + global_noise, 0, 255)
#     noise = np.random.poisson(lam=noisy_image) - noisy_image
#     depth_effect = np.exp(-optical_depth_decay_factor * noisy_image)
#     brightness_factor = 2
#     brightness_term = 50
#     return np.clip((noisy_image * depth_effect * brightness_factor)+brightness_term, 0, 255)
def optical_properties(image, baseline_noise=5):
    baseline_noise*=10
    blurred_image = gaussian_filter(image, sigma=gaussian_sigma)
    global_noise = np.random.normal(loc=0, scale=baseline_noise, size=image.shape)
    noisy_image = np.clip(blurred_image + global_noise, 0, 255)
    noise = np.random.poisson(lam=noisy_image) - noisy_image
    depth_effect = np.exp(-optical_depth_decay_factor * noisy_image)
    brightness_factor = 2
    brightness_term = 50


    image = np.clip((noisy_image * depth_effect * brightness_factor)+brightness_term, 0, 255)
    blurred_image = gaussian_filter(image, sigma=gaussian_sigma)
    return blurred_image

def create_synth_frames(data, positions, total_time, frame_rate=1):
    time_step = 1.0 / frame_rate
    num_frames = int(total_time * frame_rate)
    frames = [np.zeros((frame_size, frame_size)) for _ in range(num_frames)]

    # Neuron representation as a circle
    structure = np.zeros((neuron_diameter_pixels, neuron_diameter_pixels))
    for i in range(neuron_diameter_pixels):
        for j in range(neuron_diameter_pixels):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                structure[i, j] = 1
    #pdb.set_trace()
    for i, time in enumerate(data['times']):# tqdm(enumerate(data['times']), total=len(data['times'])):
        frame_index = int(time // time_step)
        if frame_index < num_frames:
            neuron_index = data['senders'][i]
            x, y = positions[neuron_index-1]
            x_idx, y_idx = int(x * frame_size), int(y * frame_size)


            x_start, y_start = max(x_idx - radius, 0), max(y_idx - radius, 0)
            x_end, y_end = min(x_idx + radius + 1, frame_size), min(y_idx + radius + 1, frame_size)
            # Adjust indices for structure array when close to edges
            x_struct_start = max(0, radius - (x_idx - x_start))
            y_struct_start = max(0, radius - (y_idx - y_start))
            x_struct_end = neuron_diameter_pixels - max(0, (x_idx + radius + 1) - x_end)
            y_struct_end = neuron_diameter_pixels - max(0, (y_idx + radius + 1) - y_end)
            struct_slice = structure[x_struct_start:x_struct_end, y_struct_start:y_struct_end]


            for offset in range(10):  # Decay effect over multiple frames
                decay_frame_index = frame_index + offset
                if decay_frame_index < num_frames:
                    decay_multiplier = (1.0 - decay_factor_per_ms) ** offset

                    



                    frame_slice = frames[decay_frame_index][x_start:x_end, y_start:y_end]
                    frames[decay_frame_index][x_start:x_end, y_start:y_end][:struct_slice.shape[0], :struct_slice.shape[1]] += struct_slice * 255 * decay_multiplier
                    # frame_slice[:struct_slice.shape[0], :struct_slice.shape[1]] += struct_slice * 255 * decay_multiplier


    # Apply optical properties
    frames = [optical_properties(frame) for frame in (frames)]
    return [np.clip(frame, 0, 255).astype(np.uint8) for frame in (frames)]

def create_video(frames, filename='neuron_activity.mp4', fps=1):
    writer = imageio.get_writer(filename, fps=fps)
    for frame in (frames):
        writer.append_data(frame)
    writer.close()
