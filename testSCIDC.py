import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
from tqdm import tqdm

# Parameters
scale_factor = 4
frame_size = 480
neuron_diameter_pixels = 5 * scale_factor
radius = neuron_diameter_pixels // 2
gaussian_sigma = 1
optical_depth_decay_factor = 0.01
decay_factor_per_ms = 0.5
camera_fps = 280  # Camera frame rate
time_interval_ms = 1000 / camera_fps  # Time interval per frame in milliseconds

def optical_properties(image, baseline_noise=5 * 10):
    blurred_image = gaussian_filter(image, sigma=gaussian_sigma)
    global_noise = np.random.normal(loc=0, scale=baseline_noise, size=image.shape)
    noisy_image = np.clip(blurred_image + global_noise, 0, 255)
    noise = np.random.poisson(lam=noisy_image) - noisy_image
    depth_effect = np.exp(-optical_depth_decay_factor * noisy_image)
    brightness_factor = 2
    brightness_term = 50
    return np.clip((noisy_image * depth_effect * brightness_factor) + brightness_term, 0, 255)

def create_frames(data, positions, total_time, frame_rate):
    time_step = 1.0 / frame_rate
    num_frames = int(total_time * frame_rate)
    frames = [np.zeros((frame_size, frame_size)) for _ in range(num_frames)]

    # Bin events together based on the camera's frame rate
    bins = np.arange(0, total_time, time_interval_ms / 1000)
    binned_events = np.digitize(data['times'], bins)

    # Neuron representation as a circle
    structure = np.zeros((neuron_diameter_pixels, neuron_diameter_pixels))
    for i in range(neuron_diameter_pixels):
        for j in range(neuron_diameter_pixels):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                structure[i, j] = 1

    for bin_index in tqdm(range(len(bins) - 1)):
        frame_index = bin_index
        if frame_index >= num_frames:
            break

        event_indices = np.where(binned_events == bin_index + 1)[0]
        for event_index in event_indices:
            neuron_index = data['senders'][event_index]
            x, y = positions[neuron_index]
            x_idx, y_idx = int(x * frame_size), int(y * frame_size)

            for offset in range(10):  # Decay effect over multiple frames
                decay_frame_index = frame_index + offset
                if decay_frame_index < num_frames:
                    decay_multiplier = (1.0 - decay_factor_per_ms) ** offset
                    x_start, y_start = max(x_idx - radius, 0), max(y_idx - radius, 0)
                    x_end, y_end = min(x_idx + radius + 1, frame_size), min(y_idx + radius + 1, frame_size)

                    # Adjust indices for structure array when close to edges
                    x_struct_start = max(0, radius - (x_idx - x_start))
                    y_struct_start = max(0, radius - (y_idx - y_start))
                    x_struct_end = neuron_diameter_pixels - max(0, (x_idx + radius + 1) - x_end)
                    y_struct_end = neuron_diameter_pixels - max(0, (y_idx + radius + 1) - y_end)

                    frame_slice = frames[decay_frame_index][x_start:x_end, y_start:y_end]
                    struct_slice = structure[x_struct_start:x_struct_end, y_struct_start:y_struct_end]
                    frame_slice[:struct_slice.shape[0], :struct_slice.shape[1]] += struct_slice * 255 * decay_multiplier

    # Apply optical properties
    frames = [optical_properties(frame) for frame in tqdm(frames)]
    return [np.clip(frame, 0, 255).astype(np.uint8) for frame in tqdm(frames)]

def create_video(frames, filename='neuron_activity.mp4'):
    writer = imageio.get_writer(filename, fps=camera_fps)
    for frame in tqdm(frames):
        writer.append_data(frame)
    writer.close()

# Example data and execution
num_neurons = 128
num_events = 1000
time = 100  # Total time in seconds
data = {
    'senders': np.random.randint(0, num_neurons, size=num_events),
    'times': np.sort(np.random.uniform(0, time, size=num_events))
}
positions = np.random.rand(num_neurons, 2)
frames = create_frames(data, positions, total_time=time, frame_rate=camera_fps)
create_video(frames)
