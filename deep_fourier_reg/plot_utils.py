import cv2
import numpy as np
import flow_vis
import matplotlib.pyplot as plt

def dft_amplitude(flow):
    complex_flow = flow[0, :, :] + 1j * flow[1, :, :]
    fft = np.fft.fftshift(np.fft.fft2(complex_flow))
    amplitude = np.abs(fft)
    log_amp = np.log(1 + amplitude)
    C = log_amp.max()
    return (255 / C) * log_amp

def flow_visualization(flow):
    hsv = np.zeros(flow.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 1] = 180

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print(ang.min(), ang.max())
    hsv[..., 0] = ang * 89.5 / np.pi
    norm_mag = cv2.normalize(mag, None, 20, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = norm_mag
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def flow_colorized(flow):
    # flow_color = flow_vis.flow_to_color(flow.transpose(1, 2, 0)).astype('float32')
    flow_color = flow_visualization(flow.transpose(1, 2, 0))
    return flow_color

def plotter(image, flow):
    while image.shape[0] == 1:
        image = np.squeeze(image, 0)
    while flow.shape[0] == 1:
        flow = np.squeeze(flow, 0)
    img_color = 255 * cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    flow_color = flow_colorized(flow)
    dft_amp = cv2.cvtColor(dft_amplitude(flow).astype('float32'), cv2.COLOR_GRAY2RGB)
    sep = 255 * np.ones((int(image.shape[-2] * 0.01), image.shape[-2], 3))
    plot = np.vstack([np.rot90(img_color, -1), sep, np.rot90(flow_color, -1), sep, np.rot90(dft_amp, -1)])
    return plot