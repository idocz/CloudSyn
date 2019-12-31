import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct
import scipy
import os
from skimage import io as img

def vol3d_wrapper(data):
    path = '/home/idocz/repo/CloudSyn/vol3d/'
    data_dict = {"beta" : data}
    scipy.io.savemat(f"{path}data.mat", data_dict)


def vol2dict(volfile_path):
    vol_dict = {}
    with open(volfile_path, "rb") as f:

        vol_dict['header'] = f.read(3)
        vol_dict['file_format'] = f.read(1)
        vol_dict['identifier'] = int.from_bytes(f.read(4), "little")
        vol_dict['X_size'] = int.from_bytes(f.read(4), "little")
        vol_dict['Y_size'] = int.from_bytes(f.read(4), "little")
        vol_dict['Z_size'] = int.from_bytes(f.read(4), "little")
        vol_dict['channels'] = int.from_bytes(f.read(4), "little")
        vol_dict['bounding_box'] = f.read(24)
        vol_dict['data'] = np.zeros((vol_dict['X_size'], vol_dict['Y_size']
                                     , vol_dict['Z_size'], vol_dict['channels']), dtype='float')

        for zpos in range(vol_dict['Z_size']):
            for ypos in range(vol_dict['Y_size']):
                for xpos in range(vol_dict['X_size']):
                    for chan in range(vol_dict['channels']):
                        vol_dict['data'][xpos, ypos, zpos, chan] = struct.unpack('f', f.read(4))[0]

        return vol_dict


def animate(image_list):
    ims = []
    fig = plt.figure()
    for image in image_list:
        im = plt.imshow(image, animated=True, cmap='gray')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=10)
    plt.show()

def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def show_salGAN_samples(sample_dir, images_range=None):
    dir_list = os.listdir(sample_dir)
    if images_range is None:
        images_range = range(len(dir_list))
    images_list = [f"{num}.png" for num in images_range]
    assert set(images_list).issubset(set(dir_list)), "images range is invalid"

    N_grid = int(np.ceil(np.sqrt(len(images_list))))
    for i, image_name in enumerate(images_list):
        ax = plt.subplot(N_grid, N_grid,i+1)
        image = img.imread(f"{sample_dir}/{image_name}")
        ax.imshow(image)
