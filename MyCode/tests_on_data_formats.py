import numpy as np
from scipy.io import loadmat
from utils import *
import matplotlib.pyplot as plt
from skimage import io as img
from mpl_toolkits.mplot3d import Axes3D
import torch

to_save = True
to_mat = False

vol_plot = False
animation_plot = True
hist_plot = False

from_singan = False

if from_singan:
    file_name = "cloud6.npy"
    scale = 0
    index = 10
    data = np.load(f"/home/idocz/repo/CloudSyn/SinGAN-master/Output/RandomSamples/{file_name[:-4]}/gen_start_scale={scale}/{index}.npy")
else:
    data = loadmat("data/clouds_dist.mat")['beta']
    zero_pad = 10
    data = data[:,:,27-zero_pad:112+zero_pad]
# #
# vol_dict = vol2dict(("data/small_cloud.vol"))
# data = vol_dict["data"][:,:,:,0]
# X_size = vol_dict['X_size']
# Y_size = vol_dict['Y_size']
# Z_size = vol_dict['Z_size']

X_size = data.shape[0]
Y_size = data.shape[1]
Z_size = data.shape[2]
#
# for z in range(Z_size):
#     value = np.mean(data[:, :, z])
#     if value < 0.51 and value > 0.49 :
#         data[:, :, z] = 0

# data[data<0.6] = 0

zeros = []
for z in range(Z_size):
    if (data[:,:,z] == 0).all():
        zeros.append(z)
print(zeros)




if hist_plot:
    plt.hist(data.reshape(X_size*Y_size*Z_size),bins=20)
    plt.show()


if vol_plot:

    x_pos = []
    y_pos = []
    z_pos = []
    colors = []
    for x in range(X_size):
        for y in range(Y_size):
            for z in range(Z_size):
                if data[x, y, z] != 0.0 and data[x, y, z] != 0.5:
                    x_pos.append(x)
                    y_pos.append(y)
                    z_pos.append(z)
                    colors.append(data[x, y, z])

    colors_array = np.zeros((len(colors), 4))
    white_intensity = 3/4
    colors_array[:, 0] = white_intensity
    colors_array[:, 1] = white_intensity
    colors_array[:, 2] = white_intensity
    colors_array[:, 3] = np.array(colors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    img = ax.scatter(x_pos, y_pos, z_pos, c=colors, alpha = 0.3)
    set_axes_equal(ax)
    fig.colorbar(img)
    plt.show()


if animation_plot:
    image_list = [data[:,:,i] for i in range(Z_size)]
    animate(image_list)
    plt.show()


# img.imsave("cloud.jpg", data)
print(data.shape)
if to_save:
    np.save("/home/idocz/repo/CloudSyn/SinGAN-master/Input/Images/cloud7",data)

if to_mat:
    vol3d_wrapper(data)
print("end")


