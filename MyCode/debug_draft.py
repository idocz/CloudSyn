import numpy as np
from scipy.io import loadmat
from utils import *
import matplotlib.pyplot as plt
from skimage import io as img
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
sys.path.append("/home/idocz/repo/CloudSyn/MyCode/")

data = loadmat("data/clouds_dist.mat")['beta']
from SinGAN_wrapper import *


X_size = data.shape[0]
Y_size = data.shape[1]
Z_size = data.shape[2]
animate = False
if animate:
    image_list = [data[:, :, i] for i in range(Z_size)]
    animate(image_list)
    plt.show()
np.save("/home/idocz/repo/CloudSyn/SinGAN3D/Input/Images/cloud", data)

Cloud_wrap = salGAN_wrapper("cloud.npy", load_existing_model=False)
Cloud_wrap.train()
