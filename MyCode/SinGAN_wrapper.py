#import
import os
import sys
os.chdir("/home/idocz/repo/CloudSyn/SinGAN3D")
from config import Config
from SinGAN.manipulate import SinGAN_generate
from SinGAN.training import train
import SinGAN.functions as functions
from shutil import rmtree
import torch


class salGAN_wrapper:
    def __init__(self, input_name, load_existing_model):
        self.input_name = input_name
        self.load_existing_model = load_existing_model
        self.opt = Config(input_name)
        self.dir2save = functions.generate_dir2save(self.opt)
        self.real = functions.read_volume(self.opt)
        functions.adjust_scales2image(self.real, self.opt)
        dir_exists = os.path.exists(self.dir2save)
        assert (not load_existing_model) or dir_exists, "cannot find trained model"

        if load_existing_model:
            print("Trained model has been loaded (not really)")
            self.Gs = torch.load(f'{self.dir2save}/Gs.pth')
            self.Zs = torch.load(f'{self.dir2save}/Zs.pth')
            self.reals = torch.load(f'{self.dir2save}/reals.pth')
            self.NoiseAmp = torch.load(f'{self.dir2save}/NoiseAmp.pth')
            self.is_loaded = True
        else:
            if dir_exists:
                user_input = input("Trained model has been found, type \"yes\" to overwrite: ")
                assert user_input == 'yes', "train aborted"
                rmtree(self.dir2save)
                print("train directory has been deleted")

            try:
                os.makedirs(self.dir2save)
            except OSError:
                pass

    def train(self):
        self.Gs = []
        self.Zs = []
        self.reals = []
        self.NoiseAmp = []
        train(self.opt, self.Gs, self.Zs, self.reals, self.NoiseAmp,)
        self.is_loaded = True

    def sample(self, n_samples, scale=0):
        assert self.is_loaded, "Run the train function or load a trained model"
        self.sample_dir = SinGAN_generate(self.Gs, self.Zs, self.reals, self.NoiseAmp, self.opt, num_samples=n_samples,
                                          gen_start_scale=scale)




