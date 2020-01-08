import argparse
import torch
import random


class Config(object):
    def __init__(self, input_name, input_dir='Input/Images', mode='train', not_cuda=0, netG='', netD='',
                 manualSeed=None, nc_z=3, nc_im=3, out='Output', nfc=32, min_nfc=32, ker_size=3, num_layer=5, stride=1,
                 padd_size=0, scale_factor=0.75, noise_amp=0.1, min_size=25, max_size=250, niter=2000, gamma=0.1,
                 lr_g=0.0005, lr_d=0.0005, beta1=0.5, Gsteps=3, Dsteps=3, lambda_grad=0.1, alpha=10):

        self.input_name = input_name
        self.input_dir = input_dir
        self.mode = mode
        self.not_cuda = not_cuda
        self.netG = netG
        self.netD = netD
        self.manualSeed = manualSeed
        self.nc_z = nc_z
        self.nc_im = nc_im
        self.out = out
        self.nfc = nfc
        self.min_nfc = min_nfc
        self.ker_size = ker_size
        self.num_layer = num_layer
        self.stride = stride
        self.not_cuda = not_cuda
        self.padd_size = padd_size
        self.scale_factor = scale_factor
        self.noise_amp = noise_amp
        self.min_size = min_size
        self.max_size = max_size
        self.niter = niter
        self.gamma = gamma
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.Gsteps = Gsteps
        self.Dsteps = Dsteps
        self.lambda_grad = lambda_grad
        self.alpha = alpha
        # init fixed parameters
        self.device = torch.device("cpu" if self.not_cuda else "cuda:0")
        self.niter_init = self.niter
        self.noise_amp_init = self.noise_amp
        self.nfc_init = self.nfc
        self.min_nfc_init = self.min_nfc
        self.scale_factor_init = self.scale_factor
        self.out_ = 'TrainedModels/%s/scale_factor=%f/' % (self.input_name[:-4], self.scale_factor)
        if self.mode == 'SR':
            self.alpha = 100

        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.manualSeed)
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        if torch.cuda.is_available() and self.not_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', help='task to be done', default='train')
    # workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)

    # load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
    parser.add_argument('--out', help='output folder', default='Output')

    # networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size',
                        default=0)  # math.floor(opt.ker_size/2)

    # pyramid parameters:
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor',
                        default=0.75)  # pow(0.5,1/6))
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=25)
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=250)

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)

    return parser
