import os
import sys

import torch
from PIL import Image
from torch import nn, optim
from torchvision import datasets
from tqdm import tqdm

from config import DATASETS_CONFIG
from model import Generator, Discriminator
from dataset_utils import gain_sample

"""
Training module, modify below training settings if needed
"""

# GPU settings
DEVICE = torch.device('mps')

# Model architecture settings
MAPPING_LAYER_NUM = 8
LATENT_DIM = 512
INPUT_DIM = 4
EPOCH = 5
Z_DIM = 512

# Training settings
DISCRIMINATOR_GENERATOR_RATIO = 1  # discriminator_generator_ratio
SHOW_LOSS_PER_SAMPLE = 200
STEP = 1  # Start training from (8 * 8) resolution ???
MAX_STEP = 7  # 4*4 -> 8*8 -> 16*16 -> 32*32 -> 64*64 -> 128*128 -> 256*256
# -> 512*512
LEARNING_RATE_DICT = {128: 0.0015, 256: 0.002}
BATCH_SIZE_DICT = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 16}
MINI_BATCH_SIZE = 8
# If set to true, model will train from existing checkpoint, otherwise it
# will train from scratch
CONTINUE_FROM_SAVED_MODEL = True


class StyleGANTraining:
    def __init__(self, dataset_n):
        self.generator = Generator(INPUT_DIM, LATENT_DIM, INPUT_DIM).to(DEVICE)
        self.discriminator = Discriminator().to(DEVICE)
        # Optimizers
        self.g_optim = optim.Adam([
            {'params': self.generator.convolutions.parameters(), 'lr': 0.001},
            {'params': self.generator.to_rgbs.parameters(), 'lr': 0.001}],
            lr=0.001, betas=(0.0, 0.99))
        self.g_optim.add_param_group(
            {'params': self.generator.fcs.parameters(), 'lr': 0.001 * 0.01,
             'mul': 0.01})
        # Initial training parameters
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=0.001,
                                  betas=(0.0, 0.99))
        self.dataset_config = DATASETS_CONFIG[dataset_n]
        self.dataset = datasets.ImageFolder(
            self.dataset_config['image_folder_path'])
        self.cur_epoch = 0
        self.step = STEP
        self.iteration = 0
        self.startpoint = 0
        self.used_sample = 0
        self.alpha = 0
        self.discriminator_loss_dict = [float('inf')]
        self.generator_loss_dict = [float('inf')]

        if CONTINUE_FROM_SAVED_MODEL:
            if os.path.exists('checkpoint/trained.pth'):
                print('Loading pre-trained model...')
                checkpoint = torch.load('checkpoint/trained.pth')
                self.generator.load_state_dict(checkpoint['generator'])
                self.discriminator.load_state_dict(checkpoint['discriminator'])
                self.g_optim.load_state_dict(checkpoint['g_optim'])
                self.d_optim.load_state_dict(checkpoint['d_optim'])
                (self.step, self.iteration, self.startpoint, self.used_sample,
                 self.alpha, self.cur_epoch) = checkpoint['parameters']
                self.discriminator_loss_dict = checkpoint.get(
                    'discriminator_loss_dict', [float('inf')])
                self.generator_loss_dict = checkpoint.get(
                    'generator_loss_dict', [float('inf')])
                print('Training will start from loaded model...')
            else:
                print(
                    'Cannot find pre-trained model, training will start from '
                    'scratch...')

    @staticmethod
    def set_grad_flag(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    @staticmethod
    def reset_learning_rate(optimizer, lr):
        for pam_group in optimizer.param_groups:
            mul = pam_group.get('mul', 1)
            pam_group['lr'] = lr * mul

    @staticmethod
    def save_img(tensor, iteration, epoch, index, sav_dir_path):
        grid = tensor[0]
        grid.clamp_(-1, 1).add_(1).div_(2)
        # Add 0.5 after normalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(
            'cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        os.makedirs(sav_dir_path, exist_ok=True)
        img.save(
            f'{sav_dir_path}epoch_{epoch}_iteration_{iteration}_ind_{index}.png')

    def train(self):
        """
        Function to start training
        """
        self.generator.train()
        self.discriminator.train()

        resolution = 4 * 2 ** self.step
        # length of dataset * number of epochs
        sample_per_res = self.dataset_config[
                             'dataset_length'] * EPOCH
        # total training images seen by network
        total_sample = sample_per_res * MAX_STEP
        dataset_one_round = False

        origin_loader = gain_sample(self.dataset,
                                    BATCH_SIZE_DICT.get(resolution,
                                                        MINI_BATCH_SIZE),
                                    resolution)
        data_loader = iter(origin_loader)

        self.reset_learning_rate(self.g_optim,
                                 LEARNING_RATE_DICT.get(resolution, 0.001))
        self.reset_learning_rate(self.d_optim,
                                 LEARNING_RATE_DICT.get(resolution, 0.001))
        progress_bar = tqdm(total=total_sample, initial=self.used_sample)

        # Train
        while self.used_sample < total_sample:
            self.iteration += 1
            self.alpha = min(1, self.alpha +
                             BATCH_SIZE_DICT.get(resolution,
                                                 MINI_BATCH_SIZE) / sample_per_res)

            if ((self.used_sample - self.startpoint) > sample_per_res
                    and self.step < MAX_STEP):
                self.step += 1
                self.alpha = 0
                self.startpoint = self.used_sample

                resolution = 4 * 2 ** self.step
                print('Training at next resolution', resolution)

                # Load dataloaders, set learning rates, and avoid possible
                # CUDA out of memory error
                del origin_loader
                del data_loader
                origin_loader = gain_sample(self.dataset,
                                            BATCH_SIZE_DICT.get(resolution,
                                                                MINI_BATCH_SIZE),
                                            resolution)
                data_loader = iter(origin_loader)
                self.reset_learning_rate(self.g_optim,
                                         LEARNING_RATE_DICT.get(resolution,
                                                                0.001))
                self.reset_learning_rate(self.d_optim,
                                         LEARNING_RATE_DICT.get(resolution,
                                                                0.001))

            try:
                # Try to read next image
                real_image, label = next(data_loader)

            except (OSError, StopIteration):
                # Dataset exhausted, train from the first image
                data_loader = iter(origin_loader)
                real_image, label = next(data_loader)
                dataset_one_round = True

            # Count used sample
            self.used_sample += real_image.shape[0]
            progress_bar.update(real_image.shape[0])

            # Send image to GPU
            real_image = real_image.to(DEVICE)

            # --- Train discriminator ---
            self.discriminator.zero_grad()
            self.set_grad_flag(self.discriminator, True)
            self.set_grad_flag(self.generator, False)

            # Predict Real image
            # Loss: Non-saturating loss for more stable weight update
            real_image.requires_grad = True
            real_predict = self.discriminator(real_image, self.step,
                                              self.alpha)
            real_predict = nn.functional.softplus(-real_predict).mean()
            real_predict.backward(retain_graph=True)

            grad_real = \
                torch.autograd.grad(outputs=real_predict.sum(),
                                    inputs=real_image,
                                    create_graph=True)[0]
            grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2,
                                                                            dim=1) ** 2).mean()
            grad_penalty_real = 10 / 2 * grad_penalty_real
            grad_penalty_real.backward()

            # Predict Fake image
            # Generate latent code
            latent_w1 = [torch.randn(
                (BATCH_SIZE_DICT.get(resolution, MINI_BATCH_SIZE), LATENT_DIM),
                device=DEVICE)]
            latent_w2 = [torch.randn(
                (BATCH_SIZE_DICT.get(resolution, MINI_BATCH_SIZE), LATENT_DIM),
                device=DEVICE)]
            noise_1 = []
            noise_2 = []
            for m in range(self.step + 1):
                size = 4 * 2 ** m
                noise_1.append(torch.randn((BATCH_SIZE_DICT.get(resolution,
                                                                MINI_BATCH_SIZE),
                                            1, size, size), device=DEVICE))
                noise_2.append(torch.randn((BATCH_SIZE_DICT.get(resolution,
                                                                MINI_BATCH_SIZE),
                                            1, size, size), device=DEVICE))

            # Generate fake image & backward
            fake_image = self.generator(latent_w1, self.step, self.alpha,
                                        noise_1)
            fake_predict = self.discriminator(fake_image, self.step,
                                              self.alpha)
            fake_predict = nn.functional.softplus(fake_predict).mean()
            fake_predict.backward()

            if self.iteration % SHOW_LOSS_PER_SAMPLE == 0:
                self.discriminator_loss_dict.append(
                    (real_predict + fake_predict).item())
            # Discriminator optimizer step
            self.d_optim.step()
            # Avoid CUDA out of memory error
            del grad_penalty_real, grad_real, fake_predict, real_predict, fake_image, real_image, latent_w1

            # --- Train generator based on DISCRIMINATOR_GENERATOR_RATIO ---
            if self.iteration % DISCRIMINATOR_GENERATOR_RATIO != 0:
                continue
            self.generator.zero_grad()
            self.set_grad_flag(self.discriminator, False)
            self.set_grad_flag(self.generator, True)

            fake_image = self.generator(latent_w2, self.step, self.alpha,
                                        noise_2)
            fake_predict = self.discriminator(fake_image, self.step,
                                              self.alpha)
            fake_predict = nn.functional.softplus(-fake_predict).mean()
            fake_predict.backward()
            self.g_optim.step()

            if self.iteration % 50 == 0:
                self.generator_loss_dict.append(fake_predict.item())

            if dataset_one_round:
                # Save sample generated images every epoch
                for i in range(10):
                    # Generate 10 samples
                    noise_sample = []
                    for m in range(self.step + 1):
                        size = 4 * 2 ** m
                        noise_sample.append(
                            torch.randn((BATCH_SIZE_DICT.get(resolution,
                                                             MINI_BATCH_SIZE),
                                         1, size, size), device=DEVICE))
                    latent_sample = [torch.randn((BATCH_SIZE_DICT.get(
                        resolution, MINI_BATCH_SIZE), LATENT_DIM),
                        device=DEVICE)]
                    gen_img = self.generator(latent_sample, self.step,
                                             self.alpha, noise_sample)
                    self.save_img(gen_img.data.cpu(), self.used_sample,
                                  self.cur_epoch, i,
                                  self.dataset_config['save_folder_path'])
                dataset_one_round = False
                self.cur_epoch += 1

            # Avoid CUDA out of memory error
            del fake_predict, fake_image, latent_w2

            if self.iteration % 1000 == 0:
                # Save the model every 1000 iterations
                torch.save({
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'g_optim': self.g_optim.state_dict(),
                    'd_optim': self.d_optim.state_dict(),
                    'parameters': (self.step, self.iteration, self.startpoint,
                                   self.used_sample, self.alpha,
                                   self.cur_epoch),
                    'discriminator_loss_dict': self.discriminator_loss_dict,
                    'generator_loss_dict': self.generator_loss_dict
                }, 'checkpoint/trained.pth')
                print('Model successfully saved.')

            progress_bar.set_description((
                f'Current Resolution: {resolution}*{resolution}  '
                f'Discriminator Loss: {self.discriminator_loss_dict[-1]:.4f}  '
                f'Generator Loss: {self.generator_loss_dict[-1]:.4f}  '))
        # Training finish
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optim': self.g_optim.state_dict(),
            'd_optim': self.d_optim.state_dict(),
            'parameters': (
                self.step, self.iteration, self.startpoint, self.used_sample,
                self.alpha, self.cur_epoch),
            'discriminator_loss_dict': self.discriminator_loss_dict,
            'generator_loss_dict': self.generator_loss_dict
        }, 'checkpoint/trained.pth')
        print('Final model successfully saved.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script_name.py {dataset_name}")
        sys.exit(1)

    dataset_name = sys.argv[1]
    print("Dataset name:", dataset_name)
    training_module = StyleGANTraining(dataset_name)
    training_module.train()
