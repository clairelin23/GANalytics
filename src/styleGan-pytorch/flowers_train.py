import os

import torch
from PIL import Image
from torch import nn, optim
from torchvision import datasets
from tqdm import tqdm

from dataset_preview import gain_sample
from model import StyleBased_Generator, Discriminator

"""
Training module and parameters for the Pokemon dataset
"""

# TODO: refactor dataset settings as input parameters for easier training on different dataset

# GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
n_gpu = 8
device = torch.device('cuda')

# Original Learning Rate
learning_rate = {128: 0.0015, 256: 0.002}
batch_size = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 16}
mini_batch_size = 8

# Common line below if you don't meet the problem of 'shared memory conflict'
num_workers = {128: 8, 256: 4, 512: 2}
max_workers = 16
n_fc = 8
dim_latent = 512
dim_input = 4
epoch = 100
# number of samples to show before doubling resolution
dataset_num = 8190
n_sample = dataset_num * epoch  # number of samples in dataset * number of epochs
# number of samples train model in total
n_sample_total = n_sample * 7  # multiply by number of resolution changes
DGR = 1
n_show_loss = 200
step = 1  # Train from (8 * 8)
max_step = 7
image_folder_path = '../project/flowers/'
save_folder_path = './results_flowers/'

Z_DIM = 512

# Used to continue training from last checkpoint
iteration = 0
startpoint = 0
used_sample = 0
alpha = 0

# True for start from saved model
# False for retrain from the very beginning
is_continue = True
d_losses = [float('inf')]
g_losses = [float('inf')]


def set_grad_flag(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def reset_LR(optimizer, lr):
    for pam_group in optimizer.param_groups:
        mul = pam_group.get('mul', 1)
        pam_group['lr'] = lr * mul


def imsave(tensor, iteration, epoch, index):
    grid = tensor[0]
    grid.clamp_(-1, 1).add_(1).div_(2)
    # Add 0.5 after normalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(f'{save_folder_path}epoch_{epoch}_iteration_{iteration}_ind_{index}.png')


# Train function
def train(generator, discriminator, g_optim, d_optim, dataset, step, iteration=0, startpoint=0, used_sample=0,
          d_losses=[], g_losses=[], alpha=0, cur_epoch=0):
    resolution = 4 * 2 ** step
    dataset_one_round = False

    origin_loader = gain_sample(dataset, batch_size.get(resolution, mini_batch_size), resolution)
    data_loader = iter(origin_loader)

    reset_LR(g_optim, learning_rate.get(resolution, 0.001))
    reset_LR(d_optim, learning_rate.get(resolution, 0.001))
    progress_bar = tqdm(total=n_sample_total, initial=used_sample)
    # Train
    while used_sample < n_sample_total:
        iteration += 1
        alpha = min(1, alpha + batch_size.get(resolution, mini_batch_size) / (n_sample))

        if (used_sample - startpoint) > n_sample and step < max_step:
            step += 1
            alpha = 0
            startpoint = used_sample

            resolution = 4 * 2 ** step
            print('training at next resolution', resolution)

            # Avoid possible memory leak
            del origin_loader
            del data_loader

            # Change batch size
            origin_loader = gain_sample(dataset, batch_size.get(resolution, mini_batch_size), resolution)
            data_loader = iter(origin_loader)

            reset_LR(g_optim, learning_rate.get(resolution, 0.001))
            reset_LR(d_optim, learning_rate.get(resolution, 0.001))

        try:
            # Try to read next image
            real_image, label = next(data_loader)

        except (OSError, StopIteration):
            # Dataset exhausted, train from the first image
            data_loader = iter(origin_loader)
            real_image, label = next(data_loader)
            dataset_one_round = True

        # Count used sample
        used_sample += real_image.shape[0]
        progress_bar.update(real_image.shape[0])

        # Send image to GPU
        real_image = real_image.to(device)

        # D Module ---
        # Train discriminator first
        discriminator.zero_grad()
        set_grad_flag(discriminator, True)
        set_grad_flag(generator, False)

        # Real image predict & backward
        # We only implement non-saturating loss with R1 regularization loss
        real_image.requires_grad = True
        if n_gpu > 1:
            real_predict = nn.parallel.data_parallel(discriminator, (real_image, step, alpha), range(n_gpu))
        else:
            real_predict = discriminator(real_image, step, alpha)
        real_predict = nn.functional.softplus(-real_predict).mean()
        real_predict.backward(retain_graph=True)

        grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real
        grad_penalty_real.backward()

        # Generate latent code
        latent_w1 = [torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device)]
        latent_w2 = [torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device)]

        noise_1 = []
        noise_2 = []
        for m in range(step + 1):
            size = 4 * 2 ** m  # Due to the upsampling, size of noise will grow
            noise_1.append(torch.randn((batch_size.get(resolution, mini_batch_size), 1, size, size), device=device))
            noise_2.append(torch.randn((batch_size.get(resolution, mini_batch_size), 1, size, size), device=device))

        # Generate fake image & backward
        if n_gpu > 1:
            fake_image = nn.parallel.data_parallel(generator, (latent_w1, step, alpha, noise_1), range(n_gpu))
            fake_predict = nn.parallel.data_parallel(discriminator, (fake_image, step, alpha), range(n_gpu))
        else:
            fake_image = generator(latent_w1, step, alpha, noise_1)
            fake_predict = discriminator(fake_image, step, alpha)

        fake_predict = nn.functional.softplus(fake_predict).mean()
        fake_predict.backward()

        if iteration % n_show_loss == 0:
            d_losses.append((real_predict + fake_predict).item())

        # D optimizer step
        d_optim.step()

        # Avoid possible memory leak
        del grad_penalty_real, grad_real, fake_predict, real_predict, fake_image, real_image, latent_w1

        # G module ---
        if iteration % DGR != 0: continue
        # Due to DGR, train generator
        generator.zero_grad()
        set_grad_flag(discriminator, False)
        set_grad_flag(generator, True)

        if n_gpu > 1:
            fake_image = nn.parallel.data_parallel(generator, (latent_w2, step, alpha, noise_2), range(n_gpu))
            fake_predict = nn.parallel.data_parallel(discriminator, (fake_image, step, alpha), range(n_gpu))
        else:
            fake_image = generator(latent_w2, step, alpha, noise_2)
            fake_predict = discriminator(fake_image, step, alpha)
        fake_predict = nn.functional.softplus(-fake_predict).mean()
        fake_predict.backward()
        g_optim.step()

        if iteration % 50 == 0:
            g_losses.append(fake_predict.item())

        if dataset_one_round:  # save sample generated images every epoch
            # generate 10 samples
            for i in range(10):
                noise_sample = []
                for m in range(step + 1):
                    size = 4 * 2 ** m  # Due to the upsampling, size of noise will grow
                    noise_sample.append(
                        torch.randn((batch_size.get(resolution, mini_batch_size), 1, size, size), device=device))
                latent_sample = [torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device)]
                gen_img = generator(latent_sample, step, alpha, noise_sample)

                imsave(gen_img.data.cpu(), used_sample, cur_epoch, i)
            dataset_one_round = False
            cur_epoch += 1

        # Avoid possible memory leak
        del fake_predict, fake_image, latent_w2

        if iteration % 1000 == 0:
            # Save the model every 1000 iterations
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'parameters': (step, iteration, startpoint, used_sample, alpha, cur_epoch),
                'd_losses': d_losses,
                'g_losses': g_losses
            }, 'checkpoint_flowers/trained.pth')
            print('Model successfully saved.')

        progress_bar.set_description((
            f'Resolution: {resolution}*{resolution}  D_Loss: {d_losses[-1]:.4f}  G_Loss: {g_losses[-1]:.4f}  Alpha: {alpha:.4f}'))
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optim': g_optim.state_dict(),
        'd_optim': d_optim.state_dict(),
        'parameters': (step, iteration, startpoint, used_sample, alpha, cur_epoch),
        'd_losses': d_losses,
        'g_losses': g_losses
    }, 'checkpoint_flowers/trained.pth')
    print('Final model successfully saved.')
    return d_losses, g_losses


# Create models
generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)
discriminator = Discriminator().to(device)

# Optimizers
g_optim = optim.Adam([{
    'params': generator.convs.parameters(),
    'lr': 0.001
}, {
    'params': generator.to_rgbs.parameters(),
    'lr': 0.001
}], lr=0.001, betas=(0.0, 0.99))
g_optim.add_param_group({
    'params': generator.fcs.parameters(),
    'lr': 0.001 * 0.01,
    'mul': 0.01
})
d_optim = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))
dataset = datasets.ImageFolder(image_folder_path)
cur_epoch = 0

if is_continue:
    if os.path.exists('checkpoint_flowers/trained.pth'):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load('checkpoint_flowers/trained.pth')
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        step, iteration, startpoint, used_sample, alpha, cur_epoch = checkpoint['parameters']
        d_losses = checkpoint.get('d_losses', [float('inf')])
        g_losses = checkpoint.get('g_losses', [float('inf')])
        print('Start training from loaded model...')
    else:
        print('No pre-trained model detected, restart training...')

generator.train()
discriminator.train()
d_losses, g_losses = train(generator, discriminator, g_optim, d_optim, dataset, step, iteration, startpoint,
                           used_sample, d_losses, g_losses, alpha, cur_epoch)
