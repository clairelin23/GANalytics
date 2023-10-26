import os

import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from torchvision import transforms, datasets

from dataset_preview import gain_sample
from model import StyleBased_Generator

"""
This file calculates evaluates a given dataset and calculate the FID score
"""

n_fc = 8  # number of linear layers in mapper network
dim_latent = 512
dim_input = 4
device = torch.device("cuda")
step = 7
batch = 64
latent_dim = 512

# load model
generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)
if os.path.exists('checkpoint_flowers/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load('checkpoint_flowers/trained.pth')
    generator.load_state_dict(checkpoint['generator'])

# define transformation to resize image in order to be used by ignite
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])


# define function to interpolate images in order to be used by ignite
def interpolate(image_batch):
    transformed_image = []
    for image in image_batch:
        transformed_image.append(resize_transform(image))
    return torch.stack(transformed_image)


# set global model variable
global model


# define evaluation step to generate images and interpolate generated and real images
def evaluation_step(data_batch):
    image_batch, _ = data_batch
    alpha = 0  # not implemented, only for style mixing purpose
    with torch.no_grad():
        noise_sample = []
        for m in range(step + 1):
            size = 4 * 2 ** m  # Due to the upsampling, size of noise will grow
            noise_sample.append(torch.randn((batch, 1, size, size), device=device))
        latent_sample = [torch.randn((batch, latent_dim), device=device)]
        gen_img = generator(latent_sample, step, alpha, noise_sample)

    generated_images = interpolate(gen_img)
    real_images = interpolate(image_batch)

    return generated_images, real_images


# define evaluation function to calculate FID and IS
def evaluate(evaluated_model, dataloader):
    global model
    model = evaluated_model
    model.to(device)
    model.eval()
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['inception']

    return fid_score, is_score


# define some variables for calculating FID and IS
fid = FID(device=device)
inception = InceptionScore(device=device, output_transform=lambda x: x[0])

evaluator = Engine(evaluation_step)
fid.attach(evaluator, "fid")
inception.attach(evaluator, "inception")
image_folder_path = '../project/flowers/'
dataset = datasets.ImageFolder(image_folder_path)
resolution = 4 * 2 ** step
origin_loader = gain_sample(dataset, batch, resolution)

print(evaluate(generator, origin_loader))
