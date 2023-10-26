import os
import sys

import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from torchvision import transforms, datasets

from dataset_utils import gain_sample
from config import DATASETS_CONFIG
from model import Generator
from train_stylegan_model import DEVICE, INPUT_DIM, LATENT_DIM, MAPPING_LAYER_NUM, STEP, MINI_BATCH_SIZE
"""
This file calculates evaluates a given dataset and calculate the FID score
"""

# load model
generator = Generator(MAPPING_LAYER_NUM, LATENT_DIM, INPUT_DIM).to(DEVICE)
if os.path.exists('checkpoint/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model...')
    checkpoint = torch.load('checkpoint/trained.pth')
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


def evaluation_step(data_batch):
    """
    Define evaluation step to generate images and interpolate generated
    and real images
    """
    image_batch, _ = data_batch
    alpha = 0  # not implemented, only for style mixing purpose
    with torch.no_grad():
        noise_sample = []
        for m in range(STEP + 1):
            size = 4 * 2 ** m  # Due to the upsampling, size of noise will grow
            noise_sample.append(torch.randn((MINI_BATCH_SIZE, 1, size, size),
                                            device=DEVICE))
        latent_sample = [torch.randn((MINI_BATCH_SIZE, LATENT_DIM),
                                     device=DEVICE)]
        gen_img = generator(latent_sample, STEP, alpha, noise_sample)

    generated_images = interpolate(gen_img)
    real_images = interpolate(image_batch)

    return generated_images, real_images


# define evaluation function to calculate FID and IS
def evaluate(evaluated_model, dataloader):
    global model
    model = evaluated_model
    model.to(DEVICE)
    model.eval()
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['inception']

    return fid_score, is_score


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python calculate_scores.py dataset_name {flowers}")
        sys.exit(1)

    dataset_name = sys.argv[1]
    dataset_config = DATASETS_CONFIG[dataset_name]
    print("Dataset name:", dataset_name)

    fid = FID(device=DEVICE)
    inception = InceptionScore(device=DEVICE, output_transform=lambda x: x[0])
    evaluator = Engine(evaluation_step)
    fid.attach(evaluator, "fid")
    inception.attach(evaluator, "inception")
    image_folder_path = dataset_config['image_folder_path']
    dataset = datasets.ImageFolder(image_folder_path)
    resolution = 4 * 2 ** STEP
    origin_loader = gain_sample(dataset, MINI_BATCH_SIZE, resolution)

    print(evaluate(generator, origin_loader))
