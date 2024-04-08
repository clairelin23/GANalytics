from torch.utils.data import DataLoader
from torchvision import transforms


def gain_sample(dataset, batch_size, image_size=4):
    """
    Loads dataset and return the loader
    :param dataset: (DatasetFolder) the dataset to be processed
    :param batch_size: (int) batch size for loading the dataset
    :param image_size: (int) resolution of image after crop
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize to the same size
        transforms.CenterCrop(image_size),  # Crop to get square area
        transforms.RandomHorizontalFlip(),  # Increase number of samples
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])
    dataset.transform = transform
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size,
                        num_workers=4)
    return loader
