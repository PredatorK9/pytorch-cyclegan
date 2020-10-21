from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from models import Generator
import argparse
import os

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5])
])

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', required=True, type=str,
        help='Path to the source image')
    parser.add_argument('--dest', required=False, type=str,
        default='./', help='Path to the destination folder')
    parser.add_argument('--ckpth', required=False, type=str,
        default='./', help='Path to the Generator checkpoint')

    arguments = parser.parse_args()
    return arguments


def load_image(path):
    image = Image.open(path).convert('RGB')
    image = TRANSFORM(image)[:3, :, :].unsqueeze(0)
    _, name = os.path.split(path)
    return image, name


def convert_save(path, name, ckpth, image):
    model = Generator()
    model.load_state_dict(torch.load(os.path.join(ckpth, 'G_YtoX.pth')))

    output_image = model(image).squeeze(0).detach().numpy()
    output_image = np.moveaxis(output_image, 0, 2)
    output_image = output_image * np.array((0.5, 0.5, 0.5)) + \
        np.array((0.5, 0.5, 0.5))
    output_image = np.clip(output_image, 0, 1)
    output_image = Image.fromarray(np.uint8(output_image * 255))
    output_image.save(os.path.join(path, name))


def main():
    arguments = get_arguments()
    image, name = load_image(arguments.src)
    convert_save(arguments.dest, name, arguments.ckpth, image)


if __name__ == "__main__":
    main()