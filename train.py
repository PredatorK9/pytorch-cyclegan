import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import PatchGAN, Generator
from utils import apply_weights, decaying_lr
import os
import argparse
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetA', required=True, type=str,
        help='Path to dataset of Domain A')
    parser.add_argument('--datasetB', required=True, type=str,
        help='Path to dataset of Domain B')
    parser.add_argument('--ckpth', required=False, type=str, default='./',
        help='Path for storing the checkpoints')
    parser.add_argument('--num_epochs', required=False, type=int, default=200, 
        help='Number of epochs for training of the model')
    parser.add_argument('--batch_size', required=False, type=int, default=1,
        help='The batch size of training data')
    parser.add_argument('--cycle_weight', required=False, type=int, default=10,
        help='Weight for the cycle consistency loss')
    parser.add_argument('--num_res_blocks', required=False, type=int,
        default=9, help='Number of residual blocks in the generator')
    parser.add_argument('--lr', required=False, type=float, default=0.0002,
        help='The learning rate for the optimizer')
    parser.add_argument('--beta1', required=False, type=float, default=0.5,
        help='Adam optimizer hyperparameter')
    parser.add_argument('--beta2', required=False, type=float, default=0.999,
        help='Adam optimizer hyperparameter')
    parser.add_argument('--decay_epoch', required=False, type=int, default=100,
        help='Number of epochs upto which learning rate stays constant')
    parser.add_argument('--continue_train', required=False, type=bool,
        default=False, help='Continue the training')
    parser.add_argument('--device', required=False, type=str, default='cpu',
        help='Training device')

    return parser.parse_args()


def load_data(X_path, Y_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])

    domainX_dataset = datasets.ImageFolder(X_path, transform=transform)
    domainY_dataset = datasets.ImageFolder(Y_path, transform=transform)

    dataloaderX = torch.utils.data.DataLoader(domainX_dataset, shuffle=True,
                                    batch_size=batch_size)
    dataloaderY = torch.utils.data.DataLoader(domainY_dataset, shuffle=True,
                                    batch_size=batch_size)

    return dataloaderX, dataloaderY


def create_models(num_res_blocks, device):
    G_XtoY = Generator(num_res_blocks)
    G_YtoX = Generator(num_res_blocks)

    D_X = PatchGAN()
    D_Y = PatchGAN()

    apply_weights(G_XtoY)
    apply_weights(G_YtoX)
    apply_weights(D_X)
    apply_weights(D_Y)

    G_XtoY, G_YtoX, D_X, D_Y = G_XtoY.to(device), G_YtoX.to(device), \
        D_X.to(device), D_Y.to(device)

    return G_XtoY, G_YtoX, D_X, D_Y


def train(ckpth, num_epoch, lr, beta1, beta2, decay_after, reduct, cont, dataloaderX, dataloaderY,
    LAMBDA, num_res_blocks, device):

    G_XtoY, G_YtoX, D_X, D_Y = create_models(num_res_blocks, device)

    if cont:
        G_XtoY.load_state_dict(torch.load(os.path.join(ckpth, 'G_XtoY.pth')))
        G_YtoX.load_state_dict(torch.load(os.path.join(ckpth, 'G_YtoX.pth')))
        D_X.load_state_dict(torch.load(os.path.join(ckpth, 'D_X.pth')))
        D_Y.load_state_dict(torch.load(os.path.join(ckpth, 'D_Y.pth')))
        print('Checkpoints verified and loaded....')

    mse_loss = nn.MSELoss()
    cycle_consistency_loss = nn.L1Loss()

    G_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())
    optimizerG = optim.Adam(G_params, lr=lr, betas=(beta1, beta2))
    optimizerD_X = optim.Adam(D_X.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD_Y = optim.Adam(D_Y.parameters(), lr=lr, betas=(beta1, beta2))

    for epoch in range(num_epoch):
        loop = tqdm(zip(dataloaderX, dataloaderY),
            total=min(len(dataloaderX), len(dataloaderY)))
        for (imageX, _), (imageY, _) in loop:
            imageX, imageY = imageX.to(device), imageY.to(device)

            #training the discriminator_Y
            optimizerD_Y.zero_grad()
            D_Yreal = D_Y(imageY)
            real_labels = torch.ones_like(D_Yreal, device=device)
            lossD_Yreal = mse_loss(D_Yreal, real_labels)

            fake_Y = G_XtoY(imageX)
            D_Yfake = D_Y(fake_Y.detach())
            fake_labels = torch.zeros_like(D_Yfake, device=device)
            lossD_Yfake = mse_loss(D_Yfake, fake_labels)

            lossD_Y = 0.5 * (lossD_Yreal + lossD_Yfake)
            lossD_Y.backward()
            optimizerD_Y.step()

            #training the discriminator_X
            optimizerD_X.zero_grad()
            D_Xreal = D_X(imageX)
            lossD_Xreal = mse_loss(D_Xreal, real_labels)

            fake_X = G_YtoX(imageY)
            D_Xfake = D_X(fake_X.detach())
            lossD_Xfake = mse_loss(D_Xfake, fake_labels)

            lossD_X = 0.5 * (lossD_Xreal + lossD_Xfake)
            lossD_X.backward()
            optimizerD_X.step()

            #training the generators
            optimizerG.zero_grad()

            identity_Y = D_Y(fake_Y)
            identity_Y_loss = mse_loss(identity_Y, real_labels)
            reconstructed_X = G_YtoX(fake_Y)
            cyclic_loss_X = LAMBDA * \
                cycle_consistency_loss(reconstructed_X, imageX)

            identity_X = D_X(fake_X)
            identity_X_loss = mse_loss(identity_X, real_labels)
            reconstructed_Y = G_XtoY(fake_X)
            cyclic_loss_Y = LAMBDA * \
                cycle_consistency_loss(reconstructed_Y, imageY)

            lossG = identity_X_loss + cyclic_loss_X + \
                identity_Y_loss + cyclic_loss_Y

            lossG.backward()
            optimizerG.step()

            loop.set_description(F"Epoch[{epoch+1}/{num_epoch}]")
            loop.set_postfix(lossD_X= lossD_X.item(), lossD_Y= lossD_Y.item(),
                lossG_total = lossG.item())

        torch.save(D_X.state_dict(), os.path.join(ckpth, 'D_X.pth'))
        torch.save(D_Y.state_dict(), os.path.join(ckpth, 'D_Y.pth'))
        torch.save(G_XtoY.state_dict(), os.path.join(ckpth, 'G_XtoY.pth'))
        torch.save(G_YtoX.state_dict(), os.path.join(ckpth, 'G_YtoX.pth'))

        decaying_lr(optimizerD_X, epoch, decay_after, reduct)
        decaying_lr(optimizerD_Y, epoch, decay_after, reduct)
        decaying_lr(optimizerG, epoch, decay_after, reduct)


def main():
    arguments = get_arguments()
    dataloaderX, dataloaderY = load_data(arguments.datasetA,
        arguments.datasetB, arguments.batch_size)
    reduct = arguments.lr / (arguments.num_epochs - arguments.decay_epoch)
    train(arguments.ckpth, arguments.num_epochs, arguments.lr, arguments.beta1,
        arguments.beta2, arguments.decay_epoch, reduct, arguments.continue_train,
        dataloaderX, dataloaderY, arguments.cycle_weight, 
        arguments.num_res_blocks, arguments.device)


if __name__ == "__main__":
    main()