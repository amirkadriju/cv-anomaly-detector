import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import os
import yaml

from models import get_model
from data_loader import KolektorDataset



def load_config(config_path='./config/config.yaml'):
    with open(config_path, 'r') as file: 
        return yaml.safe_load(file)


def train_autoencoder(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, _, _ in dataloader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_classifier(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, _, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_unet(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    selected_model = config['selected_model']
    model_config = config['models'][selected_model]

    # Path to train directory
    image_train_dir = './data/processed_kolektor/train/images'
    masks_train_dir = './data/processed_kolektor/train/masks'

    # Dataset and Dataloader
    dataset = KolektorDataset(image_dir=image_train_dir, mask_dir=masks_train_dir)
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 4), shuffle=True)

    # Model
    model = get_model(model_config['type'], model_config).to(device)

    # Loss & Optimizer
    if selected_model == 'autoencoder':
        criterion = nn.MSELoss()
    elif selected_model == 'classifier':
        criterion = nn.BCELoss()
    elif selected_model == 'unet':
        criterion = nn.BCELoss()
    else:
        raise ValueError(f'Unsupported model type: {selected_model}')

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))

    # Training loop
    for epoch in range(config.get('epochs', 10)):
        print(f'Epoch {epoch + 1}/{config.get("epochs", 10)}')

        if selected_model == 'autoencoder':
            train_autoencoder(model, dataloader, criterion, optimizer, device)
        elif selected_model == 'classifier':
            train_classifier(model, dataloader, criterion, optimizer, device)
        elif selected_model == 'unet':
            train_unet(model, dataloader, criterion, optimizer, device)

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'./checkpoints/{selected_model}_final.pth')
    print('Training complete and model saved!')


if __name__ == '__main__':
    main()
