import csv
import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from datahandler import get_dataloader, get_dataloader_two_folders
# from model import Classifier
from googlenet_model import GoogLeNet
from trainer import train_function, train_function_googlenet

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = "cpu"

MODEL_PATH = 'output/model.pth'
LOAD_MODEL = False
# ROOT_DIR = '/home/lulu/Downloads/dermnet/train'
ROOT_DIR = 'D:/storage/dermnet'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50


def main():
    global epoch
    epoch = 0

    LOSS_VALS = []

    data_transforms = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # train_set = get_dataloader(
    #   data_dir=ROOT_DIR,
    #     data_transforms=data_transforms,
    #     seed=42,
    #     fraction=0.05,
    #     batch_size=BATCH_SIZE,
    #     num_classes=23,
    # )

    train_set = get_dataloader_two_folders(
        train_dir=ROOT_DIR + '/train',
        test_dir=ROOT_DIR + '/test',
        data_transforms=data_transforms,
        seed=42,
        batch_size=BATCH_SIZE,
        num_classes=23,
    )

    print(
        f'Data loaded successfully.\nRunning on {"CPU" if DEVICE == "cpu" else "GPU"}'
    )

    model = GoogLeNet(input_size=3, num_classes=23, weight_decay=0.01)
    model.to(DEVICE)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=255)

    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] + 1
        LOSS_VALS = checkpoint['loss_values']
        print('Model loaded successfully')

    fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc']
    with open(os.path.join('output', 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for e in range(epoch, EPOCHS):
        print(f'Epoch: {e}')
        loss_val, accuracy_values = train_function_googlenet(train_set, model, optimizer,
                                            loss_function, DEVICE)

        batchsummary = {a: [0] for a in fieldnames}
        batchsummary['epoch'] = e
        batchsummary['train_loss'] = loss_val[0]
        batchsummary['test_loss'] = loss_val[1]
        batchsummary['train_acc'] = accuracy_values['Train'][0]
        batchsummary['test_acc'] = accuracy_values['Test'][0]
        with open(os.path.join('output', 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
            writer.writerow(batchsummary)

        LOSS_VALS.append(loss_val)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': e,
                'loss_values': LOSS_VALS
            }, MODEL_PATH.replace('model', f'{e}_model'))
        print(f'Loss: {loss_val} {accuracy_values}')


if __name__ == '__main__':
    main()
