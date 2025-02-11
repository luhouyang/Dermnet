from tqdm import tqdm

import torch


def train_function(data, model, optimizer, loss_fn, device):
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def train_function_googlenet(dataloaders, model, optimizer, loss_fn, device):
    loss_values = []
    for phase in ['Train', 'Test']:
        if phase == 'Train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0

        # Iterate over data.
        for sample in tqdm(iter(dataloaders[phase])):

            X, y = sample["image"], sample["label"]
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'Train'):
                p1, p2, p3 = model(X)

                loss_1 = loss_fn(p1, y)
                loss_2 = loss_fn(p2, y)
                loss_3 = loss_fn(p3, y)

                loss = loss_3 + loss_2 * 0.3 + loss_1 * 0.3
            if phase == 'Train':
                loss.backward()
                optimizer.step()

    return loss.item()
