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


from tqdm import tqdm


def train_function_googlenet(dataloaders, model, optimizer, loss_fn, device):
    loss_values = []
    accuracy_values = {
        'Train': [],
        'Test': []
    }  # To track accuracy for each phase

    for phase in ['Train', 'Test']:
        if phase == 'Train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over data
        for sample in tqdm(iter(dataloaders[phase])):
            X, y = sample["image"], sample["label"]
            X, y = X.to(device), y.to(device).long()

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'Train'):
                # Forward pass
                p1, p2, p3 = model(X)

                # Compute loss
                loss_1 = loss_fn(p1, y)
                loss_2 = loss_fn(p2, y)
                loss_3 = loss_fn(p3, y)
                loss = loss_3 + loss_2 * 0.3 + loss_1 * 0.3

                # Backward pass and optimization
                if phase == 'Train':
                    # loss_1.backward(retain_graph=True)
                    # loss_2.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * X.size(0)

            # Compute accuracy
            _, preds = torch.max(
                p3, 1)  # Use the final prediction (p3) for accuracy
            correct_predictions += torch.sum(preds == y).item()
            total_samples += y.size(0)

        # Calculate average loss and accuracy for the phase
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        loss_values.append(epoch_loss)
        accuracy_values[phase].append(epoch_accuracy)

        print(
            f"{phase} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return loss_values, accuracy_values
