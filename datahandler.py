from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DermnetDataset


def get_dataloader(data_dir: str,
                   num_classes: int,
                   data_transforms: Optional[Callable] = transforms.Compose([
                       transforms.CenterCrop(512),
                       transforms.Resize((512, 512)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                   ]),
                   seed: int = 42,
                   fraction: float = 0.1,
                   batch_size: int = 2):

    image_datasets = {
        x: DermnetDataset(root=data_dir,
                          seed=seed,
                          data_transforms=data_transforms,
                          fraction=fraction,
                          subset=x)
        for x in ['Train', 'Test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)
        for x in ['Train', 'Test']
    }

    return dataloaders
