import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

import torch

from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

id2label = {
    0: 'Acne and Rosacea Photos',
    1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    2: 'Atopic Dermatitis Photos',
    3: 'Bullous Disease Photos',
    4: 'Cellulitis Impetigo and other Bacterial Infections',
    5: 'Eczema Photos',
    6: 'Exanthems and Drug Eruptions',
    7: 'Hair Loss Photos Alopecia and other Hair Diseases',
    8: 'Herpes HPV and other STDs Photos',
    9: 'Light Diseases and Disorders of Pigmentation',
    10: 'Lupus and other Connective Tissue diseases',
    11: 'Melanoma Skin Cancer Nevi and Moles',
    12: 'Nail Fungus and other Nail Disease',
    13: 'Poison Ivy Photos and other Contact Dermatitis',
    14: 'Psoriasis pictures Lichen Planus and related diseases',
    15: 'Scabies Lyme Disease and other Infestations and Bites',
    16: 'Seborrheic Keratoses and other Benign Tumors',
    17: 'Systemic Disease',
    18: 'Tinea Ringworm Candidiasis and other Fungal Infections',
    19: 'Urticaria Hives',
    20: 'Vascular Tumors',
    21: 'Vasculitis Photos',
    22: 'Warts Molluscum and other Viral Infections'
}

label2id = {
    'Acne and Rosacea Photos': 0,
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 1,
    'Atopic Dermatitis Photos': 2,
    'Bullous Disease Photos': 3,
    'Cellulitis Impetigo and other Bacterial Infections': 4,
    'Eczema Photos': 5,
    'Exanthems and Drug Eruptions': 6,
    'Hair Loss Photos Alopecia and other Hair Diseases': 7,
    'Herpes HPV and other STDs Photos': 8,
    'Light Diseases and Disorders of Pigmentation': 9,
    'Lupus and other Connective Tissue diseases': 10,
    'Melanoma Skin Cancer Nevi and Moles': 11,
    'Nail Fungus and other Nail Disease': 12,
    'Poison Ivy Photos and other Contact Dermatitis': 13,
    'Psoriasis pictures Lichen Planus and related diseases': 14,
    'Scabies Lyme Disease and other Infestations and Bites': 15,
    'Seborrheic Keratoses and other Benign Tumors': 16,
    'Systemic Disease': 17,
    'Tinea Ringworm Candidiasis and other Fungal Infections': 18,
    'Urticaria Hives': 19,
    'Vascular Tumors': 20,
    'Vasculitis Photos': 21,
    'Warts Molluscum and other Viral Infections': 22
}


class DermnetDataset(VisionDataset):

    def __init__(self,
                 root: str,
                 data_transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = 'rgb'):

        super().__init__(root, data_transforms)

        image_root_path = Path(self.root)

        if not image_root_path.exists():
            raise OSError(f"{image_root_path} does not exist.")

        if image_color_mode not in ['rgb', 'grayscale']:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb | grayscale."
            )

        self.image_color_mode = image_color_mode

        self.images_names = []
        self.labels = []

        for subdir in image_root_path.glob("*"):
            image_dir = Path(self.root) / subdir
            names = sorted(image_dir.glob("*"))
            label = label2id[str(subdir.parts[-1])]
            labels = [label for _ in range(len(names))]

            self.images_names += names
            self.labels += labels

            # print(len(self.images_names), len(self.labels))
            # idx = random.randint(0, len(self.images_names))
            # print(self.images_names[idx], self.labels[idx])

        self.image_list = np.array(self.images_names)
        self.labels_list = np.array(self.labels)

        if seed:
            np.random.seed(seed)
            indices = np.arange(len(self.image_list))
            np.random.shuffle(indices)

            self.image_list = self.image_list[indices]
            self.labels_list = self.labels_list[indices]

        self.data_transforms = data_transforms

        self.fraction = fraction

        if fraction:
            if subset not in ['Train', 'Test']:
                raise ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train | Test."
                )

            split_index = int(
                np.ceil(len(self.image_list) * (1 - self.fraction)))

            if (split_index % 2 == 1):
                split_index += 1

            if subset == 'Train':
                self.images_names = self.image_list[:split_index]
                self.labels = self.labels_list[:split_index]
            else:
                self.images_names = self.image_list[split_index:]
                self.labels = self.labels_list[split_index:]

            # print(len(self.images_names), len(self.labels))

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.images_names[index]
        label = self.labels[index]

        with open(image_path, 'rb') as image_file:
            image = Image.open(image_file)

            if self.image_color_mode == 'rgb':
                image = image.convert('RGB')
            elif self.image_color_mode == 'grayscale':
                image = image.convert('L')

            # sample = {"image": image, "label": label, "dims": image.size}
            sample = {"image": image, "label": label}

            if self.data_transforms:
                sample["image"] = self.data_transforms(sample["image"])

            return sample


'''
tsf = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dermnet = DermnetDataset(root='/home/lulu/Downloads/dermnet/train',
                         data_transforms=tsf,
                         seed=42,
                         fraction=0.1,
                         subset='Test')

import matplotlib.pyplot as plt


class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

min_width = 1000
min_height = 1000

for i in range(dermnet.__len__()):
    sample = dermnet.__getitem__(i)
    size = sample["dims"]
    min_width = min(min_width, size[0])
    min_height = min(min_height, size[1])

print(min_width, min_height)

index = random.randint(0, dermnet.__len__())
sample = dermnet.__getitem__(index)
image = unnorm(sample["image"])
image = np.dstack(image.numpy())

plt.subplot(1, 1, 1)
plt.title(f"{sample['label']} : {id2label[sample['label']]}")
plt.imshow(image)
plt.show()
'''
