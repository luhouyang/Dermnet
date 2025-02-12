import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math


class L1(torch.nn.Module):

    def __init__(self, module, weight_decay=0.01):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(
            self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih,
                                   k=self.kernel_size[0],
                                   s=self.stride[0],
                                   d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw,
                                   k=self.kernel_size[1],
                                   s=self.stride[1],
                                   d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class InceptionLayer(nn.Module):

    def __init__(self, filters, input_size, weight_decay):
        super(InceptionLayer, self).__init__()

        filter_1x1a, filter_1x1b, filter_3x3, filter_1x1c, filter_5x5, filter_1x1d = filters

        # layer 1
        self.layer_1x1a = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            L1(
                nn.Conv2d(input_size, filter_1x1a, (1, 1), padding='same'),
                weight_decay,
            ),
        )

        # layer 2
        self.layer_1x1b = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            L1(
                nn.Conv2d(input_size, filter_1x1b, (1, 1), padding='same'),
                weight_decay,
            ),
            nn.BatchNorm2d(filter_1x1b),
            nn.ReLU(inplace=True),
            L1(
                nn.Conv2d(filter_1x1b, filter_3x3, (3, 3), padding='same'),
                weight_decay,
            ),
        )

        # layer 3
        self.layer_1x1c = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            L1(
                nn.Conv2d(input_size, filter_1x1c, (1, 1), padding='same'),
                weight_decay,
            ),
            nn.BatchNorm2d(filter_1x1c),
            nn.ReLU(inplace=True),
            L1(
                nn.Conv2d(filter_1x1c, filter_5x5, (5, 5), padding='same'),
                weight_decay,
            ),
        )

        # layer 4
        self.layer_1x1d = nn.Sequential(
            nn.MaxPool2d((3, 3), 1, padding=1),
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            L1(
                nn.Conv2d(input_size, filter_1x1d, (1, 1), padding='same'),
                weight_decay,
            ),
        )

    def forward(self, x):
        x_1 = self.layer_1x1a(x)
        x_2 = self.layer_1x1b(x)
        x_3 = self.layer_1x1c(x)
        x_4 = self.layer_1x1d(x)

        concat = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        return concat


class ClassifierLayer(nn.Module):

    def __init__(self, input_size, input_dim, num_classes):
        super(ClassifierLayer, self).__init__()

        h, w = input_dim
        linear_input = int(128 * (math.floor((h - 5) / 3) + 1) * (math.floor(
            (w - 5) / 3) + 1))
        # 4a 14x14x512 = 2048
        # 4e 14x14x832 = 2048

        self.classifier = nn.Sequential(
            nn.AvgPool2d((5, 5), 3),
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_size, 128, (1, 1), padding='same'),
            nn.Flatten(),
            nn.Linear(linear_input, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Softmax(),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class GoogLeNet(nn.Module):

    def __init__(self, input_size, num_classes, weight_decay):
        super(GoogLeNet, self).__init__()

        self.head_layer = nn.Sequential(
            L1(
                # nn.Conv2d(input_size, 64, (7, 7), 2),
                Conv2dSame(input_size, 64, (7, 7), 2),
                weight_decay,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), 2, padding=1),
            L1(
                nn.Conv2d(64, 192, (3, 3), 1, padding='same'),
                weight_decay,
            ),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), 2, padding=1),
        )

        self.inception_3a = InceptionLayer(
            filters=[64, 96, 128, 16, 32, 32],
            input_size=192,
            weight_decay=weight_decay,
        )

        self.inception_3b = InceptionLayer(
            filters=[128, 128, 192, 32, 96, 64],
            input_size=256,
            weight_decay=weight_decay,
        )

        self.maxpool_3 = nn.MaxPool2d((3, 3), 2, padding=1)

        self.inception_4a = InceptionLayer(
            filters=[192, 96, 208, 16, 48, 64],
            input_size=480,
            weight_decay=weight_decay,
        )

        self.inception_4b = InceptionLayer(
            filters=[160, 112, 224, 24, 64, 64],
            input_size=512,
            weight_decay=weight_decay,
        )

        self.inception_4c = InceptionLayer(
            filters=[128, 128, 256, 24, 64, 64],
            input_size=512,
            weight_decay=weight_decay,
        )

        self.inception_4d = InceptionLayer(
            filters=[112, 144, 288, 32, 64, 64],
            input_size=512,
            weight_decay=weight_decay,
        )

        self.inception_4e = InceptionLayer(
            filters=[256, 160, 320, 32, 128, 128],
            input_size=528,
            weight_decay=weight_decay,
        )

        self.classification_1 = ClassifierLayer(
            input_size=512,
            input_dim=[14, 14],
            num_classes=num_classes,
        )

        self.classification_2 = ClassifierLayer(
            input_size=832,
            input_dim=[14, 14],
            num_classes=num_classes,
        )

        self.maxpool_4 = nn.MaxPool2d((3, 3), 2, padding=1)

        self.inception_5a = InceptionLayer(
            filters=[256, 160, 320, 32, 128, 128],
            input_size=832,
            weight_decay=weight_decay,
        )

        self.inception_5b = InceptionLayer(
            filters=[384, 192, 384, 48, 128, 128],
            input_size=832,
            weight_decay=weight_decay,
        )

        self.classification_final = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((7, 7), 1),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.head_layer(x)

        out_3a = self.inception_3a(x)
        out_3b = self.inception_3b(out_3a)

        out_mp3 = self.maxpool_3(out_3b)

        out_4a = self.inception_4a(out_mp3)
        out_4b = self.inception_4b(out_4a)
        out_4c = self.inception_4c(out_4b)
        out_4d = self.inception_4d(out_4c)
        out_4e = self.inception_4e(out_4d)

        cls_1 = self.classification_1(out_4a)
        cls_2 = self.classification_2(out_4e)

        out_mp4 = self.maxpool_4(out_4e)

        out_5a = self.inception_5a(out_mp4)
        out_5b = self.inception_5b(out_5a)

        cls_3 = self.classification_final(out_5b)

        return cls_1, cls_2, cls_3
