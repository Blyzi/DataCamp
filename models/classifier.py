import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2.functional import adjust_contrast, equalize, adjust_gamma, adjust_sharpness, posterize, to_grayscale
import lightning.pytorch as pl
import torchmetrics
from PIL import Image


IMAGE_SIZE = 250
BATCH_SIZE = 32

MEAN_4288_2848 = [0.59169255, 0.28666516, 0.09397617]
STD_4288_2848 = [0.17994411, 0.12810065, 0.08768493]

MEAN_2144_1424 = [0.6072398,  0.40012667, 0.22312242]
STD_2144_1424 = [0.15314142, 0.12791779, 0.12009073]

MEAN_2048_1536 = [0.49654163, 0.25455866, 0.11922807]
STD_2048_1536 = [0.13767262, 0.09073118, 0.10520288]

MEAN = [0.59394743, 0.33779393, 0.15540815]
STD = [0.16763987, 0.13925351, 0.12256385]


class MultiLabelDataset(Dataset):
    def __init__(self, df, transforms=None, label_list=None, normalize=None):
        self.label_list = label_list
        self.df = self.encode_dataset(df)
        self.transforms = transforms
        self.normalize = normalize

    def __getitem__(self, index):
        image, label = self.get_img(index)

        if self.transforms is not None:
            # image = transforms.ToTensor()(image)
            # image = image.type(torch.float32)
            image = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ])(image)

            image = self.normalize(image)

            image = transforms.ToPILImage()(image)
            image_layers = self.transforms(image)

            for i, img in enumerate(image_layers):
                # image_layers[i] = transforms.ToTensor()(img)
                image_layers[i] = transforms.Compose([
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                ])(img)

        image = torch.cat(image_layers, dim=0)

        return image, label

    def __len__(self):
        return self.df.shape[0]

    def get_img(self, index):
        img_path, label = self.df.iloc[index]
        return Image.open(img_path), label

    def encode_dataset(self, df):
        df['labels'] = df[self.label_list].apply(lambda x: (x.values == 1).astype(int), axis=1)
        return df[['file_name', 'labels']]

    def decode_labels(self, labels):
        return [self.label_list[i] for i, x in enumerate(labels) if x == 1]


def normalize_image(image):
    transform_list = [transforms.PILToTensor()]
    match image.size:
        case (4288, 2848):
            transform_list.append(transforms.Normalize(MEAN_4288_2848, STD_4288_2848))
        case (2144, 1424):
            transform_list.append(transforms.Normalize(MEAN_2144_1424, STD_2144_1424))
        case (2048, 1536):
            transform_list.append(transforms.Normalize(MEAN_2048_1536, STD_2048_1536))
        case _:
            transform_list.append(transforms.Normalize(MEAN, STD))

    return transforms.Compose(transform_list)(image)


def resize_image(image):
    if image.size == (4288, 2848):
        image = transforms.Resize(IMAGE_SIZE)(image)
        image = image.crop((50, 0, 50 + IMAGE_SIZE, IMAGE_SIZE))

    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.CenterCrop(IMAGE_SIZE)])(image)


def augment_image(image):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(1, 1)),
    ])(image)


def transform_image(image):
    image = resize_image(image)
    image = augment_image(image)

    image_eq = adjust_contrast(image, 1.2)
    image_eq = adjust_sharpness(image_eq, 1.5)
    image_eq = equalize(image_eq)
    image_eq = adjust_gamma(image_eq, 0.9)

    image_sharp = adjust_contrast(image, 1.5)
    image_sharp = adjust_sharpness(image_sharp, 1.5)
    image_sharp = posterize(image_sharp, 3)
    image_sharp = to_grayscale(image_sharp, num_output_channels=1)

    return [image, image_eq, image_sharp]

# Define the data module
class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)


# Define the model
class MultiLabelImageClassifierModel(pl.LightningModule):
    def __init__(self, num_classes, input_size, num_channels):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.BatchNorm2d(512),

            torch.nn.Flatten(),
            torch.nn.Linear(512 * (input_size // 32) * (input_size // 32), 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


# LightningModule
class LModule(pl.LightningModule):
    def __init__(self, model, label_list, lr=1e-3, epochs=10, data_loader=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.data_loader = data_loader

        self.f1_score = torchmetrics.F1Score(top_k=1, task="multiclass", num_classes=len(label_list), average='weighted')
        self.accuracy = torchmetrics.Accuracy(top_k=1,task="multiclass", num_classes=len(label_list), average='weighted')
        self.recall = torchmetrics.Recall(top_k=1,task="multiclass", num_classes=len(label_list), average='weighted')
        self.precision = torchmetrics.Precision(top_k=1,task="multiclass", num_classes=len(label_list), average='weighted')

        self.criterion = torch.nn.BCELoss()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                max_lr=self.lr,
                                                                epochs=self.epochs,
                                                                steps_per_epoch=len(self.data_loader))

        scheduler = {'scheduler': self.scheduler, 'interval': 'step',}

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.to(torch.float32)
        y_hat = y_hat.to(torch.float32)
        loss = self.criterion(y_hat, y)

        f1_score = self.f1_score(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        recall = self.recall(y_hat, y)


        logs = {'train_loss': loss, 'lr': self.optimizers().param_groups[0]['lr'], 'train_f1': f1_score, 'train_accuracy': accuracy, 'train_recall': recall}

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.to(torch.float32)
        y_hat = y_hat.to(torch.float32)
        loss = self.criterion(y_hat, y)

        f1_score = self.f1_score(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        recall = self.recall(y_hat, y)

        logs = {'val_loss': loss, 'val_f1': f1_score, 'val_accuracy': accuracy, 'val_recall': recall}

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.to(torch.float32)
        y_hat = y_hat.to(torch.float32)
        loss = self.criterion(y_hat, y)

        f1_score = self.f1_score(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        recall = self.recall(y_hat, y)

        logs = {'test_loss': loss, 'test_f1': f1_score, 'test_accuracy': accuracy, 'test_recall': recall}

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss
