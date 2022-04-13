# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: colab,colab_type,id,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:ray-lightning]
#     language: python
#     name: conda-env-ray-lightning-py
# ---

# %% id="139b22c3"
import os

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64


# %% id="5fe164bb"
class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# %% [markdown] id="92211adb"
# By using the `Trainer` you automatically get:
# 1. Tensorboard logging
# 2. Model checkpointing
# 3. Training and validation loop
# 4. early-stopping

# %% id="9e9574d7" colab={"referenced_widgets": ["bd200cb848674c27a6ed8be04db5e92a", "e78bbfaafb00482b91f4eab0cddbdd2c", "4d981ca664bb427fbda7f999a4883276", "52a3cf8ba0894d47ba6d176eefd48bad", "194c70f182204bd0829e88a603d3c9c5"]}
# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)

# %% [markdown] id="6ec3ccda"
# ## MNIST Lightning Module Example
#
#
# ---
#
# ### Note what the following built-in functions are doing:
#
# 1. [prepare_data()](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#prepare-data) 💾
#     - This is where we can download the dataset. We point to our desired dataset and ask torchvision's `MNIST` dataset class to download if the dataset isn't found there.
#     - **Note we do not make any state assignments in this function** (i.e. `self.something = ...`)
#
# 2. [setup(stage)](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup) ⚙️
#     - Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).
#     - Setup expects a 'stage' arg which is used to separate logic for 'fit' and 'test'.
#     - If you don't mind loading all your datasets at once, you can set up a condition to allow for both 'fit' related setup and 'test' related setup to run whenever `None` is passed to `stage` (or ignore it altogether and exclude any conditionals).
#     - **Note this runs across all GPUs and it *is* safe to make state assignments here**
#
# 3. [x_dataloader()](https://pytorch-lightning.readthedocs.io/en/stable/api_references.html#core-api) ♻️
#     - `train_dataloader()`, `val_dataloader()`, and `test_dataloader()` all return PyTorch `DataLoader` instances that are created by wrapping their respective datasets that we prepared in `setup()`


# %%
class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size =  256,
        data_dir=PATH_DATASETS
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


# %% id="736c59b1"
class MNISTClassifier(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims


        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'loss' : loss}
    
    def training_epoch_end(self, outputs):

        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.logger.experiment.add_scalar(
            "Training/loss", mean_loss, self.current_epoch
        )
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", self.accuracy, prog_bar=True)
        return {'loss' : loss, 'accuracy' : acc}
    
    def validation_epoch_end(self, outputs):

        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        mean_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        

        self.logger.experiment.add_scalar(
            "Validation/loss", mean_loss, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Validation/accuracy", mean_acc, self.current_epoch
        )
        
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer




# %% id="20ca7128" colab={"referenced_widgets": ["d2c1cbb3130e4246b157582b25c0ec52", "b667094405d84527aa8cc17923a766de", "bff8dea71a504afd99c60da811ebb8fe", "c33a68b8444f47b596801f02c82f54b0", "6978fe74a7d44795836feb2fcdba92e5"]}
model = MNISTClassifier()
data_module = MNISTDataModule()
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
    progress_bar_refresh_rate=20,
)
trainer.fit(model,data_module)

# %% [markdown] id="c29dcb53"
# ### Testing
#
# To test a model, call `trainer.test(model)`.
#
# Or, if you've just trained a model, you can just call `trainer.test()` and Lightning will automatically
# test using the best saved checkpoint (conditioned on val_loss).

# %% id="25fd3ef0" colab={"referenced_widgets": ["d1699fdd99454d9a842f1fc0d4d3eddf"]}
trainer.test(model)

# %% [markdown] id="9ef7fe03"
# ### Bonus Tip
#
# You can keep calling `trainer.fit(model)` as many times as you'd like to continue training

# %% id="7729338b" colab={"referenced_widgets": ["a00c196e990949448d7a038e3c61a75d"]}
trainer.fit(model)

# %% [markdown] id="34adc78f"
# In Colab, you can use the TensorBoard magic function to view the logs that Lightning has created for you!

# %% id="8504f4ed"
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
