import torch
import torch.nn as nn
import numpy as np
import os
import time
import copy
import torch.optim as optim
from torchvision.models import vgg19
from torchvision import datasets, transforms

DEFAULT_PATH = "/Users/clem/Projets/prog/gif5000/dataset/"


class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # pretrained VGG19
        self.vgg = vgg19(pretrained=True)

        # network till the last conv layer (35th)
        self.features_conv = self.vgg.features[:36]

        # the max-pool in VGG19 that is after the last conv layer
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2,
                                           stride=2,
                                           padding=0,
                                           dilation=1,
                                           ceil_mode=False)

        # vgg's classifier
        self.classifier = self.vgg.classifier

        # extracted gradients
        self.gradients = None

        # data loader for training
        self.dataloaders_dict = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.dataset_loaded = False

    # hook

    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # VGG 19 till the last conv layer
        x = self.features_conv(x)
        # register the hook in the forward pass
        hook = x.register_hook(self.activation_hook)

        # continue finishing the VGG19
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x

    # extract gradient
    def get_activation_gradient(self):
        return self.gradients

    # extract the activation after the last ReLU
    def get_activation(self, x):
        return self.features_conv(x)

    def load_dataset(self, data_dir):
        # Data augmentation and normalization for training
        # Just normalization for validation
        batch_size = 8
        feature_extract = True
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
        }

        print("Initializing Datasets and Dataloaders... on path" + data_dir)

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val']}

        # Create training and validation dataloaders
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

        # Detect if we have a GPU available
        device = torch.device("cpu")
        # Send the model to GPU
        # model_ft = self.vgg.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = self.vgg.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in self.vgg.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.vgg.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        self.dataset_loaded = True

    def train_model(self, num_epochs=4, is_inception=False):
        since = time.time()
        model = self.vgg
        val_acc_history = []
        print("dataset loaded : " + str(self.dataset_loaded))
        if self.dataset_loaded == False:
            print("Loading dataset")
            self.load_dataset(DEFAULT_PATH)

        best_model_wts = copy.deepcopy(self.vgg.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders_dict[phase]:
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / \
                    len(self.dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double(
                ) / len(self.dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.vgg.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
