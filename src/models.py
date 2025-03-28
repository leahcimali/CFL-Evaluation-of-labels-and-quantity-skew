import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(outputs, labels):
    """
    Computes the accuracy of the predictions.

    Args:
        outputs (torch.Tensor): The output predictions from the model, typically of shape (batch_size, num_classes).
        labels (torch.Tensor): The ground truth labels, typically of shape (batch_size).

    Returns:
        torch.Tensor: A single-element tensor containing the accuracy of the predictions as a float.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    """
    Base class for image classification models.
    Methods
    -------
    training_step(batch, device):
        Performs a training step by computing the loss for a batch of images and labels.
    validation_step(batch, device):
        Performs a validation step by computing the loss and accuracy for a batch of images and labels.
    validation_epoch_end(outputs):
        Computes the average loss and accuracy over all validation batches at the end of an epoch.
    epoch_end(epoch, result):
        Prints the training and validation loss and accuracy for the epoch.
    """
    def training_step(self, batch, device):
        images, labels = batch
        images, labels = images.to(device), labels.to(device).long() 
        out = self(images)
        loss = F.cross_entropy(out, labels)  
        return loss
    
    def validation_step(self, batch, device):
        images, labels = batch
        images, labels = images.to(device), labels.to(device).long()
        out = self(images)
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class GenericLinearModel(ImageClassificationBase):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        self.network = nn.Sequential(
            nn.Linear(in_size * in_size, 200),
            nn.Linear(200, 10)
        )
        
    def forward(self, xb):
        xb = xb.view(-1, self.in_size * self.in_size)
        return self.network(xb)

# Credit : https://github.com/Moddy2024/ResNet-9/blob/main/resnet-9.ipynb    
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class GenericConvModel(ImageClassificationBase):
    def __init__(self, in_size, n_channels,num_classes=10):
        super().__init__()
        num_classes = 10 

        self.conv1 = conv_block(n_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Simple CNN model for TissueMNIST/OctMNIST
class SimpleConvModel(ImageClassificationBase):
    def __init__(self, in_size, n_channels, num_classes=10):
        super().__init__()
        self.in_size = in_size
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)  # Input: in_size x in_size, Output: 16 x in_size x in_size
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: 32 x in_size x in_size
        self.pool = nn.MaxPool2d(2, 2)  # MaxPool: in_size x in_size → in_size/2 x in_size/2
        self.fc1 = nn.Linear(32 * (in_size // 2) * (in_size // 2), 128)  # Flattened to 128 units
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for classes

    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = self.pool(F.relu(self.conv2(xb)))
        xb = xb.view(-1, 32 * (self.in_size // 2) * (self.in_size // 2))  # Flatten
        xb = F.relu(self.fc1(xb))
        xb = self.fc2(xb)
        return xb