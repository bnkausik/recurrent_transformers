######################
#bnkausik at gmail.com
#code for the recurrent tranformers in the paper https://arxiv.org/abs/2402.14746
######################


########################################
#block recurrent transformer with forget
########################################




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
from config import training_config, GPTConfig
from block_model import GPT


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as td

# Define the transformation to convert to grayscale and then to a PyTorch Tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Converts the image to grayscale with 1 channel
    transforms.ToTensor()                         # Converts the image to a PyTorch Tensor
])

# Load the CIFAR-10 training dataset with the grayscale transform
train_data = datasets.CIFAR10(
    root='./data',        # Directory to store the dataset
    train=True,           # Load the training set
    download=True,        # Download the dataset if not already present
    transform=transform   # Apply the defined transformation
)

# Load the CIFAR-10 test dataset with the grayscale transform
test_data = datasets.CIFAR10(
    root='./data',        # Directory to store the dataset
    train=False,          # Load the test set
    download=True,        # Download the dataset if not already present
    transform=transform   # Apply the defined transformation
)




# Setup logging

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("using mps##################")
    ctx = torch.autocast(device_type="mps",dtype=torch.float16)
else:
    ctx = torch.autocast(device_type="cpu",dtype=torch.float16)
print(f'Using device: {device}')

# Define model
gptconfig = GPTConfig()
model = GPT(gptconfig)
model.to(device)

lr_0 = training_config["learning_rate"]
num_steps = training_config["num_steps"]
batch_size = (training_config['batch_size'])
log_interval = training_config["log_interval"]
optimizer = optim.Adam(model.parameters(), lr=lr_0)


def get_batch(data):
    for batch in td.DataLoader(data, batch_size=batch_size, shuffle=True):
        img, label = batch
        img = (torch.flatten(img, start_dim=1)*255).type(torch.int)
        label = label[:,None]
        return img, label


# Training function
def train():
    model.train()

    loss_arr = np.zeros(log_interval)
    acc_arr = np.zeros(log_interval)

    for step in range(num_steps):

        inputs, targets = get_batch(train_data)
        inputs = inputs.to(device)
        targets = targets.to(device)

        with ctx:
            outputs,loss = model(inputs,targets,gptconfig.w)

        optimizer.zero_grad()
        loss.backward()


        optimizer.step()



        loss_arr[step%log_interval] = loss.item()
        acc_arr[step%log_interval] = 100*(torch.sum(outputs == targets)/torch.numel(targets)).item()

        if step and ((not step % log_interval) or (step == num_steps-1)):

            mean_loss = loss_arr.mean()
            mean_acc = acc_arr.mean()

            cur_lr = optimizer.param_groups[0]['lr']

            sem_loss = np.std(loss_arr)/np.sqrt(log_interval)
            sem_acc = np.std(acc_arr)/np.sqrt(log_interval)

            print(f'{step+1} {gptconfig.w} Samples: {step*batch_size} train_loss: {mean_loss:.3f} sem_loss: {sem_loss:.3e} train_acc: {mean_acc:.2f} sem_acc: {sem_acc:.3e}  lr: {cur_lr:.2e} alpha {model.alpha.item():.2f}', end=' ')
            validate()
            model.train()
            if mean_loss <0.1: break
            if training_config['decay_flag']:
                cur_lr = lr_0*(1 - 0.9*step/num_steps)
                for p in optimizer.param_groups: p['lr'] = cur_lr
            sys.stdout.flush()




# Validation function
def validate():
    loss_arr = np.zeros(log_interval)
    acc_arr = np.zeros(log_interval)

    model.eval()
    for step in range(log_interval):
        inputs, targets = get_batch(test_data)
        inputs = inputs.to(device)
        targets = targets.to(device)

        with ctx:
            outputs,loss = model(inputs,targets,gptconfig.w)


        loss_arr[step] = loss.item()
        acc_arr[step] = 100*(torch.sum(outputs == targets)/torch.numel(targets)).item()

    mean_loss = loss_arr.mean()
    mean_acc = acc_arr.mean()
    sem_loss = np.std(loss_arr)/np.sqrt(log_interval)
    sem_acc = np.std(acc_arr)/np.sqrt(log_interval)
    print(f'vLoss: {mean_loss:.3f} sem_vloss: {sem_loss:.3e} vacc: {mean_acc:.2f} sem_vacc: {sem_acc:.3e}')

if __name__ == '__main__':
    print(training_config)
    print(vars(GPTConfig))
    sys.stdout.flush()
    train()



