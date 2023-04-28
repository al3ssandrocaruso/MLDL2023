from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
import torch.utils.data
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb
from models.GRU import GRUClassifier

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name, dir=args.wandb_dir)
        wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]


def train(net,data_loader,optimizer,cost_function, device="cuda:0"):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.
    net.train() # Strictly needed if network contains layers which has different behaviours between train and test
    for batch_idx, (inputs, targets) in enumerate(data_loader): # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device) # Forward pass
        outputs = net(inputs) # Apply the loss
        loss = cost_function(outputs,targets) # Reset the optimizer
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        samples+=inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        cumulative_accuracy += predicted.eq(targets).sum().item()
    return cumulative_loss/samples, cumulative_accuracy/samples*100


def test(net, data_loader, cost_function, device="cuda:0"):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.
    net.eval() # Strictly needed if network contains layers which has different behaviours between train and test
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = cost_function(outputs,targets) # Reset the optimizer
            # Forward pass
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()
            samples+=inputs.shape[0]
    return cumulative_loss/samples, cumulative_accuracy/samples*100


def get_loss_function():
    loss_function = torch.nn.CrossEntropyLoss()
    return loss_function


def get_optimizer(net, lr, wd, momentum):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    return optimizer


def get_data(batch_size, test_batch_size=256):
    pass
    '''
    # Load data
    full_training_data = ???
    test_data = ???
    # Create train and validation splits
    num_samples = len(full_training_data)
    training_samples = int(num_samples*0.5+1)
    validation_samples = num_samples - training_samples
    training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples, validation_samples])
    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data, test_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
    '''


def main(batch_size=128, device='cuda:0', learning_rate=0.01, weight_decay=0.000001, momentum=0.9, epochs=5):
    train_loader, val_loader, test_loader = get_data(batch_size)
    # define the network
    net = GRUClassifier()
    net.to(device)
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)
    loss_function = get_loss_function()
    # loss_function.to(device)
    for e in range(epochs):
        train_loss, train_accuracy = train(net, train_loader, optimizer, loss_function)
        val_loss, val_accuracy = test(net, val_loader, loss_function)
        print('Epoch: {:d}'.format(e+1))
        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
        print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
        print("-----------------------------------------------------")
        print("After training:")
        train_loss, train_accuracy = test(net, train_loader, loss_function)
        val_loss, val_accuracy = test(net, val_loader, loss_function)
        test_loss, test_accuracy = test(net, test_loader, loss_function)
        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,train_accuracy))
        print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,val_accuracy))
        print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
        print('-----------------------------------------------------')


if __name__ == '__main__':
    main()