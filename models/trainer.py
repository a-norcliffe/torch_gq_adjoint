"""
trainer class
"""


import torch
import numpy as np
import time


class nde_trainer:
    """
    Class to handle training of Neural DEs.
    Parameters
    ----------
    device : torch.device
    model : the model to be trained
    optimiser : one of torch.optim optimisers
    """
    def __init__(self, device, model, optimiser):
        self.device = device
        self.model = model
        self.optimiser = optimiser
        self.epochs = []
        self.epoch_loss_history = []
        self.epoch_val_metric_history = []
        self.epoch_time_history = []
        self.epoch_fnfe_history = []
        self.epoch_bnfe_history = []

    def train(self, train_loader, val_loader, loss_func, val_metric, nepochs, **model_kwargs):
        """
        Trains the Neural ODE and prints relevant information as it does, including time taken
        Parameters
        ----------
        train_loader: torch dataloader
            Dataloader of the training data
        val_loader: torch dataloader
            Dataloader of the validation data
        loss_func: function, must work with torch autograd
            The loss function
        val_metric: function
            A metric to test the model when we don't use loss, for example accuracy
        nepochs: int
            Number of epochs to train for
        model_kwargs: dict
            kwargs that are used for the model 
        """
        # collect values before training
        epoch_loss, epoch_val_metric = self.evaluate_epoch(val_loader, loss_func, val_metric, **model_kwargs)
        self.epoch_loss_history.append(epoch_loss)
        self.epoch_val_metric_history.append(epoch_val_metric)
        self.epoch_time_history.append(0.)
        self.epoch_fnfe_history.append(0.)
        self.epoch_bnfe_history.append(0.)
        self.epochs.append(0)

        # print information
        print('\n\nTraining:\n')
        print('Epoch: {}, ValLoss: {:.3f}, ValMetric: {:.3f}, Epoch Time: {}, FNFE: {}, BNFE: {}'.format(0,
                epoch_loss, epoch_val_metric, 0, '-', '-'))

        for epoch in range(1, nepochs+1):
            # train an epoch
            epoch_start_time = time.time()
            fnfe, bnfe = self.train_epoch(train_loader, loss_func, **model_kwargs)
            epoch_time = time.time() - epoch_start_time

            # evaluate the model after one epoch and save numbers
            epoch_loss, epoch_val_metric = self.evaluate_epoch(val_loader, loss_func, val_metric, **model_kwargs)
            self.epoch_loss_history.append(epoch_loss)
            self.epoch_val_metric_history.append(epoch_val_metric)
            self.epoch_time_history.append(epoch_time)
            self.epoch_fnfe_history.append(fnfe)
            self.epoch_bnfe_history.append(bnfe)
            self.epochs.append(epoch)

            # print information
            print('Epoch: {}, ValLoss: {:.3f}, ValMetric: {:.3f}, Epoch Time: {:.3f}, FNFE: {:.1f}, BNFE: {:.1f}'.format(epoch,
                    epoch_loss, epoch_val_metric, epoch_time, fnfe, bnfe))

        #print the end of training
        print('\nTraining Complete')
        total_time = np.cumsum(np.array(self.epoch_time_history))[-1]
        print('ValLoss: {:.3f}, ValMetric: {:.3f}, Total Time: {:.3f}, FNFE: {:.1f}, BNFE: {:.1f}'.format(epoch_loss,
                epoch_val_metric, total_time, fnfe, bnfe))

    def train_epoch(self, train_loader, loss_func, **model_kwargs):
        fnfe = 0
        bnfe = 0
        for x0, times, y in train_loader:
            self.model.defunc.nfe = 0
            self.optimiser.zero_grad()
            x0 = x0.to(self.device)
            times = times.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(x0, times, **model_kwargs)
            fnfe += self.model.defunc.nfe
            self.model.defunc.nfe = 0
            loss = loss_func(y_pred, y)
            loss.backward()
            self.optimiser.step()
            bnfe += self.model.defunc.nfe
        return fnfe/len(train_loader), bnfe/len(train_loader)

    def evaluate_epoch(self, val_loader, loss_func, val_metric, **model_kwargs):
        # if we batch the validation set for memory, must be a perfect divisor
        # the loss and metric also have to take averages for this to make sense
        loss = 0
        metric = 0
        for x0, times, y in val_loader:
            x0 = x0.to(self.device)
            times = times.to(self.device)
            y = y.to(self.device)
            y_pred = self.model.evaluate(x0, times, **model_kwargs)
            loss += loss_func(y_pred, y).item()
            metric += val_metric(y_pred, y).item()
        return loss/len(val_loader), metric/len(val_loader)