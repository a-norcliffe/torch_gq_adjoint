"""
Common Functions used to run experiments
"""

import os
import os.path as osp

import torch
import numpy as np

from models.trainer import nde_trainer


def set_seed(x):
    # Used for consistency.
    x *= 1000
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_stats(folder, trainer, nparams):
    np.save(osp.join(folder, 'nparams.npy'), np.array([nparams]))
    np.save(osp.join(folder, 'epoch_times.npy'), np.array(trainer.epoch_time_history))
    np.save(osp.join(folder, 'epoch_loss_history.npy'), np.array(trainer.epoch_loss_history))
    np.save(osp.join(folder, 'epoch_val_metric_history.npy'), np.array(trainer.epoch_val_metric_history))
    np.save(osp.join(folder, 'epoch_fnfe_history.npy'), np.array(trainer.epoch_fnfe_history))
    np.save(osp.join(folder, 'epoch_bnfe_history.npy'), np.array(trainer.epoch_bnfe_history))
    np.save(osp.join(folder, 'epochs.npy'), np.array(trainer.epochs))


def set_device(x):
    if x == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + x if torch.cuda.is_available() else 'cpu')
    return device


def run_experiment(args, experiment_name, model, optimiser, train_loader, val_loader, loss_func, val_metric):
    """
    Function that will run the experiment
    Parameters
    ----------
    args: arguments that are given by the user
    experiment_name: str, used to create folder name for the results
    model: torch Module, the model that is being trained
    optimiser: torch Optimiser, used to train the model
    train_loader: torch data loader, the training data
    val_loader: torch data loader, the validation data
    loss_func: function (must work with torch autograd), loss function
    val_metric: function, additional metric to test the model apart from loss
    """
    # set device
    device = set_device(args.device)

    # make folder
    folder = osp.join('results/', experiment_name, str(args.width), args.adjoint_option, str(args.experiment_no))
    if not osp.exists(folder):
        os.makedirs(folder)

    # load model if necessary
    model = model.to(device)
    if args.load_model:
        print('\nLoading Model')
        model.load_state_dict(torch.load(osp.join(folder, 'trained_model.pth')))

    # trainer
    trainer = nde_trainer(device, model, optimiser)

    # get number of params
    nparams = count_parameters(model)
    
    # print relevant info
    print('\nDevice: {}, Parameters: {}, Adjoint Option: {}'.format(device, nparams, args.adjoint_option))

    # make kwargs for model
    kwargs = {'rtol': args.tol, 'atol': args.tol}
    if args.adjoint_option == 'adjoint_gq':
        kwargs['gtol'] = args.gtol

    # train model
    trainer.train(train_loader, val_loader, loss_func, val_metric, args.nepochs, **kwargs)

    # save training stats
    save_training_stats(folder, trainer, nparams)
    
    # save model if necessary
    if args.save_model:
        print('\nSaving Model')
        torch.save(model.state_dict(), osp.join(folder, 'trained_model.pth'))
        