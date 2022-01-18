import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.model import *
import src.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--data_dir', type=str, default=f'../../data/processed/')

parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=120)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--subject_count', type=int, default=None,
                    help="Limit the number of subjects we load from data folder")
parser.add_argument('--save', type=str, default='./logs/')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log_level', type=str, default="DEBUG", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


#    _____         _      _              __              _   _
#   |_   _| _ __ _(_)_ _ (_)_ _  __ _   / _|_  _ _ _  __| |_(_)___ _ _  ___
#     | || '_/ _` | | ' \| | ' \/ _` | |  _| || | ' \/ _|  _| / _ \ ' \(_-<
#     |_||_| \__,_|_|_||_|_|_||_\__, | |_|  \_,_|_||_\__|\__|_\___/_||_/__/
#                               |___/
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, lr, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, nof_classes=5):
    total_correct = 0
    total = 0
    for x, y in dataset_loader:
        x = adjust_sleep_data_dim(data=x)
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), nof_classes)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
        total += len(y)
    return total_correct / total


if __name__ == '__main__':

    #    _____          _
    #   |  __ \        | |
    #   | |  | |  __ _ | |_  __ _
    #   | |  | | / _` || __|/ _` |
    #   | |__| || (_| || |_| (_| |
    #   |_____/  \__,_| \__|\__,_|
    #
    #
    train_loader, test_loader, files_names = dl.get_data_loaders(batch_size=args.batch_size,
                                                                 directory_path=args.data_dir)
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    num_of_classes = 5

    # ============== Getting Logger ===================================================================
    logger, path_to_save_log = utils.get_logger(log_path=args.save,
                                                logfilenames=files_names,
                                                level=args.log_level)

    logger.info(f"Script Path: {os.path.abspath(__file__)}\n")
    # ====================================================================================================
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"deivce: {device}, {torch.cuda.get_device_name(0)}\n")

    #    __  __             _        _
    #   |  \/  |           | |      | |
    #   | \  / |  ___    __| |  ___ | |
    #   | |\/| | / _ \  / _` | / _ \| |
    #   | |  | || (_) || (_| ||  __/| |
    #   |_|  |_| \___/  \__,_| \___||_|
    #
    #
    # ======================= down sampling layer ============================================
    is_odenet = args.network == 'odenet'

    # TODO: get channel count from data dimensions
    channel_count = 10
    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(channel_count, 64, 3, 1),  # changed to 10 channels because 2 channels are devided to 5 bands each
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    # =================== Feature Layer and classifier Layer ==============================================
    feature_layers = [ODEBlock(ODEfunc(64), odesolver=odeint, tol=args.tol)] if is_odenet else [ResBlock(64, 64) for _
                                                                                                in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(),
                 nn.Linear(64, num_of_classes)]

    # Final model
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    logger.info(f"---Model")
    logger.info(model)
    logger.info('Number of parameters: {}\n'.format(count_parameters(model)))

    #    _______           _         _                 _____                                     _
    #   |__   __|         (_)       (_)               |  __ \                                   | |
    #      | | _ __  __ _  _  _ __   _  _ __    __ _  | |__) |__ _  _ __  __ _  _ __ ___    ___ | |_  ___  _ __  ___
    #      | || '__|/ _` || || '_ \ | || '_ \  / _` | |  ___// _` || '__|/ _` || '_ ` _ \  / _ \| __|/ _ \| '__|/ __|
    #      | || |  | (_| || || | | || || | | || (_| | | |   | (_| || |  | (_| || | | | | ||  __/| |_|  __/| |   \__ \
    #      |_||_|   \__,_||_||_| |_||_||_| |_| \__, | |_|    \__,_||_|   \__,_||_| |_| |_| \___| \__|\___||_|   |___/
    #                                           __/ |
    #                                          |___/
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()}
    ], weight_decay=0.1, lr=args.lr)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
    train_acc_lst = []
    test_acc_lst = []
    logger.info(f"---Training Hyper parameters")
    logger.info(f"Loss: {criterion}")
    logger.info(f"Optimizer: {optimizer}\n")

    #    _______           _         _
    #   |__   __|         (_)       (_)
    #      | | _ __  __ _  _  _ __   _  _ __    __ _
    #      | || '__|/ _` || || '_ \ | || '_ \  / _` |
    #      | || |  | (_| || || | | || || | | || (_| |
    #      |_||_|   \__,_||_||_| |_||_||_| |_| \__, |
    #                                           __/ |
    #                                          |___/
    logger.info("---Training")
    for itr in range(args.nepochs * batches_per_epoch):

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = adjust_sleep_data_dim(data=x)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():

                train_acc = accuracy(model, train_loader)
                val_acc = accuracy(model, test_loader)
                train_acc_lst.append(train_acc)
                test_acc_lst.append(val_acc)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )
    logger.info("-------")
    logger.info(f"Best Accuracy acheived for test {best_acc}")
    # Plotting accuracy vs epochs and saving figure
    epochs = list(range(len(test_acc_lst)))
    plt.plot(epochs, train_acc_lst, 'g', label='Training accuracy')
    plt.plot(epochs, test_acc_lst, 'b', label='Test accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path_to_save_log)
