import os
import argparse
import logging
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import src.data.dataloader as dl
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

parser.add_argument('--save', type=str, default='./logs/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def adjust_sleep_data_to_mnist(data, to_image_dim=20):
    """
    Reshape: tensor.Size(batch_size, ch_count, sleep_epoch=3000) to tensor.Size(batch_size, ch_count, 20, 150)
    Args:
        data: x data from Dataloader
        to_image_dim: dimension in which we want to reshape 1D data to image like dimensions

    Returns:
        x from dataloader.__getitem__ method reshaped to fit MNIST model conv layers. Kernel can't be larger
        than x dimensions
    """
    if data.shape[1] > 1:
        data = data.unsqueeze(2)
    else:
        data = data.unsqueeze(1)
    # reshaping to fit conv layers kernel size
    last_dime_size = data.shape[2] * data.shape[3]

    data = torch.reshape(data, (data.shape[0], data.shape[1], to_image_dim, int(last_dime_size / to_image_dim)))

    return data


def get_data_loaders(batch_size, num_of_subjects=1):
    directory_path = args.data_dir
    print(directory_path)
    np_dataset = []
    for idx, np_name in enumerate(glob.glob(directory_path + '/*.np[yz]')):
        if idx >= num_of_subjects:
            print(f"Loaded {num_of_subjects} subjects")
            break
        print(f"loading {np_name}")
        np_dataset.append(np_name)

    train_loader, test_loader = dl.data_generator_np(subject_files=np_dataset,
                                                     batch_size=batch_size)

    return train_loader, test_loader, np_dataset


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


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


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

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
        x = adjust_sleep_data_to_mnist(data=x)
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), nof_classes)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
        total += len(y)
    return total_correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)

    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(utils.CustomFormatter())
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(utils.CustomFormatter())
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        pass
        # logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':

    # ================ Getting Data loaders ================================================================
    train_loader, test_loader, files_names = get_data_loaders(batch_size=args.batch_size)
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    num_of_classes = 5

    # ============== Configuring Logger ===================================================================
    makedirs(args.save)
    log_file = os.path.basename(args.save)
    if log_file == "":
        log_file = "_".join(files_names)
        log_file = os.path.basename(log_file)
        log_file = log_file.split(".", 1)[0]
    path_to_save_log = os.path.join(args.save, log_file)
    logger = get_logger(logpath=path_to_save_log, filepath=os.path.abspath(__file__))
    # ====================================================================================================

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"deivce: {device}, {torch.cuda.get_device_name(0)}")

    # ======================= Configuring Model down sampling layer ============================================
    is_odenet = args.network == 'odenet'

    if args.downsampling_method == 'conv':
        downsampling_layers = [
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
    feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, num_of_classes)]

    # Final model
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    # ======================== Training parameters =================================
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
    # ======================================= Training Routine ===========================
    for itr in range(args.nepochs * batches_per_epoch):

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = adjust_sleep_data_to_mnist(data=x)
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
    logger.info(f"Best Accuracy acheived for test {best_acc}")
    # Plotting accuracy vs epochs and saving figure
    epochs = range(len(test_acc_lst))
    plt.plot(epochs, train_acc_lst, 'g', label='Training accuracy')
    plt.plot(epochs, test_acc_lst, 'b', label='Test accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path_to_save_log)
